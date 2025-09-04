import OpenAI from "openai";
import type {
  LanguageModelInput,
  ModelResponse,
  PartialModelResponse,
  ReasoningPart,
  ContentDelta,
  ReasoningPartDelta,
} from "../schema/index.js";
import { ContentDeltaAccumulator } from "../utils/stream.utils.js";
import {
  convertToOpenAIMessages,
  convertToOpenAITool,
  convertToOpenAIToolChoice,
  convertToOpenAISamplingParams,
} from "./openai.js";
import type {
  OpenAIModelOptions,
  OpenAIResponsesCreateParams,
  OpenAIResponseStreamEvent,
  OpenAIReasoningOptions,
  OpenAIResponsesOptions,
} from "./types.js";

/**
 * OpenAI Responses API client for handling thinking tokens and reasoning
 */
export class OpenAIResponsesClient {
  private openai: OpenAI;

  constructor(private options: OpenAIModelOptions) {
    this.openai = new OpenAI({
      baseURL: options.baseURL,
      apiKey: options.apiKey,
    });
  }

  /**
   * Create a response using the OpenAI Responses API
   */
  async createResponse(
    input: LanguageModelInput,
    responsesOptions?: OpenAIResponsesOptions,
  ): Promise<ModelResponse> {
    const params = this.buildResponsesParams(input, responsesOptions);
    params.stream = false;

    // Make request to OpenAI Responses API endpoint
    const response = await this.makeResponsesRequest(params);

    return this.mapResponseToModelResponse(response);
  }

  /**
   * Create a streaming response using the OpenAI Responses API
   */
  async *streamResponse(
    input: LanguageModelInput,
    responsesOptions?: OpenAIResponsesOptions,
  ): AsyncGenerator<PartialModelResponse, ModelResponse> {
    const params = this.buildResponsesParams(input, responsesOptions);
    params.stream = true;

    const stream = await this.makeResponsesStreamRequest(params);
    const accumulator = new ContentDeltaAccumulator();
    let finalResponse: any = null;

    for await (const event of stream) {
      const contentDeltas = this.mapEventToContentDeltas(event);

      if (contentDeltas.length > 0) {
        accumulator.addChunks(contentDeltas);

        for (const delta of contentDeltas) {
          yield { delta };
        }
      }

      // Store final response for usage information
      if (event.type === "response.completed") {
        finalResponse = event.response;
      }
    }

    return {
      content: accumulator.computeContent(),
      ...(finalResponse?.usage && {
        usage: this.mapUsage(finalResponse.usage),
      }),
    };
  }

  /**
   * Build parameters for OpenAI Responses API
   */
  private buildResponsesParams(
    input: LanguageModelInput,
    responsesOptions?: OpenAIResponsesOptions,
  ): OpenAIResponsesCreateParams {
    const messages = convertToOpenAIMessages(input, this.options);

    return {
      model: this.options.modelId,
      messages,
      ...(input.tools && {
        tools: input.tools.map((tool) =>
          convertToOpenAITool(tool, this.options),
        ),
      }),
      ...(input.toolChoice && {
        tool_choice: convertToOpenAIToolChoice(input.toolChoice),
      }),
      ...convertToOpenAISamplingParams(input),
      ...(input.reasoning && {
        reasoning: this.mapReasoningOptions(input.reasoning),
      }),
      ...(responsesOptions?.background && {
        background: responsesOptions.background,
      }),
      ...(typeof responsesOptions?.store === "boolean" && {
        store: responsesOptions.store,
      }),
      ...(responsesOptions?.include && { include: responsesOptions.include }),
    };
  }

  /**
   * Map SDK reasoning options to OpenAI reasoning options
   */
  private mapReasoningOptions(
    reasoning: NonNullable<LanguageModelInput["reasoning"]>,
  ): OpenAIReasoningOptions {
    return {
      ...(reasoning.effort && { effort: reasoning.effort }),
      ...(reasoning.maxTokens && { max_tokens: reasoning.maxTokens }),
      ...(typeof reasoning.exclude === "boolean" && {
        exclude: reasoning.exclude,
      }),
      ...(reasoning.summary && { summary: reasoning.summary }),
    };
  }

  /**
   * Map OpenAI response events to content deltas
   */
  private mapEventToContentDeltas(
    event: OpenAIResponseStreamEvent,
  ): ContentDelta[] {
    const contentDeltas: ContentDelta[] = [];

    switch (event.type) {
      case "response.reasoning_summary_text.delta": {
        const part: ReasoningPartDelta = {
          type: "reasoning",
          reasoning: event.delta,
          summary: true,
        };
        contentDeltas.push({
          index: event.index || 0,
          part,
        });
        break;
      }
      case "response.output_text.delta": {
        contentDeltas.push({
          index: event.index || 0,
          part: {
            type: "text",
            text: event.delta,
          },
        });
        break;
      }
      // Handle other event types as needed
    }

    return contentDeltas;
  }

  /**
   * Make request to OpenAI Responses API
   */
  private async makeResponsesRequest(
    params: OpenAIResponsesCreateParams,
  ): Promise<any> {
    // Since OpenAI SDK might not have responses client yet, we'll use the underlying fetch
    const response = await fetch(
      `${this.openai.baseURL || "https://api.openai.com"}/v1/responses`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.options.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      },
    );

    if (!response.ok) {
      throw new Error(`OpenAI Responses API error: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Make streaming request to OpenAI Responses API
   */
  private async makeResponsesStreamRequest(
    params: OpenAIResponsesCreateParams,
  ): Promise<AsyncIterable<OpenAIResponseStreamEvent>> {
    const response = await fetch(
      `${this.openai.baseURL || "https://api.openai.com"}/v1/responses`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.options.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      },
    );

    if (!response.ok) {
      throw new Error(`OpenAI Responses API error: ${response.statusText}`);
    }

    return this.parseServerSentEvents(response);
  }

  /**
   * Parse Server-Sent Events from the response stream
   */
  private async *parseServerSentEvents(
    response: Response,
  ): AsyncGenerator<OpenAIResponseStreamEvent> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Failed to get response body reader");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") return;

            try {
              const event = JSON.parse(data) as OpenAIResponseStreamEvent;
              yield event;
            } catch (error) {
              console.warn("Failed to parse SSE data:", data, error);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Map OpenAI response to SDK ModelResponse
   */
  private mapResponseToModelResponse(response: any): ModelResponse {
    const content: ModelResponse["content"] = [];

    // Handle reasoning content
    if (response.reasoning) {
      const reasoningPart: ReasoningPart = {
        type: "reasoning",
        reasoning: response.reasoning.summary || response.reasoning.content,
        summary: !!response.reasoning.summary,
      };
      content.push(reasoningPart);
    }

    // Handle output content
    if (response.output && Array.isArray(response.output)) {
      for (const item of response.output) {
        if (item.type === "message" && item.content) {
          for (const part of item.content) {
            if (part.type === "text") {
              content.push({
                type: "text",
                text: part.text,
              });
            }
            // Handle other content types as needed
          }
        }
      }
    }

    return {
      content,
      ...(response.usage && { usage: this.mapUsage(response.usage) }),
    };
  }

  /**
   * Map OpenAI usage to SDK usage format
   */
  private mapUsage(usage: any): ModelResponse["usage"] {
    return {
      inputTokens: usage.prompt_tokens || 0,
      outputTokens: usage.completion_tokens || 0,
      ...(usage.reasoning_tokens && {
        outputTokensDetail: {
          reasoningTokens: usage.reasoning_tokens,
          textTokens: usage.completion_tokens - (usage.reasoning_tokens || 0),
        },
      }),
    };
  }
}

/**
 * Extended OpenAI Language Model Input that supports Responses API
 */
export type OpenAIResponsesLanguageModelInput = LanguageModelInput & {
  responsesOptions?: OpenAIResponsesOptions;
};
