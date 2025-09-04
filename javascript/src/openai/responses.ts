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
    const messages = this.convertToResponsesAPIMessages(input);

    return {
      model: this.options.modelId,
      input: messages,
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
   * Convert messages to OpenAI Responses API format
   */
  private convertToResponsesAPIMessages(input: LanguageModelInput): Array<any> {
    const messages = convertToOpenAIMessages(input, this.options);

    // Map content types for Responses API
    return messages.map((message: any) => {
      if (message.content) {
        // Handle string content by converting to input_text format
        if (typeof message.content === "string") {
          return {
            ...message,
            content: [
              {
                type: message.role === "user" ? "input_text" : "output_text",
                text: message.content,
              },
            ],
          };
        }

        // Handle array content
        if (Array.isArray(message.content)) {
          const mappedContent = message.content.map((contentPart: any) => {
            switch (contentPart.type) {
              case "text":
                return {
                  ...contentPart,
                  type: message.role === "user" ? "input_text" : "output_text",
                };
              case "image_url":
                return {
                  ...contentPart,
                  type: "input_image",
                };
              default:
                return contentPart;
            }
          });

          return {
            ...message,
            content: mappedContent,
          };
        }
      }

      return message;
    });
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
   * Make request to OpenAI Responses API using OpenAI SDK's native responses.create
   */
  private async makeResponsesRequest(
    params: OpenAIResponsesCreateParams,
  ): Promise<any> {
    // Use OpenAI SDK's native responses.create method
    return await this.openai.responses.create({
      ...params,
      stream: false,
    } as any);
  }

  /**
   * Make streaming request to OpenAI Responses API using OpenAI SDK's native responses.stream
   */
  private async makeResponsesStreamRequest(
    params: OpenAIResponsesCreateParams,
  ): Promise<AsyncIterable<OpenAIResponseStreamEvent>> {
    // Use OpenAI SDK's native responses.stream method
    const stream = this.openai.responses.stream({
      ...params,
      stream: true,
    } as any);

    // Convert OpenAI SDK stream events to our format
    return this.convertOpenAIStreamToOurFormat(stream);
  }

  /**
   * Convert OpenAI SDK stream events to our format
   */
  private async *convertOpenAIStreamToOurFormat(
    stream: any,
  ): AsyncGenerator<OpenAIResponseStreamEvent> {
    for await (const event of stream) {
      // The OpenAI SDK events should already be in the correct format,
      // but we can add any necessary transformations here
      yield event as OpenAIResponseStreamEvent;
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
    const reasoningTokens = usage.output_tokens_details?.reasoning_tokens || 0;
    const outputTokens = usage.output_tokens || 0;

    return {
      inputTokens: usage.input_tokens || 0,
      outputTokens: outputTokens,
      ...(reasoningTokens > 0 && {
        outputTokensDetail: {
          reasoningTokens: reasoningTokens,
          textTokens: outputTokens - reasoningTokens,
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
