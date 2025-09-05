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
  OpenAIResponsesOptions,
  ResponseCompletedEventWithUsage,
} from "./types.js";

// Note: Using type assertions to access the responses API since the SDK types may not be fully compatible

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
    const response = await this.openai.responses.create(params);

    return this.mapResponseToModelResponse(response as OpenAI.Responses.Response);
  }

  /**
   * Create a streaming response using the OpenAI Responses API
   */
  async *streamResponse(
    input: LanguageModelInput,
    responsesOptions?: OpenAIResponsesOptions,
  ): AsyncGenerator<PartialModelResponse, ModelResponse> {
    const params = this.buildResponsesParams(input, responsesOptions) as OpenAI.Responses.ResponseCreateParamsStreaming;
    params.stream = true;

    const stream = this.openai.responses.stream(params);
    const accumulator = new ContentDeltaAccumulator();
    let finalResponse: ResponseCompletedEventWithUsage | undefined;

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
        finalResponse = event as ResponseCompletedEventWithUsage;
      }
    }

    const result: ModelResponse = {
      content: accumulator.computeContent(),
    };

    if (finalResponse && finalResponse.response.usage) {
      const usage = this.mapUsage(
        finalResponse.response.usage,
      );
      if (usage) {
        result.usage = usage;
      }
    }

    return result;
  }

  /**
   * Build parameters for OpenAI Responses API
   */
  private buildResponsesParams(
    input: LanguageModelInput,
    responsesOptions?: OpenAIResponsesOptions,
  ): OpenAI.Responses.ResponseCreateParams {
    return {
      model: this.options.modelId,
      input: this.convertToResponsesAPIMessages(input),
      ...convertToOpenAISamplingParams(input),
      ...(input.tools && {
        tools: input.tools.map((tool) => convertToOpenAITool(tool, this.options)),
      }),
      ...(input.toolChoice && {
        tool_choice: convertToOpenAIToolChoice(input.toolChoice),
      }),
      ...(input.reasoning && {
        reasoning: {
          ...(input.reasoning.effort && { effort: input.reasoning.effort }),
          ...(input.reasoning.summary && { summary: input.reasoning.summary }),
        },
        ...(input.reasoning.maxTokens && { max_output_tokens: input.reasoning.maxTokens }),
      }),
      ...(responsesOptions?.background && { background: responsesOptions.background }),
      ...(typeof responsesOptions?.store === "boolean" && { store: responsesOptions.store }),
      ...(responsesOptions?.include && { include: responsesOptions.include }),
    } as unknown as OpenAI.Responses.ResponseCreateParams;
  }

  /**
   * Convert messages to OpenAI Responses API format
   */
  private convertToResponsesAPIMessages(
    input: LanguageModelInput,
  ): OpenAI.Responses.ResponseInput[] {
    const messages = convertToOpenAIMessages(input, this.options);

    
    return messages.map((message: any) => {
      if (!message.content) return message;

      const isUser = message.role === "user";
      const textType = isUser ? "input_text" : "output_text";

      // Handle string content
      if (typeof message.content === "string") {
        return {
          ...message,
          content: [{ type: textType, text: message.content }],
        };
      }

      // Handle array content
      if (Array.isArray(message.content)) {
        return {
          ...message,
          content: message.content.map((part: any) => {
            switch (part.type) {
              case "text":
                return { ...part, type: textType };
              case "image_url":
                return { ...part, type: "input_image" };
              default:
                return part;
            }
          }),
        };
      }

      return message;
    }) as OpenAI.Responses.ResponseInput[];
  }

  /**
   * Map OpenAI response events to content deltas
   */
  private mapEventToContentDeltas(
    event: OpenAI.Responses.ResponseStreamEvent,
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
          index: event.output_index || 0,
          part,
        });
        break;
      }
      case "response.output_text.delta": {
        contentDeltas.push({
          index: event.output_index || 0,
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
   * Map OpenAI response to SDK ModelResponse
   */
  private mapResponseToModelResponse(
    response: OpenAI.Responses.Response,
  ): ModelResponse {
    const content: ModelResponse["content"] = [];

    // Handle output content
    if (response.output && Array.isArray(response.output)) {
      for (const item of response.output) {
        // Handle reasoning items
        if (item.type === "reasoning") {
          const reasoningItem = item; // ResponseReasoningItem type
          if (reasoningItem.summary && Array.isArray(reasoningItem.summary)) {
            // Combine all summary texts
            const reasoningText = reasoningItem.summary
              .map(s => s.text)
              .join('\n');
            
            const reasoningPart: ReasoningPart = {
              type: "reasoning",
              reasoning: reasoningText,
              summary: true, // Items in output are always summary
            };
            content.push(reasoningPart);
          }
        }
        // Handle message items
        else if (item.type === "message" && item.content) {
          const contentParts = item.content;
          for (const part of contentParts) {
            if (part.type === "output_text") {
              content.push({
                type: "text",
                text: part.text as string,
              });
            }
            // Handle other content types as needed
          }
        }
      }
    }

    const result: ModelResponse = {
      content,
    };

    if (response.usage) {
      const usage = this.mapUsage(response.usage);
      if (usage) {
        result.usage = usage;
      }
    }

    return result;
  }

  /**
   * Map OpenAI usage to SDK usage format
   */
  private mapUsage(
    usage: OpenAI.Responses.ResponseUsage,
  ): ModelResponse["usage"] | undefined {
    const reasoningTokens = usage.output_tokens_details.reasoning_tokens || 0;
    const outputTokens = usage.output_tokens || 0;

    const result: ModelResponse["usage"] = {
      inputTokens: usage.input_tokens || 0,
      outputTokens: outputTokens,
    };

    if (reasoningTokens > 0) {
      result.outputTokensDetail = {
        reasoningTokens: reasoningTokens,
        textTokens: outputTokens - reasoningTokens,
      };
    }

    return result;
  }
}

/**
 * Extended OpenAI Language Model Input that supports Responses API
 */
export type OpenAIResponsesLanguageModelInput = LanguageModelInput & {
  responsesOptions?: OpenAIResponsesOptions;
};
