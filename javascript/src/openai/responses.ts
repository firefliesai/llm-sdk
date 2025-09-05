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

    const stream = this.makeResponsesStreamRequest(params);
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
    const messages = this.convertToResponsesAPIMessages(input);

    const baseParams = {
      model: this.options.modelId,
      input: messages,
    };

    const toolParams = input.tools
      ? {
          tools: input.tools.map((tool) =>
            convertToOpenAITool(tool, this.options),
          ),
        }
      : {};

    const toolChoiceParams = input.toolChoice
      ? {
          tool_choice: convertToOpenAIToolChoice(input.toolChoice),
        }
      : {};

    const samplingParams = convertToOpenAISamplingParams(input);

    const reasoningParams = input.reasoning
      ? {
          reasoning: {
            ...(input.reasoning.effort && { effort: input.reasoning.effort }),
            ...(input.reasoning.summary && { summary: input.reasoning.summary }),
          },
          // Handle maxTokens at top level as per OpenAI SDK
          ...(input.reasoning.maxTokens && { max_output_tokens: input.reasoning.maxTokens }),
        }
      : {};

    const backgroundParams = responsesOptions?.background
      ? {
          background: responsesOptions.background,
        }
      : {};

    const storeParams =
      typeof responsesOptions?.store === "boolean"
        ? {
            store: responsesOptions.store,
          }
        : {};

    const includeParams = responsesOptions?.include
      ? { include: responsesOptions.include }
      : {};

    return {
      ...baseParams,
      ...toolParams,
      ...toolChoiceParams,
      ...samplingParams,
      ...reasoningParams,
      ...backgroundParams,
      ...storeParams,
      ...includeParams,
    } as unknown as OpenAI.Responses.ResponseCreateParams;
  }

  /**
   * Convert messages to OpenAI Responses API format
   */
  private convertToResponsesAPIMessages(
    input: LanguageModelInput,
  ): Array<Record<string, unknown>> {
    const messages = convertToOpenAIMessages(input, this.options);

    // Map content types for Responses API
    return (messages as unknown as Record<string, unknown>[]).map(
      (message: Record<string, unknown>) => {
        if (message["content"]) {
          // Handle string content by converting to input_text format
          if (typeof message["content"] === "string") {
            return {
              ...message,
              content: [
                {
                  type:
                    message["role"] === "user" ? "input_text" : "output_text",
                  text: message["content"],
                },
              ],
            };
          }

          // Handle array content
          if (Array.isArray(message["content"])) {
            const mappedContent = (
              message["content"] as Record<string, unknown>[]
            ).map((contentPart: Record<string, unknown>) => {
              switch (contentPart["type"]) {
                case "text":
                  return {
                    ...contentPart,
                    type:
                      message["role"] === "user" ? "input_text" : "output_text",
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
      },
    );
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
   * Make request to OpenAI Responses API using OpenAI SDK's native responses.create
   */
  private async makeResponsesRequest(
    params: OpenAI.Responses.ResponseCreateParams,
  ): Promise<Record<string, unknown>> {
    // Use OpenAI SDK's native responses.create method
    const response = await this.openai.responses.create(
      params as unknown as Parameters<typeof this.openai.responses.create>[0],
    );
    return response as unknown as Record<string, unknown>;
  }

  /**
   * Make streaming request to OpenAI Responses API using OpenAI SDK's native responses.stream
   */
  private makeResponsesStreamRequest(
    params: OpenAI.Responses.ResponseCreateParams,
  ): AsyncIterable<OpenAI.Responses.ResponseStreamEvent> {
    // Use OpenAI SDK's native responses.stream method - no conversion needed
    return this.openai.responses.stream(
      params as unknown as Parameters<typeof this.openai.responses.stream>[0],
    );
  }

  /**
   * Map OpenAI response to SDK ModelResponse
   */
  private mapResponseToModelResponse(
    response: Record<string, unknown>,
  ): ModelResponse {
    const content: ModelResponse["content"] = [];

    // Handle reasoning content
    if (response["reasoning"]) {
      const reasoning = response["reasoning"] as Record<string, unknown>;
      const reasoningPart: ReasoningPart = {
        type: "reasoning",
        reasoning: (reasoning["summary"] || reasoning["content"]) as string,
        summary: !!reasoning["summary"],
      };
      content.push(reasoningPart);
    }

    // Handle output content
    if (response["output"] && Array.isArray(response["output"])) {
      const outputItems = response["output"] as Record<string, unknown>[];
      for (const item of outputItems) {
        if (item["type"] === "message" && item["content"]) {
          const contentParts = item["content"] as Record<string, unknown>[];
          for (const part of contentParts) {
            if (part["type"] === "text") {
              content.push({
                type: "text",
                text: part["text"] as string,
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

    if (response["usage"]) {
      const usage = this.mapUsage(response["usage"] as OpenAI.Responses.ResponseUsage);
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
