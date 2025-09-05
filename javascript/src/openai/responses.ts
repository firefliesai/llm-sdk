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
    let finalResponse: Record<string, unknown> | undefined;

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

    const result: ModelResponse = {
      content: accumulator.computeContent(),
    };

    if (finalResponse && finalResponse["usage"]) {
      const usage = this.mapUsage(
        finalResponse["usage"] as Record<string, unknown>,
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
  ): OpenAIResponsesCreateParams {
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
          reasoning: this.mapReasoningOptions(input.reasoning),
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
    } as OpenAIResponsesCreateParams;
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
    params: OpenAIResponsesCreateParams,
  ): AsyncIterable<OpenAIResponseStreamEvent> {
    // Use OpenAI SDK's native responses.stream method
    const stream = this.openai.responses.stream(
      params as unknown as Parameters<typeof this.openai.responses.stream>[0],
    );

    // Convert OpenAI SDK stream events to our format
    return this.convertOpenAIStreamToOurFormat(stream);
  }

  /**
   * Convert OpenAI SDK stream events to our format
   */
  private async *convertOpenAIStreamToOurFormat(
    stream: AsyncIterable<unknown>,
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
      const usage = this.mapUsage(response["usage"] as Record<string, unknown>);
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
    usage: Record<string, unknown>,
  ): ModelResponse["usage"] | undefined {
    const outputTokensDetails = usage["output_tokens_details"] as
      | Record<string, unknown>
      | undefined;
    const reasoningTokens =
      (outputTokensDetails?.["reasoning_tokens"] as number) || 0;
    const outputTokens = (usage["output_tokens"] as number) || 0;

    const result: ModelResponse["usage"] = {
      inputTokens: (usage["input_tokens"] as number) || 0,
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
