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
  convertToOpenAITool,
  convertToOpenAIToolChoice,
  convertToOpenAISamplingParams,
} from "./openai.js";
import type {
  OpenAIModelOptions,
  OpenAIResponsesOptions,
  ResponseCompletedEventWithUsage,
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
    const response = await this.openai.responses.create(params);

    return this.mapResponseToModelResponse(
      response as OpenAI.Responses.Response,
    );
  }

  /**
   * Create a streaming response using the OpenAI Responses API
   */
  async *streamResponse(
    input: LanguageModelInput,
    responsesOptions?: OpenAIResponsesOptions,
  ): AsyncGenerator<PartialModelResponse, ModelResponse> {
    const params = this.buildResponsesParams(
      input,
      responsesOptions,
    ) as OpenAI.Responses.ResponseCreateParamsStreaming;
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

    if (finalResponse?.response.usage) {
      const usage = this.mapUsage(finalResponse.response.usage);
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
        tools: input.tools.map((tool) =>
          convertToOpenAITool(tool, this.options),
        ),
      }),
      ...(input.toolChoice && {
        tool_choice: convertToOpenAIToolChoice(input.toolChoice),
      }),
      ...(input.reasoning && {
        reasoning: {
          ...(input.reasoning.effort && { effort: input.reasoning.effort }),
          ...(input.reasoning.summary && { summary: input.reasoning.summary }),
        },
      }),
      ...(input.reasoning?.maxTokens && {
        max_output_tokens: input.reasoning.maxTokens,
      }),
      ...(responsesOptions?.background && {
        background: responsesOptions.background,
      }),
      ...(typeof responsesOptions?.store === "boolean" && {
        store: responsesOptions.store,
      }),
      ...(responsesOptions?.include && { include: responsesOptions.include }),
    } as unknown as OpenAI.Responses.ResponseCreateParams;
  }

  /**
   * Convert messages to OpenAI Responses API format
   */
  private convertToResponsesAPIMessages(
    input: LanguageModelInput,
  ): OpenAI.Responses.ResponseInput {
    const responseMessages: OpenAI.Responses.EasyInputMessage[] = [];

    // Add system prompt if present
    if (input.systemPrompt) {
      responseMessages.push({
        role: "system",
        content: input.systemPrompt,
        type: "message",
      });
    }

    // Convert each message from our internal format
    for (const message of input.messages) {
      switch (message.role) {
        case "user": {
          const content: OpenAI.Responses.ResponseInputContent[] = [];

          for (const part of message.content) {
            switch (part.type) {
              case "text":
                content.push({
                  type: "input_text",
                  text: part.text,
                });
                break;
              case "image":
                content.push({
                  type: "input_image",
                  image_url: `data:${part.mimeType};base64,${part.imageData}`,
                  detail: "auto",
                });
                break;
              case "audio":
                // Convert audio to input_file format for Responses API
                content.push({
                  type: "input_file",
                  file_data: part.audioData,
                  filename: `audio.${part.container || "wav"}`,
                });
                break;
            }
          }

          responseMessages.push({
            role: "user",
            content,
            type: "message",
          });
          break;
        }
        case "assistant": {
          const content: OpenAI.Responses.ResponseInputContent[] = [];

          for (const part of message.content) {
            switch (part.type) {
              case "text":
                content.push({
                  type: "input_text",
                  text: part.text,
                });
                break;
              case "tool-call":
                // Tool calls in Responses API are handled differently
                // For now, we'll convert them to text descriptions
                content.push({
                  type: "input_text",
                  text: `Tool call: ${part.toolName}(${JSON.stringify(part.args)})`,
                });
                break;
              case "audio":
                // Audio outputs are not directly supported in input conversion
                if (part.transcript) {
                  content.push({
                    type: "input_text",
                    text: part.transcript,
                  });
                }
                break;
              case "reasoning":
                // Reasoning parts are not included in input conversion
                // They will be generated by the model in the response
                break;
            }
          }

          if (content.length > 0) {
            responseMessages.push({
              role: "assistant",
              content,
              type: "message",
            });
          }
          break;
        }
        case "tool": {
          // Tool results in Responses API need to be converted to function call outputs
          // For now, we'll represent them as assistant messages
          for (const toolResult of message.content) {
            responseMessages.push({
              role: "assistant",
              content: [
                {
                  type: "input_text",
                  text: `Tool result for ${toolResult.toolCallId}: ${JSON.stringify(toolResult.result)}`,
                },
              ],
              type: "message",
            });
          }
          break;
        }
      }
    }

    return responseMessages;
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
    for (const item of response.output) {
      // Handle reasoning items
      if (item.type === "reasoning") {
        const reasoningPart: ReasoningPart = {
          type: "reasoning",
          reasoning: item.summary.map((s) => s.text).join("\n"),
          summary: true, // Items in output are always summary
        };
        content.push(reasoningPart);
      }
      // Handle message items
      else if (item.type === "message") {
        const contentParts = item.content;
        for (const part of contentParts) {
          if (part.type === "output_text") {
            content.push({
              type: "text",
              text: part.text,
            });
          }
          // Handle other content types as needed
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
    const outputTokens = usage.output_tokens || 0;
    const reasoningTokens = usage.output_tokens_details.reasoning_tokens || 0;

    const result: ModelResponse["usage"] = {
      inputTokens: usage.input_tokens || 0,
      outputTokens,
    };

    if (reasoningTokens > 0) {
      result.outputTokensDetail = {
        reasoningTokens,
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
