import OpenAI from "openai";

export interface OpenAIModelOptions {
  baseURL?: string;
  apiKey: string;
  modelId: string;
  /**
   * If true, use OpenAI new structured outputs feature available on new models.
   * https://platform.openai.com/docs/guides/structured-outputs
   */
  structuredOutputs?: boolean;
  /**
   * If true, for assistant audio parts, convert them to text parts if they have a transcript.
   *
   * For Audio objects, OpenAI provides message.audio.id that we can send back
   * https://platform.openai.com/docs/guides/audio#multi-turn-conversations
   *
   * However, we do not support the id field, and the audio id expires in 4 hours anyway.
   *
   * Therefore, we can convert the audio to text and send it back to OpenAI instead.
   * Setting this option to "false" currently throws an error if there is an assistant audio part.
   */
  convertAudioPartsToTextParts?: boolean;
}

// documented type is wrong
export type OpenAIPatchedPromptTokensDetails = {
  cached_tokens: number;
  text_tokens: number;
  image_tokens: number;
  audio_tokens: number;
};

// documented type is wrong
export type OpenAIPatchedCompletionTokenDetails = {
  reasoning_tokens: number;
  text_tokens: number;
  audio_tokens: number;
};

/**
 * Configuration for OpenAI reasoning behavior
 */
export interface OpenAIReasoningOptions {
  /**
   * Effort level for reasoning: "low", "medium", or "high"
   * Controls the depth of reasoning provided
   */
  effort?: "low" | "medium" | "high";
  /**
   * Maximum number of reasoning tokens to generate
   */
  max_tokens?: number;
  /**
   * Whether to include reasoning tokens in the response
   * Default: false (include reasoning tokens)
   */
  exclude?: boolean;
  /**
   * Whether to generate automatic reasoning summaries
   * Can be "auto" for automatic summaries
   */
  summary?: "auto";
}

/**
 * Configuration options for OpenAI Responses API
 */
export interface OpenAIResponsesOptions {
  /**
   * Reasoning configuration for thinking tokens
   */
  reasoning?: OpenAIReasoningOptions;
  /**
   * Whether to process the request in background mode for long-running tasks
   */
  background?: boolean;
  /**
   * Whether to store the response (affects ZDR eligibility)
   */
  store?: boolean;
  /**
   * Fields to include in the response (e.g., ["reasoning.encrypted_content"])
   */
  include?: string[];
}

/**
 * Union type for all OpenAI Response events
 */
// OpenAIResponseStreamEvent is now replaced with OpenAI.Responses.ResponseStreamEvent from the official SDK
export type OpenAIResponseStreamEvent = OpenAI.Responses.ResponseStreamEvent;

/**
 * Extended ResponseCompletedEvent with easier access to usage information
 * This extends the official OpenAI type to provide better TypeScript support
 */
export interface ResponseCompletedEventWithUsage extends OpenAI.Responses.ResponseCompletedEvent {
  response: OpenAI.Responses.Response & {
    usage: OpenAI.Responses.ResponseUsage;
  };
}
