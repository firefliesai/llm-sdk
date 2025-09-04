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
 * OpenAI Responses API request parameters
 */
export interface OpenAIResponsesCreateParams {
  /**
   * The model to use for the response
   */
  model: string;
  /**
   * Input messages or prompt (replaces 'messages' in Responses API)
   */
  input?: string | Array<any>;
  /**
   * System instructions for the assistant
   */
  instructions?: string;
  /**
   * Tools available to the model
   */
  tools?: Array<any>;
  /**
   * Response format specification
   */
  response_format?: any;
  /**
   * Reasoning configuration
   */
  reasoning?: OpenAIReasoningOptions;
  /**
   * Whether to stream the response
   */
  stream?: boolean;
  /**
   * Background processing mode
   */
  background?: boolean;
  /**
   * Storage configuration
   */
  store?: boolean;
  /**
   * Fields to include in response
   */
  include?: string[];
  /**
   * Additional parameters
   */
  [key: string]: any;
}

/**
 * Base response event from OpenAI Responses API
 */
export interface OpenAIResponseEvent {
  type: string;
  [key: string]: any;
}

/**
 * Response created event
 */
export interface OpenAIResponseCreatedEvent extends OpenAIResponseEvent {
  type: "response.created";
  response: {
    id: string;
    object: "response";
    created_at: number;
    status: "in_progress" | "completed" | "failed";
  };
}

/**
 * Response in progress event
 */
export interface OpenAIResponseInProgressEvent extends OpenAIResponseEvent {
  type: "response.in_progress";
  response: {
    id: string;
    status: "in_progress";
  };
}

/**
 * Response completed event
 */
export interface OpenAIResponseCompletedEvent extends OpenAIResponseEvent {
  type: "response.completed";
  response: {
    id: string;
    status: "completed";
    output: Array<any>;
    usage?: any;
  };
}

/**
 * Reasoning summary part added event
 */
export interface OpenAIReasoningSummaryPartAddedEvent
  extends OpenAIResponseEvent {
  type: "response.reasoning_summary_part.added";
  part: {
    index: number;
    type: "reasoning_summary";
    content?: string;
  };
}

/**
 * Reasoning summary text delta event
 */
export interface OpenAIReasoningSummaryTextDeltaEvent
  extends OpenAIResponseEvent {
  type: "response.reasoning_summary_text.delta";
  delta: string;
  index: number;
}

/**
 * Output item added event
 */
export interface OpenAIOutputItemAddedEvent extends OpenAIResponseEvent {
  type: "response.output_item.added";
  item: {
    index: number;
    type: "message";
    role: "assistant";
    content?: Array<any>;
  };
}

/**
 * Output text delta event
 */
export interface OpenAIOutputTextDeltaEvent extends OpenAIResponseEvent {
  type: "response.output_text.delta";
  delta: string;
  index: number;
}

/**
 * Union type for all OpenAI Response events
 */
export type OpenAIResponseStreamEvent =
  | OpenAIResponseCreatedEvent
  | OpenAIResponseInProgressEvent
  | OpenAIResponseCompletedEvent
  | OpenAIReasoningSummaryPartAddedEvent
  | OpenAIReasoningSummaryTextDeltaEvent
  | OpenAIOutputItemAddedEvent
  | OpenAIOutputTextDeltaEvent;
