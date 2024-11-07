// This file is auto-generated by @hey-api/openapi-ts

/**
 * Represents a message generated by the model.
 */
export type AssistantMessage = {
    role: "assistant";
    content: Array<(TextPart | ToolCallPart | AudioPart)>;
};

/**
 * The container format of the audio.
 */
export type AudioContainer = 'wav' | 'ogg' | 'flac' | 'webm';

/**
 * The encoding of the audio.
 */
export type AudioEncoding = 'linear16' | 'flac' | 'mulaw' | 'alaw' | 'aac' | 'mp3' | 'opus';

/**
 * A part of the message that contains an audio.
 */
export type AudioPart = {
    type: "audio";
    container?: AudioContainer;
    /**
     * The base64-encoded audio data.
     */
    audioData: string;
    encoding?: AudioEncoding;
    /**
     * The sample rate of the audio. E.g. 44100, 48000.
     */
    sampleRate?: number;
    /**
     * The number of channels of the audio. E.g. 1, 2.
     */
    channels?: number;
    /**
     * The transcript of the audio.
     */
    transcript?: string;
};

export type AudioPartDelta = {
    type: "audio";
    /**
     * The base64-encoded audio data.
     */
    audioData?: string;
    container?: AudioContainer;
    encoding?: AudioEncoding;
    /**
     * The sample rate of the audio. E.g. 44100, 48000.
     */
    sampleRate?: number;
    /**
     * The number of channels of the audio. E.g. 1, 2.
     */
    channels?: number;
    /**
     * The transcript of the audio.
     */
    transcript?: string;
};

export type ContentDelta = {
    index: number;
    part: (TextPartDelta | ToolCallPartDelta | AudioPartDelta);
};

/**
 * A part of the message that contains an image.
 */
export type ImagePart = {
    type: "image";
    /**
     * The MIME type of the image. E.g. "image/jpeg", "image/png".
     */
    mimeType: string;
    /**
     * The base64-encoded image data.
     */
    imageData: string;
    /**
     * The width of the image in pixels.
     */
    width?: number;
    /**
     * The height of the image in pixels.
     */
    height?: number;
};

export type LanguageModelInput = {
    /**
     * A system prompt is a way of providing context and instructions to the model
     */
    systemPrompt?: string;
    /**
     * A list of messages comprising the conversation so far.
     */
    messages: Array<Message>;
    /**
     * Definitions of tools that the model may use.
     */
    tools?: Array<Tool>;
    /**
     * Determines how the model should choose which tool to use. "auto" - The model will automatically choose the tool to use or not use any tools. "none" - The model will not use any tools. "required" - The model will be forced to use a tool. { type: "tool", toolName: "toolName" } - The model will use the specified tool.
     */
    toolChoice?: (ToolChoiceAuto | ToolChoiceNone | ToolChoiceRequired | ToolChoiceTool);
    /**
     * The format that the model must output
     */
    responseFormat?: (ResponseFormatJson | ResponseFormatText);
    /**
     * The maximum number of tokens that can be generated in the chat completion.
     */
    maxTokens?: number;
    /**
     * Amount of randomness injected into the response. Ranges from 0.0 to 1.0
     */
    temperature?: number;
    /**
     * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass Ranges from 0.0 to 1.0
     */
    topP?: number;
    /**
     * Only sample from the top K options for each subsequent token. Used to remove "long tail" low probability responses. Ranges from 0.0 to 1.0
     */
    topK?: number;
    /**
     * Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
     */
    presencePenalty?: number;
    /**
     * Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
     */
    frequencyPenalty?: number;
    /**
     * The seed (integer), if set and supported by the model, to enable deterministic results.
     */
    seed?: number;
    /**
     * The modalities that the model should support.
     */
    modalities?: Array<Modality>;
    /**
     * Extra options that the model may support.
     */
    extra?: {
        [key: string]: unknown;
    };
};

export type Message = UserMessage | AssistantMessage | ToolMessage;

export type Modality = 'text' | 'audio';

/**
 * Represents the response generated by the model.
 */
export type ModelResponse = {
    content: Array<(TextPart | ToolCallPart | AudioPart)>;
    usage?: ModelUsage;
    /**
     * The cost of the response.
     */
    cost?: number;
};

/**
 * Represents the token usage of the model.
 */
export type ModelTokensDetail = {
    textTokens?: number;
    audioTokens?: number;
    imageTokens?: number;
};

/**
 * Represents the token usage of the model.
 */
export type ModelUsage = {
    inputTokens: number;
    outputTokens: number;
    inputTokensDetail?: ModelTokensDetail;
    outputTokensDetail?: ModelTokensDetail;
};

export type Part = TextPart | ImagePart | AudioPart | ToolCallPart | ToolResultPart;

export type PartialModelResponse = {
    delta: ContentDelta;
};

export type ResponseFormatJson = {
    type: "json";
    /**
     * The JSON schema of the response that the model must output. Not all models support this option.
     */
    schema?: {
        [key: string]: unknown;
    };
};

export type ResponseFormatText = {
    type: "text";
};

/**
 * A part of the message that contains text.
 */
export type TextPart = {
    type: "text";
    text: string;
};

export type TextPartDelta = {
    type: "text";
    text: string;
};

/**
 * Represents a tool that can be used by the model.
 */
export type Tool = {
    /**
     * The name of the tool.
     */
    name: string;
    /**
     * A description of the tool.
     */
    description: string;
    /**
     * The JSON schema of the parameters that the tool accepts. The type must be "object".
     */
    parameters: {
        [key: string]: unknown;
    } | null;
};

/**
 * A part of the message that represents a call to a tool the model wants to use.
 */
export type ToolCallPart = {
    type: "tool-call";
    /**
     * The ID of the tool call, used to match the tool result with the tool call.
     */
    toolCallId: string;
    /**
     * The name of the tool to call.
     */
    toolName: string;
    /**
     * The arguments to pass to the tool.
     */
    args: {
        [key: string]: unknown;
    } | null;
};

export type ToolCallPartDelta = {
    type: "tool-call";
    /**
     * The ID of the tool call, used to match the tool result with the tool call.
     */
    toolCallId?: string;
    /**
     * The name of the tool to call.
     */
    toolName?: string;
    /**
     * The partial JSON string of the arguments to pass to the tool.
     */
    args?: string;
};

/**
 * The model will automatically choose the tool to use or not use any tools.
 */
export type ToolChoiceAuto = {
    type: "auto";
};

/**
 * The model will not use any tools.
 */
export type ToolChoiceNone = {
    type: "none";
};

/**
 * The model will be forced to use a tool.
 */
export type ToolChoiceRequired = {
    type: "required";
};

/**
 * The model will use the specified tool.
 */
export type ToolChoiceTool = {
    type: "tool";
    toolName: string;
};

/**
 * Represents tool result in the message history.
 */
export type ToolMessage = {
    role: "tool";
    content: Array<ToolResultPart>;
};

/**
 * A part of the message that represents the result of a tool call.
 */
export type ToolResultPart = {
    type: "tool-result";
    /**
     * The ID of the tool call from previous assistant message.
     */
    toolCallId: string;
    /**
     * The name of the tool that was called.
     */
    toolName: string;
    /**
     * The result of the tool call.
     */
    result: ({
    [key: string]: unknown;
} | unknown[]);
    /**
     * Marks the tool result as an error.
     */
    isError?: boolean;
};

/**
 * Represents a message sent by the user.
 */
export type UserMessage = {
    role: "user";
    content: Array<(TextPart | ImagePart | AudioPart)>;
};