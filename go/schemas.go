// Package llm_sdk provides primitives to interact with the openapi HTTP API.
//
// Code generated by github.com/oapi-codegen/oapi-codegen/v2 version v2.3.0 DO NOT EDIT.
package llm_sdk

import (
	"encoding/json"

	"github.com/oapi-codegen/runtime"
)

// AssistantMessage Represents a message generated by the model.
type AssistantMessage struct {
	Content []AssistantMessage_Content_Item `json:"content"`
	Role    string                          `json:"role"`
}

// AssistantMessage_Content_Item defines model for AssistantMessage.content.Item.
type AssistantMessage_Content_Item struct {
	union json.RawMessage
}

// ContentDelta defines model for ContentDelta.
type ContentDelta struct {
	Index int               `json:"index"`
	Part  ContentDelta_Part `json:"part"`
}

// ContentDelta_Part defines model for ContentDelta.Part.
type ContentDelta_Part struct {
	union json.RawMessage
}

// ImagePart A part of the message that contains an image.
type ImagePart struct {
	// ImageData The base64-encoded image data.
	ImageData string `json:"imageData"`

	// MimeType The MIME type of the image. E.g. "image/jpeg", "image/png".
	MimeType string `json:"mimeType"`
	Type     string `json:"type"`
}

// LanguageModelInput defines model for LanguageModelInput.
type LanguageModelInput struct {
	// FrequencyPenalty Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
	FrequencyPenalty *float32 `json:"frequencyPenalty,omitempty"`

	// MaxTokens The maximum number of tokens that can be generated in the chat completion.
	MaxTokens *int `json:"maxTokens,omitempty"`

	// Messages A list of messages comprising the conversation so far.
	Messages []Message `json:"messages"`

	// PresencePenalty Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
	PresencePenalty *float32 `json:"presencePenalty,omitempty"`

	// ResponseFormat The format that the model must output
	ResponseFormat *LanguageModelInput_ResponseFormat `json:"responseFormat,omitempty"`

	// Seed The seed (integer), if set and supported by the model, to enable deterministic results.
	Seed *int `json:"seed,omitempty"`

	// SystemPrompt A system prompt is a way of providing context and instructions to the model
	SystemPrompt *string `json:"systemPrompt,omitempty"`

	// Temperature Amount of randomness injected into the response. Ranges from 0.0 to 1.0
	Temperature *float32 `json:"temperature,omitempty"`

	// ToolChoice Determines how the model should choose which tool to use. "auto" - The model will automatically choose the tool to use or not use any tools. "none" - The model will not use any tools. "required" - The model will be forced to use a tool. { type: "tool", toolName: "toolName" } - The model will use the specified tool.
	ToolChoice *LanguageModelInput_ToolChoice `json:"toolChoice,omitempty"`

	// Tools Definitions of tools that the model may use.
	Tools *[]Tool `json:"tools,omitempty"`

	// TopK Only sample from the top K options for each subsequent token. Used to remove "long tail" low probability responses. Ranges from 0.0 to 1.0
	TopK *float32 `json:"topK,omitempty"`

	// TopP An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass Ranges from 0.0 to 1.0
	TopP *float32 `json:"topP,omitempty"`
}

// LanguageModelInput_ResponseFormat The format that the model must output
type LanguageModelInput_ResponseFormat struct {
	union json.RawMessage
}

// LanguageModelInput_ToolChoice Determines how the model should choose which tool to use. "auto" - The model will automatically choose the tool to use or not use any tools. "none" - The model will not use any tools. "required" - The model will be forced to use a tool. { type: "tool", toolName: "toolName" } - The model will use the specified tool.
type LanguageModelInput_ToolChoice struct {
	union json.RawMessage
}

// Message defines model for Message.
type Message struct {
	union json.RawMessage
}

// ModelResponse Represents the response generated by the model.
type ModelResponse struct {
	Content []ModelResponse_Content_Item `json:"content"`

	// Usage Represents the token usage of the model.
	Usage *ModelUsage `json:"usage,omitempty"`
}

// ModelResponse_Content_Item defines model for ModelResponse.content.Item.
type ModelResponse_Content_Item struct {
	union json.RawMessage
}

// ModelUsage Represents the token usage of the model.
type ModelUsage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
}

// PartialModelResponse defines model for PartialModelResponse.
type PartialModelResponse struct {
	Delta ContentDelta `json:"delta"`
}

// ResponseFormatJson defines model for ResponseFormatJson.
type ResponseFormatJson struct {
	// Schema The JSON schema of the response that the model must output. Not all models support this option.
	Schema *map[string]interface{} `json:"schema,omitempty"`
	Type   string                  `json:"type"`
}

// ResponseFormatText defines model for ResponseFormatText.
type ResponseFormatText struct {
	Type string `json:"type"`
}

// TextPart A part of the message that contains text.
type TextPart struct {
	Text string `json:"text"`
	Type string `json:"type"`
}

// Tool Represents a tool that can be used by the model.
type Tool struct {
	// Description A description of the tool.
	Description string `json:"description"`

	// Name The name of the tool.
	Name string `json:"name"`

	// Parameters The JSON schema of the parameters that the tool accepts. The type must be "object".
	Parameters map[string]interface{} `json:"parameters"`
}

// ToolCallPart A part of the message that represents a call to a tool the model wants to use.
type ToolCallPart struct {
	// Args The arguments to pass to the tool.
	Args *map[string]interface{} `json:"args,omitempty"`

	// ToolCallId The ID of the tool call, used to match the tool result with the tool call.
	ToolCallId string `json:"toolCallId"`

	// ToolName The name of the tool to call.
	ToolName string `json:"toolName"`
	Type     string `json:"type"`
}

// ToolChoiceAuto The model will automatically choose the tool to use or not use any tools.
type ToolChoiceAuto struct {
	Type string `json:"type"`
}

// ToolChoiceNone The model will not use any tools.
type ToolChoiceNone struct {
	Type string `json:"type"`
}

// ToolChoiceRequired The model will be forced to use a tool.
type ToolChoiceRequired struct {
	Type string `json:"type"`
}

// ToolChoiceTool The model will use the specified tool.
type ToolChoiceTool struct {
	ToolName string `json:"toolName"`
	Type     string `json:"type"`
}

// ToolMessage Represents tool result in the message history.
type ToolMessage struct {
	Content []ToolResultPart `json:"content"`
	Role    string           `json:"role"`
}

// ToolResultPart A part of the message that represents the result of a tool call.
type ToolResultPart struct {
	// IsError Marks the tool result as an error.
	IsError *bool `json:"isError,omitempty"`

	// Result The result of the tool call.
	Result map[string]interface{} `json:"result"`

	// ToolCallId The ID of the tool call from previous assistant message.
	ToolCallId string `json:"toolCallId"`

	// ToolName The name of the tool that was called.
	ToolName string `json:"toolName"`
	Type     string `json:"type"`
}

// UserMessage Represents a message sent by the user.
type UserMessage struct {
	Content []UserMessage_Content_Item `json:"content"`
	Role    string                     `json:"role"`
}

// UserMessage_Content_Item defines model for UserMessage.content.Item.
type UserMessage_Content_Item struct {
	union json.RawMessage
}

// AsTextPart returns the union data inside the AssistantMessage_Content_Item as a TextPart
func (t AssistantMessage_Content_Item) AsTextPart() (TextPart, error) {
	var body TextPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromTextPart overwrites any union data inside the AssistantMessage_Content_Item as the provided TextPart
func (t *AssistantMessage_Content_Item) FromTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeTextPart performs a merge with any union data inside the AssistantMessage_Content_Item, using the provided TextPart
func (t *AssistantMessage_Content_Item) MergeTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolCallPart returns the union data inside the AssistantMessage_Content_Item as a ToolCallPart
func (t AssistantMessage_Content_Item) AsToolCallPart() (ToolCallPart, error) {
	var body ToolCallPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolCallPart overwrites any union data inside the AssistantMessage_Content_Item as the provided ToolCallPart
func (t *AssistantMessage_Content_Item) FromToolCallPart(v ToolCallPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolCallPart performs a merge with any union data inside the AssistantMessage_Content_Item, using the provided ToolCallPart
func (t *AssistantMessage_Content_Item) MergeToolCallPart(v ToolCallPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t AssistantMessage_Content_Item) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *AssistantMessage_Content_Item) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// AsTextPart returns the union data inside the ContentDelta_Part as a TextPart
func (t ContentDelta_Part) AsTextPart() (TextPart, error) {
	var body TextPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromTextPart overwrites any union data inside the ContentDelta_Part as the provided TextPart
func (t *ContentDelta_Part) FromTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeTextPart performs a merge with any union data inside the ContentDelta_Part, using the provided TextPart
func (t *ContentDelta_Part) MergeTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolCallPart returns the union data inside the ContentDelta_Part as a ToolCallPart
func (t ContentDelta_Part) AsToolCallPart() (ToolCallPart, error) {
	var body ToolCallPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolCallPart overwrites any union data inside the ContentDelta_Part as the provided ToolCallPart
func (t *ContentDelta_Part) FromToolCallPart(v ToolCallPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolCallPart performs a merge with any union data inside the ContentDelta_Part, using the provided ToolCallPart
func (t *ContentDelta_Part) MergeToolCallPart(v ToolCallPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t ContentDelta_Part) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *ContentDelta_Part) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// AsResponseFormatJson returns the union data inside the LanguageModelInput_ResponseFormat as a ResponseFormatJson
func (t LanguageModelInput_ResponseFormat) AsResponseFormatJson() (ResponseFormatJson, error) {
	var body ResponseFormatJson
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromResponseFormatJson overwrites any union data inside the LanguageModelInput_ResponseFormat as the provided ResponseFormatJson
func (t *LanguageModelInput_ResponseFormat) FromResponseFormatJson(v ResponseFormatJson) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeResponseFormatJson performs a merge with any union data inside the LanguageModelInput_ResponseFormat, using the provided ResponseFormatJson
func (t *LanguageModelInput_ResponseFormat) MergeResponseFormatJson(v ResponseFormatJson) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsResponseFormatText returns the union data inside the LanguageModelInput_ResponseFormat as a ResponseFormatText
func (t LanguageModelInput_ResponseFormat) AsResponseFormatText() (ResponseFormatText, error) {
	var body ResponseFormatText
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromResponseFormatText overwrites any union data inside the LanguageModelInput_ResponseFormat as the provided ResponseFormatText
func (t *LanguageModelInput_ResponseFormat) FromResponseFormatText(v ResponseFormatText) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeResponseFormatText performs a merge with any union data inside the LanguageModelInput_ResponseFormat, using the provided ResponseFormatText
func (t *LanguageModelInput_ResponseFormat) MergeResponseFormatText(v ResponseFormatText) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t LanguageModelInput_ResponseFormat) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *LanguageModelInput_ResponseFormat) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// AsToolChoiceAuto returns the union data inside the LanguageModelInput_ToolChoice as a ToolChoiceAuto
func (t LanguageModelInput_ToolChoice) AsToolChoiceAuto() (ToolChoiceAuto, error) {
	var body ToolChoiceAuto
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolChoiceAuto overwrites any union data inside the LanguageModelInput_ToolChoice as the provided ToolChoiceAuto
func (t *LanguageModelInput_ToolChoice) FromToolChoiceAuto(v ToolChoiceAuto) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolChoiceAuto performs a merge with any union data inside the LanguageModelInput_ToolChoice, using the provided ToolChoiceAuto
func (t *LanguageModelInput_ToolChoice) MergeToolChoiceAuto(v ToolChoiceAuto) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolChoiceNone returns the union data inside the LanguageModelInput_ToolChoice as a ToolChoiceNone
func (t LanguageModelInput_ToolChoice) AsToolChoiceNone() (ToolChoiceNone, error) {
	var body ToolChoiceNone
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolChoiceNone overwrites any union data inside the LanguageModelInput_ToolChoice as the provided ToolChoiceNone
func (t *LanguageModelInput_ToolChoice) FromToolChoiceNone(v ToolChoiceNone) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolChoiceNone performs a merge with any union data inside the LanguageModelInput_ToolChoice, using the provided ToolChoiceNone
func (t *LanguageModelInput_ToolChoice) MergeToolChoiceNone(v ToolChoiceNone) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolChoiceRequired returns the union data inside the LanguageModelInput_ToolChoice as a ToolChoiceRequired
func (t LanguageModelInput_ToolChoice) AsToolChoiceRequired() (ToolChoiceRequired, error) {
	var body ToolChoiceRequired
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolChoiceRequired overwrites any union data inside the LanguageModelInput_ToolChoice as the provided ToolChoiceRequired
func (t *LanguageModelInput_ToolChoice) FromToolChoiceRequired(v ToolChoiceRequired) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolChoiceRequired performs a merge with any union data inside the LanguageModelInput_ToolChoice, using the provided ToolChoiceRequired
func (t *LanguageModelInput_ToolChoice) MergeToolChoiceRequired(v ToolChoiceRequired) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolChoiceTool returns the union data inside the LanguageModelInput_ToolChoice as a ToolChoiceTool
func (t LanguageModelInput_ToolChoice) AsToolChoiceTool() (ToolChoiceTool, error) {
	var body ToolChoiceTool
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolChoiceTool overwrites any union data inside the LanguageModelInput_ToolChoice as the provided ToolChoiceTool
func (t *LanguageModelInput_ToolChoice) FromToolChoiceTool(v ToolChoiceTool) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolChoiceTool performs a merge with any union data inside the LanguageModelInput_ToolChoice, using the provided ToolChoiceTool
func (t *LanguageModelInput_ToolChoice) MergeToolChoiceTool(v ToolChoiceTool) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t LanguageModelInput_ToolChoice) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *LanguageModelInput_ToolChoice) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// AsUserMessage returns the union data inside the Message as a UserMessage
func (t Message) AsUserMessage() (UserMessage, error) {
	var body UserMessage
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromUserMessage overwrites any union data inside the Message as the provided UserMessage
func (t *Message) FromUserMessage(v UserMessage) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeUserMessage performs a merge with any union data inside the Message, using the provided UserMessage
func (t *Message) MergeUserMessage(v UserMessage) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsAssistantMessage returns the union data inside the Message as a AssistantMessage
func (t Message) AsAssistantMessage() (AssistantMessage, error) {
	var body AssistantMessage
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromAssistantMessage overwrites any union data inside the Message as the provided AssistantMessage
func (t *Message) FromAssistantMessage(v AssistantMessage) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeAssistantMessage performs a merge with any union data inside the Message, using the provided AssistantMessage
func (t *Message) MergeAssistantMessage(v AssistantMessage) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolMessage returns the union data inside the Message as a ToolMessage
func (t Message) AsToolMessage() (ToolMessage, error) {
	var body ToolMessage
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolMessage overwrites any union data inside the Message as the provided ToolMessage
func (t *Message) FromToolMessage(v ToolMessage) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolMessage performs a merge with any union data inside the Message, using the provided ToolMessage
func (t *Message) MergeToolMessage(v ToolMessage) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t Message) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *Message) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// AsTextPart returns the union data inside the ModelResponse_Content_Item as a TextPart
func (t ModelResponse_Content_Item) AsTextPart() (TextPart, error) {
	var body TextPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromTextPart overwrites any union data inside the ModelResponse_Content_Item as the provided TextPart
func (t *ModelResponse_Content_Item) FromTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeTextPart performs a merge with any union data inside the ModelResponse_Content_Item, using the provided TextPart
func (t *ModelResponse_Content_Item) MergeTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsToolCallPart returns the union data inside the ModelResponse_Content_Item as a ToolCallPart
func (t ModelResponse_Content_Item) AsToolCallPart() (ToolCallPart, error) {
	var body ToolCallPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromToolCallPart overwrites any union data inside the ModelResponse_Content_Item as the provided ToolCallPart
func (t *ModelResponse_Content_Item) FromToolCallPart(v ToolCallPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeToolCallPart performs a merge with any union data inside the ModelResponse_Content_Item, using the provided ToolCallPart
func (t *ModelResponse_Content_Item) MergeToolCallPart(v ToolCallPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t ModelResponse_Content_Item) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *ModelResponse_Content_Item) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// AsTextPart returns the union data inside the UserMessage_Content_Item as a TextPart
func (t UserMessage_Content_Item) AsTextPart() (TextPart, error) {
	var body TextPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromTextPart overwrites any union data inside the UserMessage_Content_Item as the provided TextPart
func (t *UserMessage_Content_Item) FromTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeTextPart performs a merge with any union data inside the UserMessage_Content_Item, using the provided TextPart
func (t *UserMessage_Content_Item) MergeTextPart(v TextPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsImagePart returns the union data inside the UserMessage_Content_Item as a ImagePart
func (t UserMessage_Content_Item) AsImagePart() (ImagePart, error) {
	var body ImagePart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromImagePart overwrites any union data inside the UserMessage_Content_Item as the provided ImagePart
func (t *UserMessage_Content_Item) FromImagePart(v ImagePart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeImagePart performs a merge with any union data inside the UserMessage_Content_Item, using the provided ImagePart
func (t *UserMessage_Content_Item) MergeImagePart(v ImagePart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t UserMessage_Content_Item) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *UserMessage_Content_Item) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}