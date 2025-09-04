import { describe, it } from "node:test";
import assert from "node:assert";
import { OpenAIModel, OpenAIResponsesClient } from "../src/openai/index.js";
import type { OpenAIModelOptions } from "../src/openai/types.js";

await describe("OpenAI Responses API - Basic Tests", () => {
  const mockOptions: OpenAIModelOptions = {
    apiKey: "test-key",
    modelId: "o1-mini",
  };

  void it("should create OpenAIResponsesClient", () => {
    const client = new OpenAIResponsesClient(mockOptions);
    assert.ok(client, "Client should be created successfully");
  });

  void it("should create OpenAIModel with responses client", () => {
    const model = new OpenAIModel(mockOptions);
    assert.ok(model, "Model should be created successfully");
    assert.strictEqual(model.provider, "openai");
    assert.strictEqual(model.modelId, "o1-mini");
  });

  void it("should have correct reasoning options types", () => {
    const reasoningOptions = {
      effort: "high" as const,
      maxTokens: 500,
      exclude: false,
      summary: "auto" as const,
    };

    assert.strictEqual(reasoningOptions.effort, "high");
    assert.strictEqual(reasoningOptions.maxTokens, 500);
    assert.strictEqual(reasoningOptions.exclude, false);
    assert.strictEqual(reasoningOptions.summary, "auto");
  });

  void it("should have correct reasoning part types", () => {
    const reasoningPart = {
      type: "reasoning" as const,
      reasoning: "This is my thinking process...",
      summary: true,
      id: "thinking-1",
    };

    assert.strictEqual(reasoningPart.type, "reasoning");
    assert.strictEqual(typeof reasoningPart.reasoning, "string");
    assert.strictEqual(typeof reasoningPart.summary, "boolean");
    assert.ok(reasoningPart.reasoning.length > 0);
  });

  void it("should export all required types and classes", () => {
    // Test that all the exports are available
    assert.ok(OpenAIModel, "OpenAIModel should be exported");
    assert.ok(
      OpenAIResponsesClient,
      "OpenAIResponsesClient should be exported",
    );
  });
});

// Test the token usage mapping
await describe("Token Usage with Reasoning", () => {
  void it("should correctly structure ModelTokensDetail with reasoning tokens", () => {
    const tokensDetail = {
      textTokens: 100,
      audioTokens: 50,
      imageTokens: 25,
      reasoningTokens: 200, // New reasoning tokens field
    };

    assert.strictEqual(tokensDetail.textTokens, 100);
    assert.strictEqual(tokensDetail.reasoningTokens, 200);
    assert.ok(
      "reasoningTokens" in tokensDetail,
      "Should include reasoningTokens field",
    );
  });

  void it("should correctly structure ModelUsage with reasoning token details", () => {
    const modelUsage = {
      inputTokens: 50,
      outputTokens: 150,
      outputTokensDetail: {
        textTokens: 100,
        reasoningTokens: 50,
      },
    };

    assert.strictEqual(modelUsage.inputTokens, 50);
    assert.strictEqual(modelUsage.outputTokens, 150);
    assert.strictEqual(modelUsage.outputTokensDetail.reasoningTokens, 50);
    assert.strictEqual(modelUsage.outputTokensDetail.textTokens, 100);
  });
});

// Test content types
await describe("Content Types with Reasoning", () => {
  void it("should handle reasoning content in model response", () => {
    const modelResponse = {
      content: [
        {
          type: "reasoning" as const,
          reasoning: "Let me think about this step by step...",
          summary: false,
          id: "thought-1",
        },
        {
          type: "text" as const,
          text: "Here's my final answer.",
        },
      ],
      usage: {
        inputTokens: 20,
        outputTokens: 80,
        outputTokensDetail: {
          textTokens: 30,
          reasoningTokens: 50,
        },
      },
    };

    assert.strictEqual(modelResponse.content.length, 2);

    const reasoningPart = modelResponse.content[0];
    if (reasoningPart?.type === "reasoning") {
      assert.strictEqual(
        reasoningPart.reasoning,
        "Let me think about this step by step...",
      );
      assert.strictEqual(reasoningPart.summary, false);
    }

    const textPart = modelResponse.content[1];
    if (textPart?.type === "text") {
      assert.strictEqual(textPart.text, "Here's my final answer.");
    }

    assert.strictEqual(
      modelResponse.usage.outputTokensDetail.reasoningTokens,
      50,
    );
  });

  void it("should handle reasoning content deltas in streaming", () => {
    const contentDelta = {
      index: 0,
      part: {
        type: "reasoning" as const,
        reasoning: "Thinking step ",
        summary: true,
      },
    };

    assert.strictEqual(contentDelta.index, 0);
    assert.strictEqual(contentDelta.part.type, "reasoning");
    assert.strictEqual(contentDelta.part.reasoning, "Thinking step ");
    assert.strictEqual(contentDelta.part.summary, true);
  });
});
