# LLM SDK

A unified SDK for working with various Large Language Model providers.

## Basic Usage

```typescript
import { OpenAIModel } from "@firefliesai/llm-sdk/openai";
// or
import { GoogleModel } from "@firefliesai/llm-sdk/google";
// or
import { AnthropicModel } from "@firefliesai/llm-sdk/anthropic";
// or
import { CohereModel } from "@firefliesai/llm-sdk/cohere";
// or
import { MistralModel } from "@firefliesai/llm-sdk/mistral";

const model = new OpenAIModel({
  apiKey: process.env.OPENAI_API_KEY!,
  modelId: "gpt-5-mini", // Use reasoning-capable models
});

> Note: Reasoning/thinking tokens are currently supported only with the OpenAI Responses API. Using `reasoning` content with Anthropic, Google, Cohere, or Mistral will throw a `ModelUnsupportedMessagePart` error.

// Generate with thinking tokens
const response = await model.generate({
  messages: [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "Solve the equation 3x + 11 = 14. Show your reasoning.",
        },
      ],
    },
  ],
  reasoning: {
    effort: "high", // "low", "medium", or "high"
    summary: "auto", // Get automatic reasoning summaries
  },
});

// Access reasoning and response content
response.content.forEach((part) => {
  if (part.type === "reasoning") {
    console.log(`[THINKING]: ${part.reasoning}`);
    console.log(`[SUMMARY]: ${part.summary ? "Yes" : "No"}`);
  } else if (part.type === "text") {
    console.log(`[RESPONSE]: ${part.text}`);
  }
});

// Check token usage
console.log(`Reasoning tokens: ${response.usage?.outputTokensDetail?.reasoningTokens ?? 0}`);
console.log(`Text tokens: ${response.usage?.outputTokensDetail?.textTokens ?? 0}`);
```

## Examples

Run the included examples to see thinking tokens in action:

```bash
# Basic reasoning examples
npm run example:reasoning

# Interactive thinking token streaming
npm run example:thinking

# Visualized thinking process
npm run example:thinking:visualize
```

## Migration Guide

Existing code using the OpenAI provider continues to work unchanged. To enable thinking tokens:

1. Add a `reasoning` configuration to your input
2. Handle the new `reasoning` content type in responses
3. Monitor token usage for reasoning tokens

No breaking changes to existing functionality.

## Other Features

The SDK also supports all standard LLM features:
- Text generation and streaming
- Tool/function calling
- Structured outputs
- Multi-modal inputs (text, images, audio)
- Multiple providers (OpenAI, Anthropic, Google, Cohere, Mistral)

See the main documentation for details on these features.

## License

MIT
