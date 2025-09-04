/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/restrict-template-expressions */
import { OpenAILanguageModelInput, OpenAIModel } from "../src/openai/index.js";

// Example: Using reasoning tokens with OpenAI Responses API
async function basicReasoningExample() {
  const model = new OpenAIModel({
    apiKey: process.env["OPENAI_API_KEY"] as string,
    modelId: "gpt-5-mini", // Use a model that supports reasoning
  });

  const response = await model.generate({
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "I need to solve the equation 3x + 11 = 14. Can you show me your reasoning step by step?",
          },
        ],
      },
    ],
    reasoning: {
      effort: "high", // Request detailed reasoning
      summary: "auto", // Get automatic reasoning summaries
    },
  });

  console.log("Response content:");
  response.content.forEach((part) => {
    if (part.type === "reasoning") {
      console.log(
        `[THINKING${part.summary ? " SUMMARY" : ""}]: ${part.reasoning}`,
      );
    } else if (part.type === "text") {
      console.log(`[RESPONSE]: ${part.text}`);
    }
  });

  console.log("\nToken usage:");
  console.log(`Input tokens: ${response.usage?.inputTokens}`);
  console.log(`Output tokens: ${response.usage?.outputTokens}`);
  console.log(
    `Reasoning tokens: ${response.usage?.outputTokensDetail?.reasoningTokens}`,
  );
  console.log(`Text tokens: ${response.usage?.outputTokensDetail?.textTokens}`);
}

// Example: Streaming reasoning tokens
async function streamingReasoningExample() {
  const model = new OpenAIModel({
    apiKey: process.env["OPENAI_API_KEY"] as string,
    modelId: "gpt-5-mini",
  });

  console.log("Streaming reasoning tokens...\n");

  const stream = model.stream({
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Analyze the pros and cons of renewable energy adoption. Think through this systematically.",
          },
        ],
      },
    ],
    reasoning: {
      effort: "medium",
      summary: "auto",
    },
  });

  let thinkingBuffer = "";
  let responseBuffer = "";

  for await (const chunk of stream) {
    const part = chunk.delta.part;

    if (part.type === "reasoning") {
      if (part.reasoning) {
        thinkingBuffer += part.reasoning;
        // Show thinking process in real-time
        process.stdout.write(`\r[THINKING]: ${thinkingBuffer.slice(-50)}...`);
      }
    } else if (part.type === "text") {
      if (part.text) {
        responseBuffer += part.text;
        console.log(`\n[RESPONSE]: ${part.text}`);
      }
    }
  }

  console.log(`\nFull thinking process: ${thinkingBuffer}`);
}

// Example: Using Responses API with background processing
async function backgroundReasoningExample() {
  const model = new OpenAIModel({
    apiKey: process.env["OPENAI_API_KEY"] as string,
    modelId: "o3", // Use a model that supports background processing
  });

  const response = await model.generate({
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Write a comprehensive analysis of climate change impacts on agriculture worldwide.",
          },
        ],
      },
    ],
    reasoning: {
      effort: "high",
    },
    // Using the extended input type for Responses API options
    responsesOptions: {
      background: true, // Process in background for long-running tasks
      store: false, // Don't store for ZDR compliance
    },
  } as OpenAILanguageModelInput); // Type assertion for the extended input

  console.log("Background processing completed!");
  response.content.forEach((part) => {
    if (part.type === "reasoning") {
      const reasoningText = part.reasoning || "";
      console.log(`[THINKING]: ${reasoningText.slice(0, 200)}...`);
    } else if (part.type === "text") {
      const responseText = part.text || "";
      console.log(`[RESPONSE]: ${responseText.slice(0, 200)}...`);
    }
  });
}

// Example: Comparing with and without reasoning
async function comparisonExample() {
  const model = new OpenAIModel({
    apiKey: process.env["OPENAI_API_KEY"] as string,
    modelId: "gpt-5-mini", // Use reasoning-capable model for both comparisons
  });

  const question =
    "What's the best strategy for learning a new programming language?";

  console.log("=== Without Reasoning Tokens ===");
  const regularResponse = await model.generate({
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: question }],
      },
    ],
  });

  console.log(
    regularResponse.content[0]?.type === "text"
      ? regularResponse.content[0].text
      : "",
  );
  console.log(`Tokens used: ${regularResponse.usage?.outputTokens}`);

  console.log("\n=== With Reasoning Tokens ===");
  const reasoningResponse = await model.generate({
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: question }],
      },
    ],
    reasoning: {
      effort: "high",
      summary: "auto",
    },
  });

  reasoningResponse.content.forEach((part) => {
    if (part.type === "reasoning") {
      console.log(`[THINKING]: ${part.reasoning}`);
    } else if (part.type === "text") {
      console.log(`[RESPONSE]: ${part.text}`);
    }
  });

  console.log(`Total tokens: ${reasoningResponse.usage?.outputTokens}`);
  console.log(
    `Reasoning tokens: ${reasoningResponse.usage?.outputTokensDetail?.reasoningTokens}`,
  );
  console.log(
    `Text tokens: ${reasoningResponse.usage?.outputTokensDetail?.textTokens}`,
  );
}

// Example: Controlling reasoning verbosity
async function reasoningVerbosityExample() {
  const model = new OpenAIModel({
    apiKey: process.env["OPENAI_API_KEY"] as string,
    modelId: "gpt-5-mini",
  });

  const problem = "Design a database schema for a social media platform.";

  const efforts = ["low", "medium", "high"] as const;

  for (const effort of efforts) {
    console.log(`\n=== Reasoning Effort: ${effort.toUpperCase()} ===`);

    const response = await model.generate({
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: problem }],
        },
      ],
      reasoning: {
        effort,
        summary: "auto",
      },
    });

    const reasoningPart = response.content.find((p) => p.type === "reasoning");
    const textPart = response.content.find((p) => p.type === "text");

    if (reasoningPart?.type === "reasoning") {
      console.log(
        `Thinking (${reasoningPart.reasoning.length} chars): ${reasoningPart.reasoning.slice(0, 100)}...`,
      );
    }

    if (textPart?.type === "text") {
      console.log(`Response: ${textPart.text.slice(0, 100)}...`);
    }

    console.log(
      `Reasoning tokens: ${response.usage?.outputTokensDetail?.reasoningTokens}`,
    );
  }
}

// Run examples
async function main() {
  try {
    console.log("=== Basic Reasoning Example ===");
    await basicReasoningExample();

    console.log("\n=== Streaming Reasoning Example ===");
    await streamingReasoningExample();

    console.log("\n=== Background Reasoning Example ===");
    await backgroundReasoningExample();

    console.log("\n=== Comparison Example ===");
    await comparisonExample();

    console.log("\n=== Reasoning Verbosity Example ===");
    await reasoningVerbosityExample();
  } catch (error) {
    console.error("Error running examples:", error);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
