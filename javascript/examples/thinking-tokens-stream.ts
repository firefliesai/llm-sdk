/* eslint-disable @typescript-eslint/restrict-template-expressions */
import { PartialModelResponse } from "../src/index.js";
import { OpenAIModel } from "../src/openai/index.js";

/**
 * Example: Real-time thinking token streaming with visualization
 * This demonstrates how to capture and display thinking tokens as they arrive
 */
async function streamThinkingTokens() {
  const apiKey = process.env["OPENAI_API_KEY"];
  if (!apiKey) throw new Error("OPENAI_API_KEY is required");
  const model = new OpenAIModel({
    apiKey,
    modelId: "gpt-5-mini", // Model that supports reasoning
  });

  console.log("🧠 Starting thinking token stream...\n");

  const stream = model.stream({
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Create a step-by-step plan for building a sustainable smart city. Consider environmental, technological, and social factors.",
          },
        ],
      },
    ],
    reasoning: {
      effort: "high",
      summary: "auto",
    },
  });

  let thinkingContent = "";
  let responseContent = "";
  let reasoningChunkCount = 0;
  let textChunkCount = 0;

  // Track timing
  const startTime = Date.now();
  let firstThinkingTime: number | null = null;
  let firstResponseTime: number | null = null;

  console.log("┌─ Thinking Process ─┐");

  for await (const chunk of stream) {
    const currentTime = Date.now();
    const part = chunk.delta.part;

    if (part.type === "reasoning") {
      if (!firstThinkingTime) {
        firstThinkingTime = currentTime;
        console.log("🟡 Thinking started...");
      }

      if (part.reasoning) {
        thinkingContent += part.reasoning;
        reasoningChunkCount++;

        // Show live thinking progress
        const preview =
          part.reasoning.length > 50
            ? part.reasoning.slice(0, 50) + "..."
            : part.reasoning;

        process.stdout.write(`\r💭 ${preview.padEnd(60)}`);
      }

      if (part.summary) {
        console.log("\n🔵 Reasoning summary received");
      }
    } else if (part.type === "text") {
      if (!firstResponseTime) {
        firstResponseTime = currentTime;
        console.log("\n└─────────────────────┘");
        console.log("\n┌─ Response ─┐");
        console.log("🟢 Response started...");
      }

      if (part.text) {
        responseContent += part.text;
        textChunkCount++;

        // Show response in real-time
        process.stdout.write(part.text);
      }
    }
  }

  const endTime = Date.now();

  console.log("\n└──────────────┘");
  console.log("\n📊 Stream Statistics:");
  console.log(`   Total time: ${endTime - startTime}ms`);
  console.log(
    `   Time to first thinking: ${firstThinkingTime ? firstThinkingTime - startTime : "N/A"}ms`,
  );
  console.log(
    `   Time to first response: ${firstResponseTime ? firstResponseTime - startTime : "N/A"}ms`,
  );
  console.log(`   Reasoning chunks received: ${reasoningChunkCount}`);
  console.log(`   Text chunks received: ${textChunkCount}`);
  console.log(`   Total thinking length: ${thinkingContent.length} characters`);
  console.log(`   Total response length: ${responseContent.length} characters`);

  // Show first 200 characters of thinking process
  console.log("\n🧠 Thinking Process Preview:");
  console.log(
    `"${thinkingContent.slice(0, 200)}${thinkingContent.length > 200 ? "..." : ""}"`,
  );
}

/**
 * Example: Building a thinking token visualizer
 */
class ThinkingTokenVisualizer {
  private thinkingBuffer = "";
  private responseBuffer = "";

  async visualize(stream: AsyncGenerator<PartialModelResponse>) {
    console.clear();
    console.log("🧠 Thinking Token Visualizer");
    console.log("=".repeat(80));
    console.log("\n💭 THINKING PROCESS:");
    console.log("─".repeat(40));
    console.log("Starting to capture thinking tokens...\n");

    for await (const chunk of stream) {
      const part = chunk.delta.part;

      if (part.type === "reasoning" && part.reasoning) {
        this.thinkingBuffer += part.reasoning;

        // Show real-time thinking progress
        process.stdout.write(part.reasoning);
      } else if (part.type === "text" && part.text) {
        this.responseBuffer += part.text;

        // When we start getting response, show the transition
        if (this.responseBuffer === part.text) {
          console.log("\n\n💬 RESPONSE:");
          console.log("─".repeat(40));
        }

        // Show response in real-time
        process.stdout.write(part.text);
      }
    }

    // Final summary
    console.log("\n\n📊 SUMMARY:");
    console.log("─".repeat(40));
    console.log(`💭 Thinking tokens: ${this.thinkingBuffer.length} characters`);
    console.log(`💬 Response tokens: ${this.responseBuffer.length} characters`);
  }
}

/**
 * Example: Using the visualizer
 */
async function visualizedThinkingExample() {
  const apiKey = process.env["OPENAI_API_KEY"];
  if (!apiKey) throw new Error("OPENAI_API_KEY is required");
  const model = new OpenAIModel({
    apiKey,
    modelId: "gpt-5-mini",
  });

  const stream = model.stream({
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Explain quantum computing and its potential applications in cryptography. Be thorough in your analysis.",
          },
        ],
      },
    ],
    reasoning: {
      effort: "high",
      summary: "auto", // Add summary option to match working example
    },
  });

  const visualizer = new ThinkingTokenVisualizer();
  await visualizer.visualize(stream);
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const mode = args[0] || "stream";

  try {
    switch (mode) {
      case "stream":
        await streamThinkingTokens();
        break;
      case "visualize":
        await visualizedThinkingExample();
        break;
      default:
        console.log("Usage: npm run example:thinking [stream|visualize]");
        console.log("  stream    - Simple thinking token stream");
        console.log("  visualize - Interactive thinking token visualizer");
    }
  } catch (error) {
    console.error("Error:", error);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
