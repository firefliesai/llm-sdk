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
  private lastUpdate = Date.now();
  private updateInterval = 100; // Update UI every 100ms

  async visualize(stream: AsyncGenerator<PartialModelResponse>) {
    console.clear();
    console.log("🧠 Thinking Token Visualizer\n");

    for await (const chunk of stream) {
      const part = chunk.delta.part;
      const now = Date.now();

      if (part.type === "reasoning" && part.reasoning) {
        this.thinkingBuffer += part.reasoning;
      } else if (part.type === "text" && part.text) {
        this.responseBuffer += part.text;
      }

      // Update display periodically to avoid overwhelming the terminal
      if (now - this.lastUpdate > this.updateInterval) {
        this.updateDisplay();
        this.lastUpdate = now;
      }
    }

    // Final update
    this.updateDisplay();
  }

  private updateDisplay() {
    console.clear();
    console.log("🧠 Thinking Token Visualizer");
    console.log("=".repeat(80));

    console.log("\n💭 THINKING PROCESS:");
    console.log("─".repeat(40));
    const thinkingPreview = this.thinkingBuffer.slice(-200); // Show last 200 chars
    console.log(this.formatText(thinkingPreview, 70));
    console.log(
      `\n[Thinking length: ${this.thinkingBuffer.length} characters]`,
    );

    if (this.responseBuffer) {
      console.log("\n💬 RESPONSE:");
      console.log("─".repeat(40));
      console.log(this.formatText(this.responseBuffer, 70));
    }

    // Add a blinking cursor effect for active thinking
    if (this.responseBuffer === "") {
      process.stdout.write(" ▊");
    }
  }

  private formatText(text: string, maxWidth: number): string {
    const words = text.split(" ");
    const lines: string[] = [];
    let currentLine = "";

    for (const word of words) {
      if ((currentLine + word).length > maxWidth) {
        if (currentLine) {
          lines.push(currentLine.trim());
          currentLine = word + " ";
        } else {
          lines.push(word.slice(0, maxWidth));
          currentLine = word.slice(maxWidth) + " ";
        }
      } else {
        currentLine += word + " ";
      }
    }

    if (currentLine) {
      lines.push(currentLine.trim());
    }

    return lines.join("\n");
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
