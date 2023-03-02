import { OpenAI } from "langchain";
import { initializeAgentExecutor } from "langchain/agents";
import { SerpAPI } from "langchain/tools";
import dotenv from "dotenv";

dotenv.config();

const model = new OpenAI({ temperature: 0 });
const tools = [new SerpAPI()];

const executor = await initializeAgentExecutor(
  tools,
  model,
  "zero-shot-react-description"
);
console.log("Loaded agent.");

const input = "Who is the better putter on the pga tour?";
console.log(`Executing with input "${input}"...`);

const result = await executor.call({ input });

console.log(`Got output ${result.output}`);
