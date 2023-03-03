import { OpenAI } from "langchain/llms";
import { ChatVectorDBQAChain, VectorDBQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as readline from 'node:readline/promises';
import { exit, stdin as input, stdout as output } from 'node:process';
import dotenv from "dotenv";

dotenv.config();

/* Initialize the LLM to use to answer the question */
const model = new OpenAI({ verbose: true });
/* Load in the file we want to do question answering over */
const text = fs.readFileSync("src/data/gearfocusfaq.txt", "utf8");
/* Split the text into chunks */
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
const docs = await textSplitter.createDocuments([text]);
/* Create the vectorstore */
const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
const chain = VectorDBQAChain.fromLLM(model, vectorStore);
/* Ask it a question */
const rl = readline.createInterface({ input, output });
while (true) {
    const question = await rl.question('Q: ');
    const res = await chain.call({
        input_documents: docs,
        query: question,
    });
    console.log("A: " + res["text"]);
}
// /* Create the chain */
// const chain = ChatVectorDBQAChain.fromLLM(model, vectorStore);
// /* Ask it a question */
// let chat_history = ""
// while (true) {
//     const question = await rl.question('Q: ');
//     const res: unknown = await chain.call({ question, chat_history });
//     if (res && typeof res === 'object' && "text" in res) {
//         chat_history += question + res["text"];
//         console.log("A: " + res["text"]);
//     }
//     else {
//         console.error("Unexpected response: " + res);
//         exit(1);
//     }

// }
