// step3.js: a short answer, written in the page by a small language model,
// from the notices the search found.
import './config.js';
import { pipeline } from '/vendor/@huggingface/transformers/dist/transformers.js';

export const MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct';
let generator = null;

export async function loadGenerator() {
  // 135 million parameters in 8-bit: one file of about 137 MB.
  if (!generator) generator = await pipeline('text-generation', MODEL, { dtype: 'q8', device: 'wasm' });
  return generator;
}

// Your turn. Build the chat the model will see: a system message telling it
// to answer only from the notices, and a user message holding the notices'
// text and then the question. Return an array like
//   [{ role: 'system', content: '...' }, { role: 'user', content: '...' }]
// Put each notice's title and text in the user message, one notice per
// paragraph, before the question.
export function buildMessages(question, notices) {
  // YOUR CODE HERE
  return [];
}

export async function answer(question, notices, maxNewTokens = 60) {
  const messages = buildMessages(question, notices);
  if (!messages.length) return { messages, answer: '', new_tokens: 0 };
  const gen = await loadGenerator();
  // do_sample false: always take the most likely next token, so the same
  // question gives the same answer every time.
  const out = await gen(messages, { max_new_tokens: maxNewTokens, do_sample: false });
  const text = out[0].generated_text.at(-1).content.trim();
  return { messages, answer: text, new_tokens: gen.tokenizer.encode(text).length };
}
