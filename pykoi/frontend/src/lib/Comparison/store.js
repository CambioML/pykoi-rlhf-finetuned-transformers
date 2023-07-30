import { writable } from "svelte/store";

export const comparisonData = writable([
  { model: "llama", qid: 1, rank: 1, answer: "Llama's first unique answer." },
  {
    model: "gpt3.5",
    qid: 1,
    rank: 2,
    answer: "GPT-3.5's first unique answer.",
  },
  { model: "gpt4", qid: 1, rank: 3, answer: "GPT-4's first unique answer." },
  { model: "claude", qid: 1, rank: 4, answer: "Claude's first unique answer." },

  { model: "llama", qid: 2, rank: 4, answer: "Llama's second unique answer." },
  {
    model: "gpt3.5",
    qid: 2,
    rank: 2,
    answer: "GPT-3.5's second unique answer.",
  },
  { model: "gpt4", qid: 2, rank: 4, answer: "GPT-4's second unique answer." },
  { model: "claude", qid: 2, rank: 3, answer: "'s second unique answer." },
]);
