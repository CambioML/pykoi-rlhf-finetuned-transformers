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

  { model: "llama", qid: 3, rank: 1, answer: "Llama's third unique answer." },
  {
    model: "gpt3.5",
    qid: 3,
    rank: 4,
    answer: "GPT-3.5's third unique answer.",
  },
  { model: "gpt4", qid: 3, rank: 4, answer: "GPT-4's third unique answer." },
  { model: "claude", qid: 3, rank: 2, answer: "'s third unique answer." },

  { model: "llama", qid: 4, rank: 1, answer: "Llama's fourth unique answer." },
  { model: "gpt3.5", qid: 4, rank: 3, answer: "GPT-3.5's  unique answer." },
  { model: "gpt4", qid: 4, rank: 2, answer: "GPT-4's  unique answer." },
  { model: "claude", qid: 4, rank: 4, answer: "'s  unique answer." },

  { model: "llama", qid: 5, rank: 1, answer: "Llama's  unique answer." },
  { model: "gpt3.5", qid: 5, rank: 2, answer: "GPT-3.5's nique answer." },
  { model: "gpt4", qid: 5, rank: 3, answer: "GPT-4's fifth unique answer." },
  { model: "claude", qid: 5, rank: 4, answer: "'s fifthnswer." },

  { model: "llama", qid: 6, rank: 1, answer: "Llama's sixth unique answer." },
  {
    model: "gpt3.5",
    qid: 6,
    rank: 2,
    answer: "GPT-3.5's sixth unique answer.",
  },
  { model: "gpt4", qid: 6, rank: 3, answer: "GPT-4's sixth unique answer." },
  { model: "claude", qid: 6, rank: 4, answer: "'s sixth uniq." },
]);
