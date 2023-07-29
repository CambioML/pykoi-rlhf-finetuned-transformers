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
  { model: "llama", qid: 7, rank: 4, answer: "Llama's seventh unique answer." },
  {
    model: "gpt3.5",
    qid: 7,
    rank: 3,
    answer: "GPT-3.5's seventh unique answer.",
  },
  { model: "gpt4", qid: 7, rank: 2, answer: "GPT-4's seventh unique answer." },
  {
    model: "claude",
    qid: 7,
    rank: 1,
    answer: "Claude's seventh unique answer.",
  },

  { model: "llama", qid: 8, rank: 3, answer: "Llama's eighth unique answer." },
  {
    model: "gpt3.5",
    qid: 8,
    rank: 4,
    answer: "GPT-3.5's eighth unique answer.",
  },
  { model: "gpt4", qid: 8, rank: 1, answer: "GPT-4's eighth unique answer." },
  {
    model: "claude",
    qid: 8,
    rank: 2,
    answer: "Claude's eighth unique answer.",
  },

  { model: "llama", qid: 9, rank: 2, answer: "Llama's ninth unique answer." },
  {
    model: "gpt3.5",
    qid: 9,
    rank: 1,
    answer: "GPT-3.5's ninth unique answer.",
  },
  { model: "gpt4", qid: 9, rank: 4, answer: "GPT-4's ninth unique answer." },
  { model: "claude", qid: 9, rank: 3, answer: "Claude's ninth unique answer." },

  { model: "llama", qid: 10, rank: 4, answer: "Llama's tenth unique answer." },
  {
    model: "gpt3.5",
    qid: 10,
    rank: 3,
    answer: "GPT-3.5's tenth unique answer.",
  },
  { model: "gpt4", qid: 10, rank: 2, answer: "GPT-4's tenth unique answer." },
  {
    model: "claude",
    qid: 10,
    rank: 1,
    answer: "Claude's tenth unique answer.",
  },

  {
    model: "llama",
    qid: 11,
    rank: 3,
    answer: "Llama's eleventh unique answer.",
  },
  {
    model: "gpt3.5",
    qid: 11,
    rank: 4,
    answer: "GPT-3.5's eleventh unique answer.",
  },
  {
    model: "gpt4",
    qid: 11,
    rank: 1,
    answer: "GPT-4's eleventh unique answer.",
  },
  {
    model: "claude",
    qid: 11,
    rank: 2,
    answer: "Claude's eleventh unique answer.",
  },

  {
    model: "llama",
    qid: 12,
    rank: 2,
    answer: "Llama's twelfth unique answer.",
  },
  {
    model: "gpt3.5",
    qid: 12,
    rank: 1,
    answer: "GPT-3.5's twelfth unique answer.",
  },
  { model: "gpt4", qid: 12, rank: 4, answer: "GPT-4's twelfth unique answer." },
  {
    model: "claude",
    qid: 12,
    rank: 3,
    answer: "Claude's twelfth unique answer.",
  },

  {
    model: "llama",
    qid: 13,
    rank: 4,
    answer: "Llama's thirteenth unique answer.",
  },
  {
    model: "gpt3.5",
    qid: 13,
    rank: 3,
    answer: "GPT-3.5's thirteenth unique answer.",
  },
  {
    model: "gpt4",
    qid: 13,
    rank: 2,
    answer: "GPT-4's thirteenth unique answer.",
  },
  {
    model: "claude",
    qid: 13,
    rank: 4,
    answer: "Claude's thirteenth unique answer.",
  },

  {
    model: "llama",
    qid: 14,
    rank: 3,
    answer: "Llama's fourteenth unique answer.",
  },
  {
    model: "gpt3.5",
    qid: 14,
    rank: 1,
    answer: "GPT-3.5's fourteenth unique answer.",
  },
  {
    model: "gpt4",
    qid: 14,
    rank: 1,
    answer: "GPT-4's fourteenth unique answer.",
  },
  {
    model: "claude",
    qid: 14,
    rank: 1,
    answer: "Claude's fourteenth unique answer.",
  },
]);
