import { writable } from "svelte/store";
import { tweened } from "svelte/motion";
export const wronglyAccepted = writable(1);
export const wronglyRejected = writable(1);

export const questionDistribution = tweened([
  {
    question: "Who",
    count: 24,
  },
  {
    question: "What",
    count: 20,
  },
  {
    question: "How",
    count: 15,
  },
  {
    question: "Why",
    count: 15,
  },
  {
    question: "Where",
    count: 5,
  },
  {
    question: "Does",
    count: 25,
  },
  {
    question: "Can",
    count: 15,
  },
  {
    question: "N/A",
    count: 45,
  },
]);
