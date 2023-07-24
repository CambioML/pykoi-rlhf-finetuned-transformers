import { writable } from "svelte/store";
import { tweened } from "svelte/motion";

export const state = writable({});
export const chatLog = writable([]);
export const rankChatLog = writable([]);

// feedback (rewrite)
export const feedbackSelection = writable("all"); // up, down, or NA

export const stackedData = writable({ "n/a": 1, up: 1, down: 1 });

const questions = ["Who", "What", "How", "Why", "Where", "Does", "Can", "N/A"];

export const questionDistribution = tweened(
  questions.map((question) => ({ question, count: 0 }))
);
