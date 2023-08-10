import { writable } from "svelte/store";

export const uploadedFiles = writable([]);
export const projections = writable({
    labelNames: ["txt2.txt", "pdf2.txt"],
    labels: [0, 3],
    projection: [
      [0.1443715864989249, 0.00463214192040876, 0.11292804485781914],
      [-0.05626482326843836, -0.013694651798983428, 0.01709368313171561],
    ],
  });