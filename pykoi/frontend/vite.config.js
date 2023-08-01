import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default {
  plugins: [svelte()],
  build: {
    rollupOptions: {
      output: {
        format: "iife",
        manualChunks: undefined, // Disable code splitting
      },
    },
  },
};
