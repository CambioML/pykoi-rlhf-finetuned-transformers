<script>
  import { slide } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import QACard from "./QACard.svelte";
  import {
    chatLog,
    feedbackSelection,
    questionDistribution,
  } from "../../../store";
  import { tallyQuestions } from "../../../utils";

  $: qadata =
    $feedbackSelection === "all"
      ? $chatLog
      : $chatLog.filter((d) => d.vote_status === $feedbackSelection);

  const customSlide = (
    node,
    { delay = 0, duration = 1000, easing = cubicOut }
  ) => {
    return slide(node, { delay, duration, easing });
  };

  $: $questionDistribution = tallyQuestions(qadata);
</script>

<div class="qa-container">
  {#each qadata as { question, answer, vote_status }}
    <div transition:customSlide|local={{ duration: 300 }}>
      <QACard {question} {answer} feedback={vote_status} />
    </div>
  {/each}
</div>

<style>
  .qa-container {
    border: 2px solid var(--background);
    max-height: 100%;
    overflow-y: auto;
  }
</style>
