<script>
    import Feedback from "./Feedback.svelte";
    import RagFeedback from "./RAGFeedback.svelte";

    export let feedback;
</script>

{#if feedback === "vote"}
    <Feedback />
{:else if feedback === "rag"}
    <RagFeedback />
{:else}
    <Feedback />
{/if}
