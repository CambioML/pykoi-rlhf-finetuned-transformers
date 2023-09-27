<script>
  import Chatbot from "./Chatbot.svelte";
  import RankedChatbot from "./RankedChatbot.svelte";
  import RAGChatbot from "./RAGChatbot.svelte";
  export let feedback;
  export let is_retrieval;
</script>

{#if !feedback}
  <Chatbot {is_retrieval} />
{/if}
{#if feedback === "vote"}
  <Chatbot feedback={true} {is_retrieval} />
{/if}
{#if feedback === "rag"}
  <RAGChatbot feedback={true} {is_retrieval} />
{/if}
{#if feedback === "rank"}
  <!-- TODO: Refactor to support ranking with retrieval-->
  {#if is_retrieval}
    <Chatbot feedback={true} {is_retrieval} />
  {:else}
    <RankedChatbot />
  {/if}
{/if}
