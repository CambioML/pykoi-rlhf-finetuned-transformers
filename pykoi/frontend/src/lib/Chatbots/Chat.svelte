<script>
  import Chatbot from "./Chatbot.svelte";
  import RankedChatbot from "./RankedChatbot.svelte";

  export let feedback;
</script>

{#if !feedback}
  <Chatbot />
{/if}
{#if feedback === "vote"}
  <Chatbot feedback={true} />
{/if}
{#if feedback === "rank"}
  <RankedChatbot />
{/if}
