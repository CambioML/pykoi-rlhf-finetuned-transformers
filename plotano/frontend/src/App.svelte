<!-- App.svelte -->
<script>
  import { writable } from "svelte/store";
  import Chatbot from "./Chatbot.svelte";
  import Dropdown from "./Dropdown.svelte";
  import Feedback from "./Feedback.svelte";
  import RankedChatbot from "./RankedChatbot.svelte";
  import Chat from "./Chat.svelte";

  const components = writable([]);

  fetch("/components")
    .then((response) => response.json())
    .then((data) => {
      components.set(data);
    });
</script>

{#each $components as component}
  {#if component.svelte_component === "Feedback"}
    <Feedback {...component.props} />
  {/if}
  {#if component.svelte_component === "Chatbot"}
    <!-- <Chatbot {...component.props} /> -->
    <Chat {...component.props} />
  {/if}

  {#if component.svelte_component === "Dropdown"}
    <Dropdown {...component.props} />
  {/if}
{/each}

<style>
  .footer-logo {
    font-size: var(--smallText);
    margin: auto;
    width: 100%;
    margin-top: 1rem;
  }
</style>
