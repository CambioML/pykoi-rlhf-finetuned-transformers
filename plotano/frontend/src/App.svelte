<script>
  import { writable } from "svelte/store";
  import Chatbot from "./lib/Chatbots/Chatbot.svelte";
  import Dropdown from "./lib/UIComponents/Dropdown.svelte";
  import Feedback from "./lib/Dashboards/Feedback.svelte";
  import RankedChatbot from "./lib/Chatbots/RankedChatbot.svelte";
  import Chat from "./lib/Chatbots/Chat.svelte";

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
    <Chat {...component.props} />
  {/if}

  {#if component.svelte_component === "Dropdown"}
    <Dropdown {...component.props} />
  {/if}
{/each}
