<!-- App.svelte -->
<script>
  import { writable } from "svelte/store";
  import Chatbot from "./Chatbot.svelte";
  import Dropdown from "./Dropdown.svelte";

  const components = writable([]);

  fetch("/components")
    .then((response) => response.json())
    .then((data) => {
      console.log("component", data);
      components.set(data);
    });
</script>

{#each $components as component}
  {#if component.svelte_component === "Chatbot"}
    <Chatbot {...component.props} />
  {/if}

  {#if component.svelte_component === "Dropdown"}
    <Dropdown {...component.props} />
  {/if}
{/each}
<p class="footer-logo">Made with Cambio.</p>

<style>
  .footer-logo {
    font-size: var(--smallText);
  }
</style>
