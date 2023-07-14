<!-- App.svelte -->
<script>
  import { writable } from "svelte/store";
  import Chatbot from "./Chatbot.svelte";
  import Dropdown from "./Dropdown.svelte";

  const components = writable([]);

  fetch("/components")
    .then((response) => response.json())
    .then((data) => {
      console.log("data", data);
      components.set(data);
    });
</script>

<p>hey</p>
{#each $components as component}
  {#if component.svelte_component === "Chatbot"}
    <Chatbot {...component.props} />
  {/if}

  {#if component.svelte_component === "Dropdown"}
    <Dropdown {...component.props} />
  {/if}
{/each}
