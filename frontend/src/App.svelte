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

{#if $components.includes("Chatbot")}
  <Chatbot />
{/if}

{#if $components.includes("Dropdown")}
  <Dropdown />
{/if}
