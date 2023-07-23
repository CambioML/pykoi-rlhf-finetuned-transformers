<script>
  import { writable } from "svelte/store";
  import Chat from "./lib/Chatbots/Chat.svelte";
  import Dropdown from "./lib/UIComponents/Dropdown.svelte";
  import Feedback from "./lib/Dashboards/Feedback.svelte";

  const components = writable([]);

  const componentMap = {
    Chatbot: Chat,
    Dropdown: Dropdown,
    Feedback: Feedback,
  };

  fetch("/components")
    .then((response) => response.json())
    .then((data) => {
      components.set(data);
    });
</script>

{#each $components as component}
  <svelte:component
    this={componentMap[component.svelte_component]}
    {...component.props}
  />
{/each}
