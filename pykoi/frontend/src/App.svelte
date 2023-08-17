<script>
  import { writable } from "svelte/store";
  import Chat from "./lib/Chatbots/Chat.svelte";
  import Dropdown from "./lib/UIComponents/Dropdown.svelte";
  import Feedback from "./lib/Dashboards/Feedback.svelte";
  import ComparisonChat from "./lib/Chatbots/ComparisonChat.svelte";
  import ComparisonChart from "./lib/Comparison/ComparisonChart.svelte";
  import QuestionRating from "./lib/Annotations/QuestionRating.svelte";
  import RankedChatbot from "./lib/Chatbots/RankedChatbot.svelte";

  const components = writable([]);
  const selectedPage = writable(null);

  const componentMap = {
    Chatbot: Chat,
    Dropdown: Dropdown,
    Feedback: Feedback,
    Compare: ComparisonChat,
  };

  const setSelectedPage = (component) => {
    selectedPage.set(component);
  };

  // Attempt to fetch components to load from server
  fetch("/components")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      components.set(data);
      selectedPage.set(data[0]);
    })
    .catch((error) => {
      console.log("Fetch request failed", error);
    });
</script>

<!-- tabs for components -->
<!-- only draw if more than one -->
{#if $components.length > 1}
  {#each $components as component}
    <button
      aria-label={`Load ${component.svelte_component} component`}
      on:click={() => setSelectedPage(component)}
      >{component.svelte_component}</button
    >
  {/each}
{/if}

<!-- Loaded selected component (tab) -->
{#if $selectedPage}
  <svelte:component
    this={componentMap[$selectedPage.svelte_component]}
    {...$selectedPage.props}
  />
{/if}
