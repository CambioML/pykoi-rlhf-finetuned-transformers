<!-- Header.svelte -->
<script>
  import { state } from "../../store";

  export let level;
  export let text;

  const levelNum = +level.slice(1);
  const isStore = Array.isArray(text);
</script>

{#if levelNum === 1}
  <h1>{isStore ? $state[text] : text}</h1>
{:else if levelNum === 2}
  <h2>{isStore ? $state[text] : text}</h2>
{:else if levelNum === 3}
  <h3>{isStore ? $state[text] : text}</h3>
{:else if levelNum === 4}
  <h4>{isStore ? $state[text] : text}</h4>
{:else if levelNum === 5}
  <h5>{isStore ? $state[text] : text}</h5>
{:else}
  <h6>{isStore ? $state[text] : text}</h6>
{/if}
