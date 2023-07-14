<!-- Dropdown.svelte -->
<script>
  import { onMount } from "svelte";
  let options = [];

  onMount(async () => {
    const response = await fetch("/data/dropdown");
    options = await response.json();
  });
</script>

<select>
  {#each options as option}
    <option>{option}</option>
  {/each}
</select>
