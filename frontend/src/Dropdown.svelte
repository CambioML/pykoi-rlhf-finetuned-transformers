<!-- Dropdown.svelte -->
<script>
  import { onMount } from "svelte";
  export let data_endpoint; // receive the prop from the parent
  let options = [];

  onMount(async () => {
    const response = await fetch(`/data/${data_endpoint}`);
    options = await response.json();
  });
</script>

<select>
  {#each options as option}
    <option>{option}</option>
  {/each}
</select>
