<script>
  import { state } from "../../store";
  export let identifier;
</script>

<input bind:value={$state[identifier]} placeholder={identifier} />
