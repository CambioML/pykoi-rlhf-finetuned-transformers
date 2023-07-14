<!-- Text.svelte -->
<script>
  import { state } from "../../store";
  export let text = "";

  // Reactive statement that updates the text whenever 'text' changes
  let newText;
  $: {
    newText = text.replace(/\['(.+?)'\]/g, (_, varName) => {
      // Access the store value directly.
      // The enclosing ${ and } are required to use the value of the variable as a key
      return state[varName] ? `$state['${varName}']` : _;
    });
  }
</script>

<p>{newText}</p>
