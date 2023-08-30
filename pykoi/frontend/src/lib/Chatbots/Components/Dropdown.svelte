<script>
  export let documents = [];

  let expanded = false;
  let checkboxes; // This will hold our dropdown reference

  function toggleCheckboxes() {
    expanded = !expanded;
  }
</script>

<form>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <div class="multiselect">
    <div class="selectBox" on:click={toggleCheckboxes}>
      <select>
        <option>Documents</option>
      </select>
      <div class="overSelect" />
    </div>
    <div
      bind:this={checkboxes}
      class="dropdown-content"
      style="display: {expanded ? 'block' : 'none'};"
    >
      {#each documents as doc, index}
        <label for={doc.id}>
          <input type="checkbox" id={doc.id} />{doc.name}
        </label>
      {/each}
    </div>
  </div>
</form>

<style>
  .multiselect {
    position: relative;
    max-width: 200px;
  }

  .selectBox {
    position: relative;
  }

  .selectBox select {
    width: 100%;
    font-weight: bold;
  }

  .overSelect {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
  }

  .dropdown-content {
    position: absolute; /* has to be abs to prevent document overflow */
    top: 100%;
    left: 0;
    width: 100%;
    border: 1px #dadada solid;
    background-color: white;
    z-index: 1;
  }
</style>
