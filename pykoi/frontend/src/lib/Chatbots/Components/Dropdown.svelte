<script>
  import { writable } from "svelte/store";
  import { checkedDocs } from "../../../store";
  import { selectAll } from "d3-selection";
  import { tooltip } from "../../../utils.js";
  import { clickOutside } from "../../../utils.js";

  export let documents = [];

  let expanded = false;
  let checkboxes; // This will hold our dropdown reference

  function toggleCheckboxes(e) {
    e.preventDefault();
    expanded = !expanded;
  }

  function handleCheckboxChange(docName, event) {
    if (event.target.checked) {
      $checkedDocs.add(docName);
    } else {
      $checkedDocs.delete(docName);
    }
    checkedDocs.set(new Set($checkedDocs)); // Trigger an update to the store
  }

  function handleSelectAll(e) {
    e.preventDefault();
    $checkedDocs = new Set(documents.map((doc) => doc.name));
    checkedDocs.set(new Set($checkedDocs)); // Trigger an update to the store
  }

  function handleUnselectAll(e) {
    e.preventDefault();
    $checkedDocs = new Set();
    checkedDocs.set(new Set($checkedDocs)); // Trigger an update to the store
  }

  function centerTruncate(text) {
    const maxLen = 16;
    const halfLen = Math.floor(maxLen / 2);
    if (text.length > maxLen) {
      return text.slice(0, halfLen) + "..." + text.slice(-halfLen);
    }
    return text;
  }

  function handleClickOutside(e) {
    e.preventDefault();
    expanded = false;
  }

  $: console.log($checkedDocs);
</script>

<form>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <div class="multiselect" use:clickOutside on:click_outside={handleClickOutside}>
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
      {#if documents.length === 0}
        <div>
          No documents found. Please upload via the RetrievalQA component.
        </div>
      {:else}
        <div class="select-button-container">
          <button on:click={handleSelectAll}>Select All</button>
          <button on:click={handleUnselectAll}>Deselect All</button>
        </div>
        <div class="checkbox-container">
          {#each documents as doc, index}
            <label for={doc.id}>
              <!-- Use a checked attribute and a change handler instead of two-way binding -->
              <input
                type="checkbox"
                id={doc.id}
                checked={$checkedDocs.has(doc.name)}
                on:change={(event) => handleCheckboxChange(doc.name, event)}
              />
              <span use:tooltip={doc.name}>{centerTruncate(doc.name)}</span>
            </label>
          {/each}
        </div>
      {/if}
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
    bottom: 100%;
    left: 0;
    width: 100%;
    border: 1px #dadada solid;
    background-color: white;
    z-index: 1;
    padding: 0.5em;
  }
  .checkbox-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }

  .checkbox-container label {
    display: inline-block;
    padding-right: 10px;
    white-space: nowrap;
  }
  .checkbox-container input {
    cursor: pointer;
    vertical-align: middle;
  }
  .checkbox-container label span {
    cursor: pointer;
    vertical-align: middle;
  }

  .select-button-container {
    width: 100%;
    display: flex;
    justify-content: center;
    padding-top: 1em;
  }
  .select-button-container button {
    color: var(--darkGrey);
    font-size: small;
    margin: 0 2px;
    padding: 1em;
  }
</style>
