<script>
  import Bubble from "./Components/Charts/Bubble.svelte";
  import Table from "./Components/tanstackTable/Table.svelte";
  import { onMount } from "svelte";
  import { uploadedFiles, projections } from "../store.js";

  let selectedFiles = [];
  let indexed = false;
  let indexing = false;
  async function handleFileChange(event) {
    selectedFiles = event.target.files;
    const formData = new FormData();
    // iterate over selectedFiles and add them to array
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("files", selectedFiles[i]);
    }
    const response = await fetch("/retrieval/file/upload", {
      method: "POST",
      body: formData,
    });
    indexFiles();
    loadServerData();
    getEmbeddings();
  }

  async function loadServerData() {
    const response = await fetch("/retrieval/file/get");
    const data = await response.json();
    // Transform the received data
    const filesData = data.files.map((file) => {
      return {
        file: file.name,
        size: file.size, // Random file size
        type: file.type,
      };
    });
    $uploadedFiles = [...filesData];
  }

  async function indexFiles() {
    console.log("index!");
    indexing = true;
    const response = await fetch("/retrieval/vector_db/index", {
      method: "POST",
    });
    const data = await response.json();
    indexed = true;
    indexing = false;
  }

  async function getEmbeddings() {
    console.log("getting embeddings...");
    const response = await fetch("/retrieval/vector_db/get");
    const embeddingData = await response.json();
    console.log("embeddingData", embeddingData);
    $projections = embeddingData;
  }

  onMount(() => {
    loadServerData();
  });

  let dotState = 0;

  // Set an interval to periodically change the number of dots
  setInterval(() => {
    dotState = (dotState + 1) % 4;
  }, 200);

  // Use a reactive statement to create the string with the correct number of dots
  $: dots = "Indexing" + ".".repeat(dotState);
</script>

<div class="data-grid">
  <div class="file-container">
    <div class="upload-container">
      <div class="upload-box">
        <h4>Upload Data</h4>
        <br />
        <form>
          <input type="file" multiple on:change={handleFileChange} />
        </form>
        {#if indexing && !indexed}
          <p>{dots}</p>
        {/if}
        {#if indexed}
          <p>Data Successfully indexed!</p>
        {/if}
        <p>These are the files your model will use as context.</p>
        <p>Currently <strong>pdf</strong>, txt, and md are supported.</p>
      </div>
    </div>
  </div>
  <div class="charts-container">
    <Bubble />
    {#if selectedFiles}
      <Table />
    {/if}
  </div>
</div>

<style>
  .file-container {
    display: grid;
    height: calc(100% - var(--headerHeight));
    align-items: center; /* Align vertically */
    justify-content: center; /* Align horizontally */
  }

  .upload-container {
    margin: auto;
    max-width: 100%;
    text-align: center;
  }
  .charts-container {
    height: calc(100vh - var(--headerHeight));
    display: grid;
    gap: 2%;
    grid-template-columns: 100%;
    grid-template-rows: 50% 40%;
  }

  .data-grid {
    display: grid;
    grid-template-columns: 45% 50%;
    gap: 0;
    margin: auto;
    max-width: 1200px;
    padding-top: 20px;
  }

  .upload-box {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    max-height: 50vh;
    margin: auto;
    border: 5px dashed var(--grey);
    padding: 20px;
    box-sizing: border-box;
  }
</style>
