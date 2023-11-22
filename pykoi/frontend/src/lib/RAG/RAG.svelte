<script>
  import Bubble from "./Components/Charts/Bubble.svelte";
  import Table from "./Components/tanstackTable/Table.svelte";
  import { onMount } from "svelte";
  import { uploadedFiles, projections } from "../store.js";
  import CloudArrowUp from "../../assets/CloudArrowUp.svelte";

  let selectedFiles = [];
  let indexed = false;
  let indexing = false;
  async function handleFileChange(event) {
    event.preventDefault();
    let selectedFiles = [];
    if (event.dataTransfer) {
      if (event.dataTransfer.items) {
        // Use DataTransferItemList interface to access the file(s)
        [...event.dataTransfer.items].forEach((item, i) => {
          // If dropped items aren't files, reject them
          if (item.kind === "file") {
            const file = item.getAsFile();
            selectedFiles.push(file);
          }
        });
      } else {
        // Use DataTransfer interface to access the file(s)
        [...event.dataTransfer.files].forEach((file, i) => {
          selectedFiles.push(file);
        });
      }
    } else {
      selectedFiles = event.target.files;
    }
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

  function dragOverHandler(event) {
    console.log("File(s) in drop zone");
    // Prevent default behavior (Prevent file from being opened)
    event.preventDefault();
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
        <p>These are the files your model will use as context.</p>
        <div class="upload-files-container">
          <form>
            <input type="file" multiple on:change={handleFileChange} />
          </form>
          <div
            class="drop-zone"
            on:drop={handleFileChange}
            on:dragover={dragOverHandler}
          >
            <CloudArrowUp height={32} width={32} />
            Drag and drop files here
          </div>
        </div>
        {#if indexing && !indexed}
          <p>{dots}</p>
        {/if}
        {#if indexed}
          <p>Data Successfully indexed!</p>
        {/if}
        <p>Currently pdf, txt, and md are supported.</p>
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
  .drop-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 2px dashed var(--grey);
    background-color: var(--lightGrey);
    width: 100%;
    color: #444;
    min-height: 300px;
  }
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
    gap: 8px;
    margin: auto;
    max-width: 1200px;
    padding-top: 20px;
  }

  .upload-box {
    display: flex;
    gap: 10px;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    margin: auto;
    padding: 20px;
    border: 1px solid #333;
    box-sizing: border-box;
  }
  .upload-files-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 90%;
  }

  p {
    margin: 0;
  }
</style>
