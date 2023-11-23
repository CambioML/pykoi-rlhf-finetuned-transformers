<script>
  import Bubble from "./Components/Charts/Bubble.svelte";
  import Table from "./Components/tanstackTable/Table.svelte";
  import { onMount } from "svelte";
  import { uploadedFiles, projections } from "../store.js";
  import CloudArrowUp from "../../assets/CloudArrowUp.svelte";
  import { slide } from "svelte/transition";

  const UPLOAD_STATES = {
    WAITING: "waiting",
    IN_PROGRESS: "in-progress",
    DONE: "done",
  };

  let selectedFiles = [];
  let uploadState = UPLOAD_STATES.WAITING;
  let indexState = UPLOAD_STATES.WAITING;
  let loadState = UPLOAD_STATES.WAITING;
  let embedState = UPLOAD_STATES.WAITING;

  function resetStates() {
    uploadState = UPLOAD_STATES.WAITING;
    indexState = UPLOAD_STATES.WAITING;
    loadState = UPLOAD_STATES.WAITING;
    embedState = UPLOAD_STATES.WAITING;
  }

  async function handleFileChange(event) {
    event.preventDefault();
    resetStates();
    uploadState = UPLOAD_STATES.IN_PROGRESS;
    selectedFiles = [];
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
    const data = await response.json();
    console.log("Upload complete! Response:", data);
    uploadState = UPLOAD_STATES.DONE;
    indexFiles();
    loadServerData();
    getEmbeddings();
  }

  async function loadServerData() {
    if (indexState === UPLOAD_STATES.IN_PROGRESS) {
      loadState = UPLOAD_STATES.IN_PROGRESS;
    }
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
    if (loadState === UPLOAD_STATES.IN_PROGRESS) {
      loadState = UPLOAD_STATES.DONE;
    }
  }

  async function indexFiles() {
    console.log("index!");
    indexState = UPLOAD_STATES.IN_PROGRESS;
    const response = await fetch("/retrieval/vector_db/index", {
      method: "POST",
    });
    const data = await response.json();
    indexState = UPLOAD_STATES.DONE;
  }

  async function getEmbeddings() {
    embedState = UPLOAD_STATES.IN_PROGRESS;
    console.log("getting embeddings...");
    const response = await fetch("/retrieval/vector_db/get");
    const embeddingData = await response.json();
    console.log("embeddingData", embeddingData);
    $projections = embeddingData;
    embedState = UPLOAD_STATES.DONE;
  }

  function dragOverHandler(event) {
    event.preventDefault();
  }

  function getSelectedFileNames() {
    let fileNameStr = "";
    for (let i = 0; i < selectedFiles.length; i++) {
      fileNameStr += selectedFiles[i].name + ", ";
    }
    return fileNameStr.slice(0, -2);
  }

  onMount(() => {
    loadServerData();
  });

  $: {
    if (
      uploadState === UPLOAD_STATES.DONE &&
      indexState === UPLOAD_STATES.DONE &&
      loadState === UPLOAD_STATES.DONE &&
      embedState === UPLOAD_STATES.DONE
    ) {
      setTimeout(resetStates, 3000);
    }
  }
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
        {#if uploadState !== UPLOAD_STATES.WAITING}
          <div class="processing-container" transition:slide={{ duration: 250 }}>
            <div>File{selectedFiles.length > 1 ? `s (${selectedFiles.length})` : ""}</div>
            <div class="processing-files">
              {getSelectedFileNames()}
            </div>
          </div>
          <div class="upload-status" transition:slide={{ duration: 250 }}>
            <div class={`loading load-left ${uploadState}`}>Upload</div>
            <div class={`loading ${indexState}`}>Index</div>
            <div class={`loading ${loadState}`}>Load</div>
            <div class={`loading load-right ${embedState}`}>Embed</div>
          </div>
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
    width: 100%;
    height: 100%;
    margin: auto;
    padding: 20px;
    border: 1px solid #333;
    box-sizing: border-box;
    overflow: hidden;
  }
  .upload-files-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 90%;
  }
  .processing-container {
    color: grey;
    display: flex;
    gap: 4px;
    font-size: small;
  }

  .processing-files {
    margin: 0;
    max-width: 280px;
    max-height: 2em;
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    border: 1px solid var(--grey);
    border-radius: 0.1em;
    padding: 0.1em 0.5em;
  }

  .upload-status {
    display: flex;
    gap: 3px;
    justify-content: center;
    align-items: center;
    height: 100%;
    border-radius: 500px;
    padding: 0 10px;
  }

  @keyframes color {
    0% {
      background-color: var(--yellow);
    }
    50% {
      background-color: var(--lightGrey);
    }
    100% {
      background-color: var(--yellow);
    }
  }

  .loading {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    padding: 0 10px;
    border-radius: 2px;
  }
  .waiting {
    background-color: var(--lightGrey);
    color: var(--grey);
  }
  .in-progress {
    background-color: var(--yellow);
    animation-name: color;
    animation-duration: 1s;
    animation-iteration-count: infinite;
  }

  .done {
    background-color: var(--green);
    color: var(--lightGrey);
  }
  .load-left {
    border-top-left-radius: 0.5em;
    border-bottom-left-radius: 0.5em;
  }

  .load-right {
    border-top-right-radius: 0.5em;
    border-bottom-right-radius: 0.5em;
  }
  p {
    margin: 0;
  }
</style>
