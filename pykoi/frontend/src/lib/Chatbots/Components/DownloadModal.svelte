<script>
    import Modal from "../../UIComponents/Modal.svelte";
    export let showModal, table;
    let dialog;
    let file_name = "";
    const DOWNLOAD_STATE = {
        FILE_INPUT: 0,
        DOWNLOADED: 1,
        FAILED_DOWNLOAD: 2,
        OVERWRITE: 3,
    };
    let downloadState = DOWNLOAD_STATE.FILE_INPUT;

    const saveToCSV = async (file_name) => {
        const request_body = {
            file_name,
        };
        const response = await fetch(`/chat/${table}/save_to_csv`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(request_body),
        });
        const data = await response.json();
        console.log("Download Complete: ", data);
        if (data.status === "200") {
            console.log("success");
            downloadState = DOWNLOAD_STATE.DOWNLOADED;
        } else {
            console.log("failed");
            downloadState = DOWNLOAD_STATE.FAILED_DOWNLOAD;
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await fetch(
            `/file_exists/?file_name=${file_name}.csv`
        );
        const { file_exists } = await response.json();
        if ( file_exists === true) {
            downloadState = DOWNLOAD_STATE.OVERWRITE;
        } else {
            saveToCSV(file_name);
        }
    };

    function handleClose() {
        showModal = false;
        downloadState = DOWNLOAD_STATE.FILE_INPUT;
        dialog.close();
    }
</script>

<Modal bind:showModal bind:dialog {handleClose}>
    <h4 slot="header">Download Data</h4>
    {#if downloadState === DOWNLOAD_STATE.FILE_INPUT}
        <form on:submit={handleSubmit}>
            <div class="inputs">
                <label for="file_name">Filename</label>
                <input
                    bind:value={file_name}
                    type="text"
                    placeholder="Please enter filename"
                    name="file_name"
                    required
                />
            </div>
            <div class="btn-container">
                <button type="submit">Download</button>
            </div>
        </form>
    {/if}
    {#if downloadState === DOWNLOAD_STATE.DOWNLOADED}
        <div>
            ✅ Data downloaded to /pykoi/{file_name}.csv
        </div>
        <div class="btn-container">
            <button on:click={handleClose}>Close</button>
        </div>
    {/if}
    {#if downloadState === DOWNLOAD_STATE.FAILED_DOWNLOAD}
        <div>⚠️ Download failed. Please try again.</div>
        <div class="btn-container">
            <button on:click={() => (downloadState = DOWNLOAD_STATE.FILE_INPUT)}>Retry</button>
            <button on:click={handleClose}>Close</button>
        </div>
    {/if}
    {#if downloadState === DOWNLOAD_STATE.OVERWRITE}
        <div>
            ⚠️ {file_name}.csv already exists. Do you wish to overwrite it?
        </div>
        <div class="btn-container">
            <button on:click={() => (downloadState = DOWNLOAD_STATE.FILE_INPUT)}>Back</button>
            <button on:click={() => saveToCSV(file_name)}>Overwrite</button>
        </div>
    {/if}
</Modal>

<style>
    .btn-container {
        display: flex;
        justify-content: center;
        padding-top: 1em;
    }
    div.inputs {
        display: grid;
        grid-template-columns: max-content max-content;
        grid-gap: 5px;
    }
    div.inputs label {
        text-align: right;
    }
    h4 {
        margin: 20px 0 20px 0;
    }
</style>
