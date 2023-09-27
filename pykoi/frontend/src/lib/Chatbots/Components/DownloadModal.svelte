<script>
    import Modal from "../../UIComponents/Modal.svelte";
    export let showModal, table;
    let dialog;
    let file_name = "";

    const handleSubmit = async (e) => {
        e.preventDefault();
        dialog.close();
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
    };
</script>

<Modal bind:showModal bind:dialog>
    <h4 slot="header">Download Data</h4>
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
