<script>
    import SourceRow from "./SourceRow.svelte";
    export let sources = [];
    export let source_content = [];
    let show_content = false;
</script>

<div class="sources">
    {#if sources !== undefined}
        {#if sources[0] === "N/A"}
            <h5 class="bold">⚠️ No Retrieval Sources selected</h5>
        {:else if sources[0] === "Loading..."}
            <h5 class="bold">📖 Loading...</h5>
        {:else if sources[0] === "Not loaded"}
            <h5 class="bold">⚠️ No sources loaded</h5>
        {:else}
            <div
                class="sources-header"
                on:click={() => (show_content = !show_content)}
            >
                <h5 class="bold">📖 Response Sources ({sources.length})</h5>
                {#if show_content}
                    <span>&#8963;</span>
                {:else}
                    <span>&#8964;</span>
                {/if}
            </div>
            {#if show_content}
                {#each sources as source, i}
                    <SourceRow
                        {source}
                        source_content={source_content[i]}
                        {i}
                    />
                {/each}
            {/if}
        {/if}
    {/if}
</div>

<style>
    .sources {
        display: inline-block;
        text-align: left;
        padding: 5px;
        border: 1pt solid var(--grey);
    }
    .sources-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
    }
</style>
