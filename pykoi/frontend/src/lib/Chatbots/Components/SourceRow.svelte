<script>
    import { slide } from "svelte/transition";
    export let source = "";
    export let source_content = "";
    export let i = 0;
    let show_content = i === 0;
</script>

<div class="source">
    <div class="source_tab" on:click={() => (show_content = !show_content)} transition:slide|global>
        <p class="bold">{i + 1}: {source}</p>
        {#if show_content}
            <p>&#8963;</p>
        {:else}
            <p>&#8964;</p>
        {/if}
    </div>
    {#if show_content}
        <div class="source_content" transition:slide>
            <p class="bold">{source_content}</p>
        </div>
    {/if}
</div>

<style>
    .source {
        text-align: left;
        background-color: var(--lightGrey);
        border: 1px solid var(--grey);
        border-top: none;
        padding: 0px 5px;
        margin: 0px;
        color: var(--darkGrey);
        box-sizing: border-box;
    }

    .source_tab {
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
    }

    .source_content {
        background-color: white;
        border: 1pt solid var(--grey);
        padding: 5px;
        margin-bottom: 5px;
        background-color: white;
        color: var(--darkGrey);
    }

    p {
        margin: 0;
        padding: 0;
    }
    .source:nth-of-type(2) {
        border-top: 1px solid var(--grey);
    }
</style>
