<script>
    export let showModal; // boolean
    export let dialog; // HTMLDialogElement
    export let handleClose;
    $: if (dialog && showModal) dialog.showModal();
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
<dialog
    bind:this={dialog}
    on:close={handleClose}
    on:click|self={handleClose}
>
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div on:click|stopPropagation>
        <!-- svelte-ignore a11y-autofocus -->
        <div class="btn-container">
            <button class="close-button" on:click={handleClose}>X</button>
        </div>
        <slot name="header" />
        <slot />
    </div>
</dialog>

<style>
    dialog {
        max-width: 32em;
        border-radius: 1em;
        border: none;
        padding: 0;
    }
    dialog::backdrop {
        background: rgba(0, 0, 0, 0.3);
    }
    dialog > div {
        padding: 1em;
    }
    dialog[open] {
        animation: zoom 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    @keyframes zoom {
        from {
            transform: scale(0.95);
        }
        to {
            transform: scale(1);
        }
    }
    dialog[open]::backdrop {
        animation: fade 0.2s ease-out;
    }
    @keyframes fade {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    button {
        display: block;
    }

    .close-button {
        display: block;
        height: 30px;
        width: 30px;
        border-radius: 50%;
        border: 1px solid var(--grey);
        padding: 0;
    }

    .close-button:hover {
        background-color: var(--lightGrey);
    }

    .btn-container {
        display: flex;
        justify-content: flex-end;
    }
</style>
