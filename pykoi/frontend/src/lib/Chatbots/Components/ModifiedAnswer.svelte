<script>
    import { onMount } from "svelte";
    export let message = {};
    const answerPlaceholder = "";
    let elm;

    const updateAnswer = async (newAnswer) => {
        const answerUpdate = {
            id: message.id,
            new_answer: newAnswer,
        };
        const response = await fetch("/chat/rag_table/update_answer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(answerUpdate),
        });

        if (response.ok) {
            console.log("Answer updated successfully", response);
            message.edited_answer = newAnswer;
        } else {
            const err = await response.text();
            alert(err);
        }
    };

    const handleUpdate = (e) => {
        e.preventDefault();
        updateAnswer(message.edited_answer);
    };

    const handleReset = (e) => {
        e.preventDefault();
        updateAnswer(answerPlaceholder);
    };

    const useText = (e) => {
        e.preventDefault();
        message.edited_answer = message.answer;
    };

    const handleKeystroke = (e) => {
        if (e.key == "Enter" && message.edited_answer === answerPlaceholder) {
            e.preventDefault();
            console.log("ENTER");
            message.edited_answer = message.answer;
        }
    };

    onMount(function () {
        elm.focus();
    });
</script>

<form>
    <textarea
        bind:value={message.edited_answer}
        placeholder={message.answer}
        on:keydown={handleKeystroke}
        bind:this={elm}
    />
    <div class="button-container">
        <div class="note">
            {#if message.edited_answer === answerPlaceholder}
                Press ENTER to autofill with the RAG answer.
            {/if}
        </div>
        <div>
            <button on:click={handleUpdate}>Update</button>
            <button on:click={handleReset}>Reset</button>
        </div>
    </div>
</form>

<style>
    .button-container {
        display: flex;
        justify-content: space-between;
    }

    .note {
        font-size: var(--smallText);
        color: var(--gray);
    }
</style>
