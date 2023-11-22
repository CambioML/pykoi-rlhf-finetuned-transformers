<script>
    import { chatLog } from "../../../store.js";

    export let message = {};
    export let feedback = false;
    export let index = 0;
    export let title = false;

    async function insertVote(feedbackUpdate) {
        const response = await fetch("/chat/rag_table/update", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(feedbackUpdate),
        });

        if (response.ok) {
            console.log("response", response);
            message.vote_status = feedbackUpdate.vote_status;
        } else {
            const err = await response.text();
            alert(err);
        }
    }

    function logVote(event, vote, index) {
        const messageLog = $chatLog[index];
        messageLog.vote = vote;
        const feedbackUpdate = {
            id: index + 1, // increment bc sqlite 1-indexed
            vote_status: vote,
        };
        insertVote(feedbackUpdate);
    }
    console.log("Answer", message.vote_status);
</script>

<div class="answer">
    {#if title}
        <h5 class="bold">Response:</h5>
    {/if}
    <p>{message.answer}</p>
    {#if feedback}
        <div class="feedback-buttons">
            <button
                on:click={(event) => logVote(event, "up", index)}
                class="small-button thumbs-up"
                class:vote-selected={message.vote_status === "up"}
                class:vote-not-selected={message.vote_status === "down"}
                >👍</button
            >
            <button
                on:click={(event) => logVote(event, "down", index)}
                class="small-button thumbs-down"
                class:vote-selected={message.vote_status === "down"}
                class:vote-not-selected={message.vote_status === "up"}
                >👎</button
            >
        </div>
    {/if}
</div>

<style>
    .answer {
        display: inline-block;
        text-align: left;
        padding: 10px;
        border: 1px solid var(--black);
        width: 100%;
    }
    .small-button {
        margin-left: 10px;
        background: none;
        border: 3px solid transparent;
        color: inherit;
        padding: 6px 10px;
        cursor: pointer;
        box-shadow: none;
        font-size: var(--smallText);
    }

    .feedback-buttons {
        display: flex;
        text-align: center;
        margin: auto;
        width: 20%;
    }

    .small-button:hover {
        box-shadow: var(--shadow-md);
    }

    .thumbs-up,
    .thumbs-up:hover,
    .thumbs-up::selection {
        background: var(--green);
    }
    .thumbs-down,
    .thumbs-down:hover,
    .thumbs-down::selection {
        background: var(--red);
    }

    .vote-selected {
        border: 3px solid black;
        opacity: 1;
    }

    .vote-not-selected {
        border: 3px solid transparent;
        opacity: 0.65;
    }
</style>
