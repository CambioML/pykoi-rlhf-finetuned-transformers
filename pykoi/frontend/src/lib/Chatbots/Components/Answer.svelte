<script>
    import { chatLog } from "../../../store.js";
    import { select } from "d3-selection";

    export let message = {};
    export let feedback = false;
    export let index = 0;

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

        select(event.currentTarget.parentNode)
            .selectAll("button")
            .style("border", "3px solid transparent")
            .style("opacity", 0.65);
        select(event.currentTarget)
            .style("border", "3px solid var(--black)")
            .style("opacity", 1);
    }
</script>

<div>
    {message.answer}
</div>
{#if feedback}
    <div class="feedback-buttons">
        <button
            on:click={(event) => logVote(event, "up", index)}
            class="small-button thumbs-up">👍</button
        >
        <button
            on:click={(event) => logVote(event, "down", index)}
            class="small-button thumbs-down">👎</button
        >
    </div>
{/if}

<style>
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
</style>
