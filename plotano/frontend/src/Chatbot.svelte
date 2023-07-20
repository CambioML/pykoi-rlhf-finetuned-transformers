<script>
  import { chatLog } from "./store";
  import { onMount } from "svelte";

  export let feedback = false;

  let mymessage = "";
  let messageplaceholder = "";
  let chatLoading = false;

  onMount(() => {
    getDataFromDB();
  });

  async function getDataFromDB() {
    console.log("fetching data from db");
    const response = await fetch(
      "http://127.0.0.1:5000/chat/qa_table/retrieve"
    );
    const data = await response.json();
    const dbRows = data["rows"];
    const formattedRows = dbRows.map((row) => ({
      id: row[0],
      question: row[1],
      answer: row[2],
      vote_status: row[3],
    }));
    $chatLog = [...formattedRows];
  }

  // insert entry into database
  // async function insertToDatabase(entry) {
  //   console.log("INSERT");
  //   console.log("entry", entry);
  //   const response = await fetch(
  //     "http://127.0.0.1:5000/preference_table/insert",
  //     {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //       body: JSON.stringify(entry),
  //     }
  //   );

  //   if (response.ok) {
  //     console.log("ok! data inserted", response);
  //     console.log("entry", entry);
  //     // update chatlog
  //     // $chatLog = [...$chatLog, entry];
  //   } else {
  //     const err = await response.text();
  //     alert(err);
  //   }
  // }

  const askModel = async (event) => {
    event.preventDefault(); // Prevents page refresh
    mymessage = messageplaceholder;
    messageplaceholder = "";
    chatLoading = true;
    let currentEntry = {
      id: $chatLog.length + 1,
      question: mymessage,
      answer: "Loading...",
      vote_status: "na",
    };
    console.log("adding to log here");
    $chatLog = [...$chatLog, currentEntry];

    const response = await fetch(`/chat/${mymessage}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: mymessage,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      currentEntry["answer"] = data["answer"];
      // $chatLog[$chatLog.length - 1] = currentEntry;
      chatLog.update((state) => {
        state[state.length - 1] = currentEntry;
        return state;
      });
      //   insertToDatabase(currentEntry);
    } else {
      const err = await response.text();
      alert(err);
    }
    chatLoading = false;
  };

  function scrollToView(node) {
    setTimeout(() => {
      node.scrollIntoView({ behavior: "smooth" });
    }, 0);
  }

  let dotState = 0;

  setInterval(() => {
    dotState = (dotState + 1) % 4;
  }, 200);

  $: dots = ".".repeat(dotState).padEnd(3);

  async function logVote(vote, index) {
    const messageLog = $chatLog[index];
    messageLog.vote = vote;
    const feedbackUpdate = {
      id: index + 1, // increment bc sqlite 1-indexed
      vote_status: vote,
    };
    console.log(feedbackUpdate);
    console.log(feedbackUpdate);
    const response = await fetch("http://127.0.0.1:5000/chat/qa_table/update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(feedbackUpdate),
    });

    if (response.ok) {
      console.log("update worked!", response);
    } else {
      const err = await response.text();
      alert(err);
    }
  }

  $: console.log("chat updated!", $chatLog);
</script>

<section class="chatbox">
  <div class="chat-log">
    {#each $chatLog as message, index (index)}
      <div
        class="chat-message"
        use:scrollToView={index === $chatLog.length - 1}
      >
        <div class="chat-message-center">
          <div class="avatar">
            <!-- <img src={logo} alt="SvelteKit" /> -->
          </div>
          <div class="message-content">
            <p>Q: {message.question}</p>
            <p>A: {message.answer}</p>
            {#if feedback}
              <button
                on:click={() => logVote("up", index)}
                class="small-button thumbs-up">👍</button
              >
              <button
                on:click={() => logVote("down", index)}
                class="small-button thumbs-down">👎</button
              >
            {/if}
          </div>
        </div>
      </div>
    {/each}
  </div>

  <div class="chat-input-holder">
    <form on:submit={askModel} class="chat-input-form">
      <input
        bind:value={messageplaceholder}
        class="chat-input-textarea"
        placeholder="Type Message Here"
      />
      <button
        class="btnyousend {messageplaceholder === '' ? '' : 'active'}"
        type="submit">{chatLoading ? dots : "Send"}</button
      >
    </form>
    <p class="message">Note - may produce inaccurate information.</p>
  </div>
</section>

<!-- STYLING -->
<style>
  .small-button {
    margin-left: 10px;
    background: none;
    border: none;
    color: inherit;
    padding: 6px 10px;
    cursor: pointer;
    box-shadow: none;
    font-size: calc(var(--baseFontSize) * 1.1);
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
  .message {
    font-size: var(--smallText);
    padding: 0;
    margin: 0 auto;
  }
  .chatbox {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: calc(100vh - var(--headerHeight));
    background-color: var(--white);
  }

  .chat-input-holder {
    display: flex; /* Add this line */
    justify-content: space-between; /* Add this line to align items to the left and right of the holder */
    padding: 24px;
    width: 100%;
    max-width: 640px;
    margin: auto;
    text-align: center;
  }

  .chat-input-textarea {
    background-color: var(--lightgrey);
    cursor: pointer;
    width: 100%;
    border-radius: 5px;
    border: var(--line);
    border-color: none;
    margin: 12px;
    outline: none;
    padding: 12px;
    color: var(--black);
    font-size: var(--baseFontSize);
    box-shadow: var(--shadow-md);
  }

  .chat-log {
    flex: 1;
    overflow-y: auto;
    text-align: left;
  }

  /*  */

  .chat-message {
    background-color: var(--white);
    border-bottom: var(--line);
  }
  .chat-message.chatgpt {
    background-color: var(--lightGrey);
    color: var(--black);
  }

  .chat-message-center {
    display: flex;
    max-width: 640px;
    margin-left: auto;
    margin-right: auto;
    padding: 12px;
    padding-left: 24px;
    padding-right: 24px;
    align-items: center;
    /* border-top: var(--line); */
  }

  .avatar,
  img {
    background: var(--lightgrey);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    margin-right: 5px;
  }

  .avatar.chatgpt {
    background: var(--primary);
    border-radius: 50%;
    width: 40px;
    height: 40px;
  }

  .message {
    padding-left: 40px;
    padding-right: 40px;
  }

  .chat-input-holder {
    display: flex;
    flex-direction: column; /* Stacks children vertically */
    align-items: center; /* Center the items */
    padding: 24px;
    width: 100%;
    max-width: 640px;
    margin: auto;
  }

  .chat-input-form {
    display: flex; /* Make the input and button side by side */
    width: 100%; /* Take the full width of the parent */
  }

  .btnyousend {
    border-radius: 0px;
    margin-top: 12px;
    margin-bottom: 12px;
    margin-left: -15px;
    background: var(--primary);
    color: var(--black);
    opacity: 0.5;
    transition: all 0.3s;
  }

  .active {
    opacity: 1;
  }

  .chat-input-textarea {
    /* Input will take the remaining space */
    flex: 3;
    border-radius: 0px;
    border-right: 0px;
  }

  .message-content {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: start;
  }

  .message-content {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
  }

  .message-content p {
    margin-right: 10px;
  }
</style>
