<script>
  import { chatLog, checkedDocs } from "../../store";
  import { onMount } from "svelte";
  import { select } from "d3-selection";
  import { writable } from "svelte/store";
  import Dropdown from "./Components/Dropdown.svelte";
  import { tooltip } from "../../utils.js";
  import SourceContainer from "./Components/SourceContainer.svelte";
  import Tabs from "../UIComponents/Tabs.svelte";
  import ModifiedAnswer from "./Components/ModifiedAnswer.svelte";
  import Answer from "./Components/Answer.svelte";

  export let feedback = false;
  export let is_retrieval = false;

  const uploadedFiles = writable([]);

  let mymessage = "";
  let messageplaceholder = "";
  let chatLoading = false;
  let editting_response = true;

  let items = [
    {
      label: "Answer",
      value: 1,
      component: Answer,
    },
    {
      label: "Modified Answer",
      value: 2,
      component: ModifiedAnswer,
    },
  ];

  onMount(() => {
    getDataFromDB();
    loadRetrievalFiles();
  });

  async function loadRetrievalFiles() {
    const response = await fetch("/retrieval/file/get");
    const data = await response.json();
    console.log("data", data["files"]);
    const fileData = data["files"];
    const fileNames = fileData.map((row, index) => ({
      id: String(index),
      name: row.name,
    }));
    console.log("files", fileNames);
    $uploadedFiles = [...fileNames];
  }

  async function getDataFromDB() {
    const response = await fetch("/chat/rag_table/retrieve");
    const data = await response.json();
    const dbRows = data["rows"];
    console.log("Got data from db", dbRows);
    const formattedRows = dbRows.map((row) => ({
      id: row[0],
      question: row[1],
      answer: row[2],
      edited_answer: row[3],
      vote_status: row[4],
      rag_sources: row[5],
      source: row[6],
      source_content: row[7],
    }));
    $chatLog = [...formattedRows];
  }

  const askModel = async (event) => {
    event.preventDefault(); // Prevents page refresh
    mymessage = messageplaceholder;
    messageplaceholder = "";
    chatLoading = true;
    const file_names = [...$checkedDocs];
    let currentEntry = {
      id: $chatLog.length + 1,
      question: mymessage,
      answer: "Loading...",
      rag_sources: file_names,
      vote_status: "na",
      source: ["Loading..."],
      source_content: ["Loading..."],
    };
    $chatLog = [...$chatLog, currentEntry];

    const response = is_retrieval
      ? await fetch(`/retrieval/new_message`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: mymessage,
            file_names: file_names,
          }),
        })
      : await fetch(`/chat/${mymessage}`, {
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
      console.log("response data", data);
      currentEntry["answer"] = data["answer"];
      currentEntry["source"] = data["source"];
      currentEntry["source_content"] = data["source_content"];
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

  function getRAGSources(message) {
    if (message.rag_sources.length === 0) return "No Sources";
    const ragSources = message.rag_sources;
    const ragSourcesString = ragSources.join(", ");
    return ragSourcesString;
  }
</script>

<div class="ranked-feedback-container">
  <div class="instructions">
    <h5 class="underline bold">Vote Feedback Instructions</h5>
    <p>
      Ask a question to receive an answer from the chatbot. If the response is
      satisfactory, click on the <span class="inline-button green">👍</span>
      button. If the response is not satisfactory, click on the
      <span class="inline-button red">👎</span> button.
    </p>
    <!-- <button>Download Data</button> -->
  </div>
  <div class="ranked-chat">
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
                <div class="question">
                  <h5 class="bold">Question:</h5>
                  <p>{message.question}</p>
                  <div class="rag-sources">
                    <p class="bold" use:tooltip={getRAGSources(message)}>
                      ℹ️ Retrieval Sources
                    </p>
                  </div>
                </div>
                <div class="answers">
                  <div class="answer">
                    <Tabs {items} tabProps={{ message, feedback, index }} />
                  </div>
                </div>
                <SourceContainer
                  sources={message.source}
                  source_content={message.source_content}
                />
              </div>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <div class="chat-input-holder">
      <div class="chat-and-question">
        <Dropdown documents={$uploadedFiles} />

        <form on:submit={askModel} class="chat-input-form">
          <input
            bind:value={messageplaceholder}
            class="chat-input-textarea"
            placeholder="Type Question Here"
          />
          <button
            class="btnyousend {messageplaceholder === '' ? '' : 'active'}"
            type="submit">{chatLoading ? dots : "Send"}</button
          >
        </form>
      </div>
      <p class="message">Note - may produce inaccurate information.</p>
    </div>
  </div>
</div>

<style>
  .chat-and-question {
    display: grid;
    grid-template-columns: 20% 80%;
    width: 100%;
  }
  /* .small-button {
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
  } */
  .ranked-chat {
    height: 100vh;
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 80% 20%;
  }

  .message {
    font-size: var(--smallText);
    padding-left: 40px;
    padding-right: 40px;
    margin: 0 auto;
  }

  .chat-input-holder {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 5px;
    width: 100%;
    max-width: 820px;
    margin: auto;
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
    flex: 3;
    border-radius: 0px;
    border-right: 0px;
  }

  .chat-input-form {
    display: flex;
    width: 100%;
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

  .green {
    border-bottom: 2px solid var(--green);
  }

  .red {
    border-bottom: 2px solid var(--red);
  }

  .instructions {
    text-align: center;
    padding: 5%;
  }

  .instructions h5 {
    text-align: left;
  }

  .instructions p {
    font-size: var(--smallText);
    text-align: left;
  }

  .instructions button {
    font-size: var(--smallText);
  }

  .ranked-feedback-container {
    display: grid;
    grid-template-columns: 20% 80%;
  }

  .underline {
    border-bottom: var(--line);
  }

  :global(.bold) {
    font-weight: bold;
    font-size: var(--smallText);
    margin: 0;
    padding: 0;
  }

  .chatbox {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: calc(100vh - var(--headerHeight));
    background-color: var(--white);
    box-sizing: border-box;
    width: 95%;
    margin: auto;
    height: 100%;
  }

  .chat-log {
    flex: 1;
    overflow-y: auto;
    padding: 0 10px;
    box-sizing: border-box;
  }

  .chat-message {
    background-color: var(--white);
    border-bottom: var(--line);
    box-sizing: border-box;
  }

  .chat-message-center {
    display: flex;
    flex-direction: column;
    margin-left: auto;
    margin-right: auto;
    padding: 12px;
    box-sizing: border-box;
  }

  .message-content {
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
  }

  .message-content .question {
    text-align: left;
    border: 1px solid var(--grey);
    padding: 5px;
    /* margin-bottom: 10px; */
    background-color: var(--lightGrey);
  }

  .message-content .answer {
    display: inline-block;
    text-align: left;
    padding: 10px;
    border: 1px solid var(--black);
  }

  .message-content .answers {
    display: grid;
    grid-template-columns: 100%;
    gap: 0%;
    width: 100%;
    margin: auto;
  }

  .rag-sources {
    display: flex;
  }

  :global(.tooltip) {
    white-space: nowrap;
    position: relative;
    padding-top: 0.35rem;
    cursor: zoom-in;
  }

  :global(#tooltip) {
    position: absolute;
    bottom: 100%;
    right: 0.78rem;
    transform: translate(calc(100% - 120px), 0);
    padding: 0.2rem 0.35rem;
    background: hsl(0, 0%, 20%);
    color: hsl(0, 0%, 98%);
    font-size: 0.95em;
    border-radius: 0.25rem;
    filter: drop-shadow(0 1px 2px hsla(0, 0%, 0%, 0.2));
    width: max-content;
  }

  :global(.tooltip #tooltip::before) {
    content: "";
    position: absolute;
    top: 100%;
    left: 10px;
    width: 0.6em;
    height: 0.25em;
    background: inherit;
    clip-path: polygon(0% 0%, 100% 0%, 50% 100%);
  }
</style>
