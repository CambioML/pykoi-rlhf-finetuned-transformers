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
            <p class="question">Q: {message.question}</p>
            <div class="answers">
              <p class="answer">1: {message.answer}</p>
              <p class="answer">2: {message.answer}</p>
            </div>
          </div>
        </div>
      </div>
    {/each}
  </div>
</section>

<style>
  .chatbox {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: calc(100vh - var(--headerHeight));
    background-color: var(--white);
    box-sizing: border-box;
    width: 75%;
    margin: auto;
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
    border: 1px solid #000;
    padding: 5px;
    border-radius: 5px;
    margin-bottom: 10px;
  }

  .message-content .answer {
    display: inline-block;
    text-align: left;
    background-color: var(--lightGrey);
    padding: 10px;
    border: 1px solid var(--black);
    margin: 5px 0;
  }

  .message-content .answers {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }
</style>
