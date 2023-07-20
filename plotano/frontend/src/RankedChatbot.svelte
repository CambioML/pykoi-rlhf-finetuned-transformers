<script>
  import { chatLog } from "./store";
  import { onMount } from "svelte";
  import { select } from "d3-selection";

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

  function handleClick() {
    select(this.parentNode)
      .selectAll("div")
      .style("outline", "2px solid var(--red)")
      .style("border", "1px solid var(--red");
    select(this)
      .style("outline", "2px solid var(--green)")
      .style("border", "1px solid var(--green");
  }
</script>

<div class="ranked-feedback-container">
  <div class="instructions">
    <h5 class="underline bold">Ranked Feedback Instructions</h5>
    <p>
      Ask a question and click on the better of the two responses. The better
      response will be outlined in <span class="green">green</span>, the worse
      response outlined in <span class="red">red</span>. This data will be
      automatically fed to RLHF.
    </p>
    <button>Download Data</button>
  </div>
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
              </div>
              <div class="answers">
                <div class="answer" on:click={handleClick}>
                  <h5 class="bold underline">Response 1:</h5>
                  <p>{message.answer}</p>
                </div>
                <div class="answer" on:click={handleClick}>
                  <h5 class="bold underline">Response 2:</h5>
                  <p>{message.answer}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  </section>
</div>

<style>
  .green {
    border-bottom: 2px solid var(--green);
  }
  .red {
    border-bottom: 2px solid var(--red);
  }
  .instructions {
    /* border: 2px solid black; */
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
  .bold {
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
    margin-bottom: 10px;
    background-color: var(--lightGrey);
  }

  .message-content .answer {
    display: inline-block;
    text-align: left;
    padding: 10px;
    border: 1px solid var(--black);
  }

  .message-content .answers {
    /* display: flex;
    flex-direction: column;
    align-items: flex-end; */
    display: grid;
    grid-template-columns: 49% 49%;
    gap: 2%;
    width: 100%;
    margin: auto;
  }
</style>
