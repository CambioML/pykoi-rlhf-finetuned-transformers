<script>
  import { onMount } from "svelte";
  import { select } from "d3-selection";
  import { rankChatLog } from "../../store";

  let mymessage = "";
  let messageplaceholder = "";
  let chatLoading = false;

  onMount(() => {
    // getDataFromDB();
  });

  async function getDataFromDB() {
    const response = await fetch("/chat/ranking_table/retrieve");
    const data = await response.json();
    const dbRows = data["rows"];
    const formattedRows = dbRows.map((row) => ({
      id: row[0],
      question: row[1],
      up_ranking_answer: row[2],
      low_ranking_answer: row[3],
    }));
    $rankChatLog = [...formattedRows];
  }

  const askModel = async (event) => {
    event.preventDefault(); // Prevents page refresh
    mymessage = messageplaceholder;
    messageplaceholder = "";
    chatLoading = true;
    let currentEntry = {
      question: mymessage,
      up_ranking_answer: "Loading...",
      low_ranking_answer: "Loading...",
    };
    $rankChatLog = [...$rankChatLog, currentEntry];

    const response = await fetch(`/chat/multi_responses/${mymessage}`, {
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
      currentEntry["up_ranking_answer"] = data["answer"][0];
      currentEntry["low_ranking_answer"] = data["answer"][1];
      rankChatLog.update((state) => {
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

  async function insertRanking(rankingUpdate) {
    const response = await fetch("/chat/ranking_table/update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(rankingUpdate),
    });

    if (response.ok) {
      console.log("response", response);
    } else {
      const err = await response.text();
      alert(err);
    }
  }

  function handleClick(event, msg, questionIndex, answerIndex) {
    // insert message[index] as good
    const currQuestion = $rankChatLog[questionIndex];

    const id2answer = {
      1: "up_ranking_answer",
      2: "low_ranking_answer",
    };

    const rankingUpdate = {
      question: currQuestion.question,
      up_ranking_answer: currQuestion[id2answer[answerIndex]],
      low_ranking_answer: currQuestion[id2answer[(answerIndex % 2) + 1]],
    };

    // add to db
    insertRanking(rankingUpdate);

    select(event.currentTarget.parentNode)
      .selectAll("div")
      .style("outline", "2px solid var(--red)")
      .style("border", "1px solid var(--red)");
    select(event.currentTarget)
      .style("outline", "2px solid var(--green)")
      .style("border", "1px solid var(--green)");
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
  <div class="ranked-chat">
    <section class="chatbox">
      <div class="chat-log">
        {#each $rankChatLog as message, index (index)}
          <div
            class="chat-message"
            use:scrollToView={index === $rankChatLog.length - 1}
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
                  <div
                    class="answer"
                    on:click={(event) => handleClick(event, message, index, 1)}
                    on:keydown={(event) =>
                      handleClick(event, message, index, 1)}
                  >
                    <h5 class="bold underline">Response 1:</h5>
                    <p>{message.up_ranking_answer}</p>
                  </div>
                  <div
                    class="answer"
                    on:click={(event) => handleClick(event, message, index, 2)}
                    on:keydown={(event) =>
                      handleClick(event, message, index, 2)}
                  >
                    <h5 class="bold underline">Response 2:</h5>
                    <p>{message.low_ranking_answer}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </section>
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
  </div>
</div>

<style>
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
    padding: 24px;
    width: 100%;
    max-width: 640px;
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
    display: grid;
    grid-template-columns: 49% 49%;
    gap: 2%;
    width: 100%;
    margin: auto;
  }
</style>
