<script>
  import { onMount } from "svelte";
  import { compareChatLog } from "../../store";
  import Sortable from "sortablejs";
  import { select } from "d3-selection";

  export let numModels = 1;
  export let models = [0];

  let mymessage = "";
  let messageplaceholder = "";
  let chatLoading = false;
  $: gridTemplate = "1fr ".repeat(numModels).trim();
  let answerOrder = [];

  onMount(async () => {
    // Give the DOM some time to render
    await new Promise((r) => setTimeout(r, 200));
    if (answersContainer) {
      const sortable = new Sortable(answersContainer, {
        animation: 150,
        onUpdate: function (/**Event*/ evt) {
          answerOrder = sortable.toArray();
        },
      });
      answerOrder = sortable.toArray();
    }
    // retrieveDBData();
  });

  async function retrieveDBData() {
    const response = await fetch("/chat/comparator/db/retrieve");
    const data = await response.json();
    // const dbRows = data["data"];
    // const formattedRows = dbRows.map((row) => ({
    //   id: row[0],
    //   question: row[1],
    //   up_ranking_answer: row[2],
    //   low_ranking_answer: row[3],
    // }));
    // $compareChatLog = [...dbRows];
  }

  const askModel = async (event) => {
    event.preventDefault();
    mymessage = messageplaceholder;
    messageplaceholder = "";
    chatLoading = true;
    let currentEntry = {
      question: mymessage,
    };
    for (let model of models) {
      currentEntry[model] = "Loading...";
    }
    $compareChatLog = [...$compareChatLog, currentEntry];

    const response = await fetch(`/chat/comparator/${mymessage}`, {
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
      models = Object.keys(data["answer"]);
      numModels = models.length;
      for (let model of models) {
        currentEntry[model] = data["answer"][model];
      }
      currentEntry["qid"] = data["qid"];

      compareChatLog.update((state) => {
        state[state.length - 1] = currentEntry;
        return state;
      });
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

  let answersContainer;
  let sortable;

  // Watch for changes to answersContainer and re-initialize Sortable when it changes
  $: {
    if (answersContainer) {
      sortable = new Sortable(answersContainer, {
        animation: 150,
        dataIdAttr: "id",
        onUpdate: function (evt) {
          answerOrder = sortable.toArray();
        },
      });
      answerOrder = sortable.toArray();
    }
  }

  async function updateComparisonDB(payload) {
    console.log("run update", payload);
    const rankingUpdate = { data: payload };
    const response = await fetch("/chat/comparator/db/update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(rankingUpdate),
    });

    if (response.ok) {
      console.log("ok", response);
    } else {
      const err = await response.text();
      alert(err);
    }
  }

  function handleDragEnd(event, questionIndex) {
    const qid = questionIndex;
    const parent = select(event.currentTarget.parentNode);
    // update ranks to reflect re-order
    const answerOrder = parent
      .selectAll("div.answer")
      .nodes()
      .map((node) => node.id);
    answerOrder.forEach((id, index) => {
      select(`#${id}`)
        .select(".answer-rank")
        .property("value", index + 1);
    });
    let answerOrders = [];
    for (let [index, answer] of answerOrder.entries()) {
      const modelIndex = parseInt(answer.split("-")[1]);
      const modelName = models[modelIndex];
      const rank2model = { rank: index, model: modelName };
      answerOrders.push(rank2model);
    }

    let curr = $compareChatLog.filter((d) => d.qid === qid);
    let payload = [];
    for (let modelEntry of answerOrders) {
      const entry = {
        model: modelEntry.model,
        qid: parseInt(qid),
        rank: parseInt(modelEntry.rank) + 1,
        answer: curr[0][modelEntry.model],
      };
      payload.push(entry);
    }
    updateComparisonDB(payload);
  }

  function handleSelectChange(event, questionIndex, modelIndex) {
    const qid = questionIndex;
    const newRank = parseInt(event.target.value);
    const selModel = models[modelIndex];
    const updatedValues = {
      qid: qid,
      rank: parseInt(newRank),
      model: selModel,
    };
    let payload = [];
    let curr = $compareChatLog.filter((d) => d.qid === updatedValues.qid);
    let currAnswer = curr[0][updatedValues.model];
    const entry = {
      model: updatedValues.model,
      qid: parseInt(updatedValues.qid),
      rank: parseInt(updatedValues.rank),
      answer: currAnswer,
    };
    payload.push(entry);
    updateComparisonDB(payload);
  }
</script>

<div class="ranked-feedback-container">
  <div class="instructions">
    <h5 class="underline bold">Q & A Comparison Instructions</h5>
    <br />
    <p>
      Ask a question and rank the answers across the models. Drag each answer to
      rank it, in ascending order, from left-to-right. Optionally, select the
      rank for each via the corresponding dropdown.
    </p>
    <br />
    <button>Download Data</button>
  </div>
  <div class="ranked-chat">
    <section class="chatbox">
      <div class="chat-log">
        {#each $compareChatLog as message, index (index)}
          <div
            class="chat-message"
            use:scrollToView={index === $compareChatLog.length - 1}
          >
            <div class="chat-message-center">
              <div class="message-content">
                <div class="question">
                  <h5 class="bold">Question: {message.qid}</h5>
                  <p>{message.question}</p>
                </div>
                <div
                  class="answers"
                  bind:this={answersContainer}
                  style="display: grid; grid-template-columns: {gridTemplate}; gap: .25%; width: 100%; margin: auto;"
                >
                  {#each Array(numModels).fill() as _, i (i)}
                    <div
                      class="answer"
                      id={`answer-${i}-${message.qid}`}
                      on:dragend={(event) => handleDragEnd(event, message.qid)}
                    >
                      <h5 class="bold underline">{models[i]}:</h5>
                      <p>
                        {message[models[i]]}
                      </p>
                      <div>
                        Rank:
                        <select
                          class="answer-rank"
                          on:change={(event) =>
                            handleSelectChange(event, message.qid, i)}
                        >
                          {#each Array(numModels).fill() as n, i}
                            <option>{i + 1}</option>
                          {/each}
                        </select>
                      </div>
                    </div>
                  {/each}
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

  .instructions {
    text-align: center;
    padding: 5%;
    border-right: var(--line);
  }

  .instructions h5 {
    text-align: left;
  }

  .instructions p {
    font-size: var(--smallText);
    text-align: left;
    margin: 0;
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

  option {
    font-weight: bold;
    font-size: 120%;
  }

  p {
    margin: 0;
  }
</style>
