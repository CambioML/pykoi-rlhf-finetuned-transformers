<script>
  let index = 0;

  const data = [
    {
      question: "Which is a species of Fish? Trout or Bear?",
      answer: "Answer 1",
    },
    { question: "Question 2", answer: "Answer 2" },
    { question: "Question 3", answer: "Answer 3" },
    // Add more as needed
  ];

  const next = () => {
    if (index < data.length - 1) {
      index += 1;
    }
  };

  const prev = () => {
    if (index > 0) {
      index -= 1;
    }
  };
</script>

<!-- <div class="container">
  <h1>{data[index].question}</h1>
  <h2>{data[index].answer}</h2>

  <div class="buttons">
    <button on:click={prev}>PREV</button>
    <button on:click={next}>NEXT</button>
  </div>
</div> -->

<div class="main-annotation-container">
  <div class="annotation-container">
    <div class="instructions">
      <h5 class="underline bold">Annotation Instructions</h5>
      <p>
        Given the Question an Answer, rate the question from 1 - 5. Optionally,
        add a comment as to why the question was good, or where it needs
        improvement.
      </p>
      <p>Rate the response:</p>
      <div class="buttons">
        <button class="rating-button">1</button>
        <button class="rating-button">2</button>
        <button class="rating-button">3</button>
        <button class="rating-button">4</button>
        <button class="rating-button">5</button>
      </div>
      <p>
        If the response is incorrect, write a response: <span
          style="color: var(--grey);">(optional)</span
        >
      </p>
      <textarea />
      <button>Download Data</button>
    </div>
    <div class="ranked-chat">
      <section class="chatbox">
        <div class="chat-message-center">
          <div class="avatar">
            <!-- <img src={logo} alt="SvelteKit" /> -->
          </div>
          <div class="message-content">
            <div class="question">
              <h5 class="bold">Question:</h5>
              <p>{data[index].question}</p>
            </div>
            <div class="answers">
              <div class="answer">
                <h5 class="bold underline">Response 1:</h5>
                <p>{data[index].answer}</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
  <div class="chat-input-holder">
    <button on:click={prev}>PREV</button>
    <button on:click={next}>NEXT</button>
  </div>
</div>

<style>
  .main-annotation-container {
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 90% 10%;
    height: 100vh;
    width: 100%;
    margin: auto;
  }
  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .buttons {
    display: flex;
    background: var(--grey);
    /* justify-content: center; */
  }

  .rating-button {
    padding: 10px;
    border: none;
  }

  /* Remove the space between buttons */
  .rating-button + .rating-button {
    margin-left: -1px;
  }
  .ranked-chat {
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 100%;
    border: var(--line);
    margin: 12px;
  }

  .message {
    font-size: var(--smallText);
    padding-left: 40px;
    padding-right: 40px;
    margin: 0 auto;
  }

  .chat-input-holder {
    /* display: flex; */
    /* flex-direction: row; */
    align-items: center;
    /* padding: 24px; */
    width: 100%;
    /* max-width: 640px; */
    margin: auto;
    border-top: var(--line);
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

  .annotation-container {
    display: grid;
    grid-template-columns: 40% 60%;
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

  .chat-message-center {
    display: flex;
    flex-direction: column;
    padding: 12px;
    box-sizing: border-box;
  }

  .message-content {
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
    display: block;
  }

  .message-content .question {
    text-align: left;
    border: 1px solid var(--grey);
    padding: 5px;
    margin-bottom: 10px;
    background-color: var(--lightGrey);
  }

  .message-content .answer {
    display: block;
    text-align: left;
    padding: 10px;
    border: 1px solid var(--black);
  }

  .message-content .answers {
    display: grid;
    grid-template-columns: 100%;
    gap: 2%;
    width: 100%;
    margin: auto;
  }
</style>
