<script>
  export let question = "What is gradient descent?";
  export let answer =
    "Gradient descent is an algorithm used to minimize a loss function by taking small steps in the direction of steepest descent.";
  export let feedback = "Good: 👍";

  const emojiObj = {
    up: "Good 👍",
    down: "Bad 👎",
    "n/a": "No Rating",
    all: "All",
  };

  const colorObj = {
    up: "#00ebc7",
    down: "#FF5470",
    "n/a": "#fde24f",
    all: "#bfbfbf",
  };
</script>

<div class="qa-card">
  <div class="question">Q: {question}</div>
  <div class="answer">A: {answer}</div>
  <div class="feedback {feedback}">Rating: {emojiObj[feedback]}</div>
</div>

<style>
  .qa-card {
    border: 2px solid var(--black);
    display: flex;
    flex-direction: column;
    font-size: var(--smallText);
    padding: 4px;
    margin: 6px;
  }

  .qa-card:hover {
    border: 3px solid var(--black);
  }

  .question {
    background-color: var(--white);
    margin: 0;
    padding: 8px;
    border-bottom: 1px solid var(--black);
  }

  .answer {
    background-color: var(--lightGrey);
    margin: 0;
    padding: 8px;
    /* border-bottom: 1px solid var(--black); */
  }

  .feedback {
    margin: 0;
    padding: 8px;
  }

  .up {
    background-color: var(--green);
  }

  .down {
    background-color: var(--red);
  }

  .na {
    background-color: var(--yellow);
  }
</style>
