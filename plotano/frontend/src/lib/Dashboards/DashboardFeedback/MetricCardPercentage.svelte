<script>
  import { tweened } from "svelte/motion";
  import { format } from "d3-format";
  import { chatLog, feedbackSelection } from "../../../store";

  let outerHeight;
  let outerWidth;

  let tweenedNum = tweened(0);

  const formatter = format(".1%");

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
    all: "var(--white)",
  };

  function countFeedback(dataArray, feedback) {
    const n = dataArray.length;
    if (feedback === "all") {
      return 1;
    }
    const goodVotes = dataArray.filter((item) => item.vote_status === feedback);
    return goodVotes.length / n;
  }
  $: feedbackCount = countFeedback($chatLog, $feedbackSelection);
  $: tweenedNum.set(feedbackCount);
</script>

<div class="card-container" style="background: {colorObj[$feedbackSelection]}">
  <div>
    <p class="card-text">Questions</p>
    <select
      id="feedback-dropdown"
      name="feedback-dropdown"
      style="background: {colorObj[$feedbackSelection]}"
      bind:value={$feedbackSelection}
    >
      <option value="all">All</option>
      <option value="up">Good</option>
      <option value="down">Bad</option>
      <option value="n/a">N/A</option>
    </select>
  </div>
  <div bind:clientWidth={outerWidth} bind:clientHeight={outerHeight}>
    <svg width={outerWidth} height={outerHeight}>
      <text
        dominant-baseline="middle"
        x={outerWidth / 2}
        y={outerHeight / 2}
        text-anchor="middle"
      >
        {formatter($tweenedNum)}</text
      >
    </svg>
  </div>
  <p class="small">percentage of responses</p>
</div>

<style>
  .card-container {
    border: 2px solid var(--black);
    width: 100%;
    height: 100%;
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 35% 50% 10%;
    padding: 5px;
    box-sizing: border-box;
  }
  .card-text {
    text-align: left;
    padding: 0;
    margin: 0;
    padding-left: 2px;
    font-weight: bold;
    font-size: 0.8em;
  }
  #feedback-dropdown {
    width: 98%;
    border: none;
    border-bottom: 1px solid var(--black);
    border-top: 1px solid var(--black);
    font-size: var(--smallText);
    margin-bottom: 1px;
  }
  svg {
    width: 100%;
    height: 100%;
    border-bottom: 1px solid var(--black);
  }

  .small {
    font-size: 0.6em;
  }
  text {
    font-size: 3rem;
    transition: opacity 0.3s;
    stroke: white;
    stroke-width: 6px;
    fill: var(--black);
    stroke-linejoin: round;
    paint-order: stroke fill;
    pointer-events: none;
  }
</style>
