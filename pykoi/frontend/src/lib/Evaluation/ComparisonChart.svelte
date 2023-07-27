<script>
  import Linechart from "./LineChart.svelte";
  import EvalLineChart from "./EvalLineChart.svelte";
  import { writable } from "svelte/store";
  import Table from "./Table.svelte";
  import Bar from "./Bar.svelte";
  import { data } from "./data";

  const dataStore = writable([]);
  let options = {
    /* Your options here */
  };

  function generateData(size) {
    return Array.from({ length: size }, (_, i) => ({
      Question: `What is this question asking you ${i + 1}?`,
      Answer: `The answer to the asked questio tatere rea raer aer ea rearerearea reareara eran is simply is ${
        1990 + i * 5
      }`,
      Feedback: (i % 5) - 2,
    }));
  }

  // slider value
  let sliderValue = 20;
  // let data = [];
  // $: {
  //     dataStore.set(generateData(sliderValue));
  //     dataStore.subscribe(value => {
  //         data = value;
  //     });
  // }
</script>

<div class="main-container">
  <div class="instructions">
    <h5 class="underline bold">Comparisons</h5>
    <p>This panes represents the training performance of your model.</p>
    <button>Download Data</button>
  </div>

  <div class="eval-container">
    <div class="left-charts">
      <div class="chart-captions">
        <h4>Model Comparisons</h4>
        <p>
          View the performance of your model over time. GPU stats are available
          to the right.
        </p>
        <div class="buttons">
          <button class="rating-button">Absolute</button>
          <button class="rating-button">Cumulative</button>
        </div>
        <p>Add a sentence here</p>
      </div>
      <div class="eval-main">
        <EvalLineChart />
      </div>
      <div class="eval-table">
        <Table {data} {options} />
      </div>
    </div>
    <div class="right-charts">
      <div class="right-chart-1">
        <Linechart />
      </div>
      <div class="right-chart-2">
        <Bar />
      </div>
      <div class="right-chart-3">
        <Bar />
      </div>
    </div>
  </div>
</div>

<!-- <Linechart /> -->

<style>
  .buttons {
    justify-content: center;
  }

  .rating-button {
    text-transform: uppercase;
    margin: 0;
    /* transition: all 0.1s; */
  }
  button {
    margin: 0;
  }

  .rating-button:hover {
    color: var(--white);
    background: var(--black);
  }

  /* Remove the space between buttons */
  .rating-button + .rating-button {
    margin: 0;
  }
  .chart-captions {
    /* border: 1px solid black; */
    margin: auto;
    width: 80%;
    text-align: left;
    height: 100%;
  }
  .chart-captions h4 {
    padding: 0;
    margin: 0;
  }
  .chart-captions p {
    font-size: var(--smallText);
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

  .underline {
    border-bottom: var(--line);
  }

  .bold {
    font-weight: bold;
    font-size: var(--smallText);
    margin: 0;
    padding: 0;
  }

  .instructions {
    border-right: 1px solid #eee;
  }
  .main-container {
    display: grid;
    grid-template-columns: 20% 80%;
  }
  .eval-container {
    display: grid;
    height: 100vh;
    grid-template-rows: 100%;
    grid-template-columns: 70% 30%;
    padding: 1rem;
  }
  .left-charts {
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 25% 50% 25%;
  }

  .eval-table {
    margin: auto;
    width: 100%;
  }

  .right-charts {
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 33.3% 33.3% 33.3%;
    height: 100vh;
  }
</style>
