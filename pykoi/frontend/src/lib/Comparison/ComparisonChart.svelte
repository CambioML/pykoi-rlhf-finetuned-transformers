<script>
  import BumpChart from "./BumpChart.svelte";
  import { scaleOrdinal } from "d3-scale";
  import { onMount } from "svelte";
  import HorizontalBar from "./HorizontalBar.svelte";
  import Table from "./Table.svelte";
  import { comparisonData } from "./store";

  import Heatmap from "./Heatmap.svelte";

  $: models = Array.from(new Set($comparisonData.map((d) => d.model)));
  $: colorScale = scaleOrdinal()
    .domain($comparisonData.map((d) => d.model))
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F", "red"]);

  async function retrieveDBData() {
    const response = await fetch("/chat/comparator/db/retrieve");
    const data = await response.json();
    console.log("uploooo", data);
    const dbRows = data["data"];
    const formattedRows = dbRows.map((d) => ({
      model: d.model,
      answer: d.answer,
      qid: parseInt(d.qid),
      rank: parseInt(d.rank),
    }));
    $comparisonData = [...formattedRows];
  }

  onMount(() => {
    retrieveDBData();
  });

  function highLight(i) {
    document
      .querySelectorAll(".model-path, .model-path-outer, .model-circle")
      .forEach((el) => {
        el.style.opacity = 0.12;
      });
    document
      .querySelectorAll(
        `.model-path[data-model="${models[i]}"], .model-circle[data-model="${models[i]}"]`
      )
      .forEach((el) => {
        el.style.opacity = 1;
      });
  }

  function unHighlight() {
    document
      .querySelectorAll(".model-path, .model-path-outer, .model-circle")
      .forEach((el) => {
        el.style.opacity = 1;
      });
  }
</script>

<div class="main-container">
  {#if $comparisonData.length > 0}
    <div class="eval-container">
      <div class="left-charts">
        <div class="chart-captions">
          <h4>Model Comparisons</h4>

          {#each models as model, i}
            <button
              data-model={model}
              class="model-path"
              style="color: white; background: {colorScale(model)}"
              on:mouseover={() => highLight(i)}
              on:focus={() => highLight(i)}
              on:mouseout={unHighlight}
              on:blur={unHighlight}>{model}</button
            >
          {/each}
        </div>
        <div class="eval-main">
          <BumpChart />
        </div>
        <div class="eval-table">
          <Table />
        </div>
      </div>
      <div class="right-charts">
        <div class="right-chart-1" />
        <div class="right-chart-2">
          <HorizontalBar />
        </div>
        <div class="right-chart-3">
          <Heatmap />
        </div>
      </div>
    </div>
  {:else}
    <div class="holder">
      <h5>
        To view the comparison dashboard, you must first rank some comparisons!
      </h5>
    </div>
  {/if}
</div>

<style>
  .holder {
    height: 100vh;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  button {
    margin: 0;
  }

  .chart-captions {
    margin: auto;
    width: 100%;
    text-align: left;
    height: 100%;
  }
  .chart-captions h4 {
    padding: 0;
    margin: 0;
  }

  .main-container {
    margin: auto;
  }
  .eval-container {
    display: grid;
    height: 100vh;
    grid-template-rows: 100%;
    grid-template-columns: 65% 35%;
    padding: 1rem;
  }
  .left-charts {
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 15% 60% 25%;
  }

  .eval-table {
    margin: auto;
    width: 100%;
  }

  .right-charts {
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 10% 33% 33%;
    gap: 1%;
  }
</style>
