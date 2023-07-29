<script>
  import { extent, max } from "d3-array";
  import { scaleLinear, scalePoint, scaleOrdinal } from "d3-scale";
  import { line, curveStepBefore } from "d3-shape";
  import { data } from "./data";
  import { comparisonData } from "./store";

  function calculateCumulativeRanks(data, maxRank = 4) {
    const cumulativeRanks = [];
    const modelRanks = {};

    data.forEach((item) => {
      const invertedRank = maxRank + 1 - item.rank; // Invert the rank

      // If the model hasn't been seen before, initialize it in the modelRanks object
      if (!modelRanks[item.model]) {
        modelRanks[item.model] = invertedRank;
      } else {
        modelRanks[item.model] += invertedRank;
      }

      // If the qid hasn't been seen before, add a new object to the cumulativeRanks array
      if (!cumulativeRanks[item.qid - 1]) {
        cumulativeRanks[item.qid - 1] = { ...modelRanks, qid: item.qid };
      } else {
        cumulativeRanks[item.qid - 1][item.model] = modelRanks[item.model];
      }
    });

    return cumulativeRanks;
  }

  const sumData = calculateCumulativeRanks($comparisonData);
  console.log(sumData);

  let maxVal = max(Object.values(sumData[sumData.length - 1]));

  const firstData = $comparisonData
    .filter((d) => d.qid === 1)
    .map((d) => ({ model: d.model, rank: d.rank }));

  let models = Array.from(new Set($comparisonData.map((d) => d.model)));

  let outerHeight;
  let outerWidth;

  let margin = {
    top: 50,
    bottom: 0,
    left: 65,
    right: 15,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  // scales
  $: xScale = scalePoint()
    .domain($comparisonData.map((d) => d.qid))
    .padding(0.3)
    .range([margin.left, width - margin.right]);

  $: yScale = scaleLinear()
    .domain([0, maxVal])
    .range([margin.top, height - margin.bottom]);

  $: colorScale = scaleOrdinal()
    .domain($comparisonData.map((d) => d.model))
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F"]);

  // the path generator
  $: pathLine = line()
    .x((d) => xScale(d.qid))
    .y((d) => yScale(d.rank))
    .curve(curveStepBefore);

  $: modelData = models.map((model) =>
    sumData.map((d) => ({ model: model, qid: d.qid, rank: maxVal - d[model] }))
  );
  $: console.log(modelData);
</script>

<div
  id="chart-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    <line
      class="axis-line"
      x1={margin.left}
      x2={width - margin.right}
      y1={height - margin.bottom}
      y2={height - margin.bottom}
    />

    <!-- x-ticks -->
    {#each xScale.domain() as tick}
      {#if Number.isInteger(tick)}
        <g
          transform={`translate(${xScale(tick) + 0} ${height - margin.bottom})`}
        >
          <line
            class="axis-tick"
            x1="0"
            x2="0"
            y1={0}
            y2={-height + margin.bottom + margin.top}
            stroke="black"
            stroke-dasharray="4"
          />
          <text class="axis-text" y="15" text-anchor="middle"
            >{`Q.${tick}`}</text
          >
        </g>
      {/if}
    {/each}

    <!-- axis labels -->
    <text
      class="chart-title"
      y={margin.top / 2}
      x={margin.left}
      text-anchor="start">Cumulative Rank</text
    >
    <text
      class="chart-subtitle"
      y={margin.top / 2 + 15}
      x={margin.left}
      opacity=".6"
      text-anchor="start"
    />
    <text
      class="axis-label"
      y={height + margin.bottom + 10}
      x={(width + margin.left) / 2}
      text-anchor="middle"
    />
    <text
      class="axis-label"
      y={margin.left / 2}
      x={-(height / 2)}
      text-anchor="middle"
      transform="rotate(-90)"
    />

    <!-- model paths -->
    {#each modelData as d, i}
      <path
        class="model-path-outer"
        d={pathLine(d)}
        stroke="white"
        data-model={models[i]}
      />
      <path
        class="model-path"
        data-model={models[i]}
        d={pathLine(d)}
        stroke={colorScale(models[i])}
      />
    {/each}
  </svg>
</div>

<style>
  @import url("https://fonts.googleapis.com/css?family=Work+Sans:400|Lato:400|Inconsolata:400");

  * {
    font-family: "Lato";
  }
  #chart-holder {
    height: 100%;
    width: 100%;
  }
  .axis-line {
    stroke-width: 3;
    stroke: black;
    fill: none;
  }
  .axis-tick {
    stroke-width: 2;
    stroke: black;
    fill: none;
    opacity: 0.04;
  }
  .axis-text {
    font-family: Arial;
    font-size: 12px;
  }

  .bump-text {
    font-size: 12px;
  }

  .model-path {
    fill: none;
    stroke-width: 5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .model-path-outer {
    fill: none;
    stroke-width: 5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .chart-subtitle {
    font-size: 12px;
  }
</style>
