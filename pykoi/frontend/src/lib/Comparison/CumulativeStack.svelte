<script>
  import { extent, max } from "d3-array";
  import { scaleLinear, scalePoint, scaleOrdinal, scaleBand } from "d3-scale";
  import {
    area,
    line,
    curveStepBefore,
    stackOrderReverse,
    stackOffsetSilhouette,
    stack,
  } from "d3-shape";
  import { data } from "./data";
  import { comparisonData } from "./store";

  function calculateCumulativeRanks(data, maxRank = 4) {
    const cumulativeRanks = [];
    const modelRanks = {};

    $comparisonData.forEach((item) => {
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

  let outerHeight;
  let outerWidth;

  let margin = {
    top: 30,
    bottom: 0,
    left: 45,
    right: 15,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  const sumData = calculateCumulativeRanks($comparisonData);
  console.log("sumData", sumData);

  const models = Object.keys(sumData[0]).filter((d) => d != "qid");

  const stackd = stack().keys(models).order(stackOrderReverse);

  const stackedSeries = stackd(sumData);

  let maxValue = max(stackedSeries, (d) => max(d, (d) => d[1]));

  $: xScale = scaleBand()
    .domain(sumData.map((d) => d.qid))
    .range([margin.left, width - margin.right]);

  $: yScale = scaleLinear()
    .domain([0, maxValue])
    .range([height - margin.bottom, margin.top]);

  $: colorScale = scaleOrdinal()
    .domain($comparisonData.map((d) => d.model))
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F"]);

  $: areaGenerator = area()
    .x((d, i) => xScale(d.data.qid) + xScale.bandwidth() / 2)
    .y0((d) => yScale(d[0]))
    .y1((d) => yScale(d[1]))
    .curve(curveStepBefore);

  $: pathLine = line()
    .x((d, i) => xScale(d.data.qid) + xScale.bandwidth() / 2)
    .y((d) => yScale(d[1]))
    .curve(curveStepBefore);
</script>

<div
  id="chart-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
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
    <!-- y-ticks -->
    {#each yScale.ticks() as tick}
      <g transform={`translate(${margin.left} ${yScale(tick) + 0})`}>
        <line
          class="axis-tick"
          x1="0"
          x2={width - margin.left - margin.right}
          y1={0}
          y2={0}
          stroke="black"
          stroke-dasharray="4"
        />
        <text
          class="axis-text"
          x="-5"
          y="0"
          text-anchor="end"
          dominant-baseline="middle">{""}</text
        >
      </g>
    {/each}

    {#each stackedSeries as d (d.key)}
      <!-- <path  
            class="area" 
            fill={colorScale(d.key)} 
            d={areaGenerator(d)} 
            stroke="white" 
                  opacity=.99
            stroke-width={2}>
          </path> -->
      <path
        class="area"
        fill="none"
        d={pathLine(d)}
        stroke={"white"}
        stroke-width={10}
      />
      <path
        class="area"
        fill="none"
        d={pathLine(d)}
        stroke={colorScale(d.key)}
        stroke-width={6}
      />
    {/each}

    <line
      class="axis-line"
      x1={margin.left}
      x2={width - margin.right}
      y1={height - margin.bottom}
      y2={height - margin.bottom}
    />

    <text
      class="chart-title"
      y={margin.top / 2}
      x={margin.left}
      text-anchor="start">Per-Question Ranks</text
    >
    <text
      class="chart-subtitle"
      y={margin.top / 2 + 15}
      x={margin.left}
      opacity=".6"
      text-anchor="start">Answer ranks over time, by model.</text
    >
  </svg>
</div>

<style>
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
</style>
