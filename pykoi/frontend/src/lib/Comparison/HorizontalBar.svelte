<script>
  import { max } from "d3-array";
  import { scaleLinear, scaleBand, scaleOrdinal } from "d3-scale";
  import { comparisonData } from "./store";

  $: averageRanks = $comparisonData.reduce((acc, curr) => {
    if (!acc[curr.model]) {
      acc[curr.model] = { sum: curr.rank, count: 1 };
    } else {
      acc[curr.model].sum += curr.rank;
      acc[curr.model].count++;
    }
    return acc;
  }, {});

  $: avgRankData = Object.keys(averageRanks).map((key) => ({
    model: key,
    avgRank: averageRanks[key].sum / averageRanks[key].count,
  }));

  let outerHeight = 300;
  let outerWidth = 500;

  let margin = {
    top: 50,
    bottom: 0,
    left: 100,
    right: 0,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  $: yScale = scaleBand()
    .rangeRound([margin.top, height - margin.bottom])
    .padding(0.05)
    .domain(avgRankData.map((d) => d.model));

  $: xScale = scaleLinear()
    .rangeRound([margin.left, width - margin.right])
    .domain([0, max(avgRankData, (d) => d.avgRank)]);

  $: colorScale = scaleOrdinal()
    .domain(avgRankData.map((d) => d.model))
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F"]);

  $: models = Array.from(new Set($comparisonData.map((d) => d.model)));
</script>

<div
  id="bar-chart-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    <!-- y-ticks -->
    {#each avgRankData.map((d) => d.model) as tick}
      <g
        transform={`translate(${margin.left} ${
          yScale(tick) + yScale.bandwidth() / 2
        })`}
      >
        <text class="axis-text" x="-5" y="0" text-anchor="end">{tick}</text>
      </g>
    {/each}

    <!-- x-ticks -->
    {#each xScale.ticks() as tick}
      {#if tick % 2 == 0}
        <g transform={`translate(${xScale(tick)}, ${height - margin.bottom})`}>
          <text class="axis-text" y="15" text-anchor="middle">{tick}</text>
        </g>
      {/if}
    {/each}

    <!-- bars -->
    {#each avgRankData as d, i}
      <rect
        y={yScale(d.model)}
        x={margin.left}
        width={xScale(d.avgRank) - margin.left}
        height={yScale.bandwidth()}
        fill={colorScale(d.model)}
        class="model-path"
        data-model={models[i]}
      />
      <text
        class="label-text"
        y={yScale(d.model) + yScale.bandwidth() / 2}
        x={xScale(d.avgRank) + 5}
        text-anchor="start"
        dominant-baseline="middle"
      >
        {d.avgRank.toFixed(2)}
      </text>
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
      text-anchor="start">Average Rank</text
    >
    <text
      class="chart-subtitle"
      y={margin.top / 2 + 15}
      x={margin.left}
      opacity=".6"
      text-anchor="start"
    />
  </svg>
</div>

<style>
  #bar-chart-holder {
    height: 100%;
    width: 100%;
  }
  .axis-text {
    font-size: 9px;
  }
  .axis-line {
    stroke-width: 3;
    stroke: black;
    fill: none;
  }
  .label-text {
    font-size: 9px;
  }
</style>
