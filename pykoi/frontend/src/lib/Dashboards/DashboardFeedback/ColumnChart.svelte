<script>
  import { max } from "d3-array";
  import { format } from "d3-format";
  import { scaleLinear, scaleBand } from "d3-scale";
  import { questionDistribution, feedbackSelection } from "../../../store.js";

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

  let outerHeight = 300;
  let outerWidth = 300;

  let margin = {
    top: 35,
    bottom: 10,
    left: 25,
    right: 5,
  };

  const formatter = format(".1f");

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  $: xScale = scaleBand()
    .rangeRound([margin.left, width - margin.right])
    .padding(0.05)
    .domain($questionDistribution.map((d) => d.question));

  $: yScale = scaleLinear()
    .rangeRound([height - margin.bottom, margin.top])
    .domain([0, max($questionDistribution, (d) => d.count)]);

  $: console.log($questionDistribution);
</script>

<div
  id="stackedrect-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    <!-- x-ticks -->
    {#each $questionDistribution.map((d) => d.question) as tick}
      <g
        transform={`translate(${xScale(tick) + xScale.bandwidth() / 2} ${
          height - margin.bottom
        })`}
      >
        <line
          class="axis-tick"
          x1="0"
          x2="0"
          y1={0}
          y2={-height + margin.bottom + margin.top}
          stroke="var(--squidink)"
          stroke-dasharray="4"
        />
        <text class="axis-text" y="15" text-anchor="middle">{tick}</text>
      </g>
    {/each}

    <!-- y-ticks -->
    {#each yScale.ticks() as tick}
      <g transform={`translate(${margin.left} ${yScale(tick) + 0})`}>
        <line
          class="axis-tick"
          x1={0}
          x2={width - margin.right - margin.left}
          y1="0"
          y2="0"
          stroke="black"
        />
      </g>
    {/each}
    <!-- stacked rects -->
    {#each $questionDistribution as d}
      <g class="series">
        <rect
          x={xScale(d.question)}
          y={yScale(d.count)}
          height={height - yScale(d.count) - margin.bottom}
          fill={colorObj[$feedbackSelection]}
          fill-opacity="0.95"
          width={xScale.bandwidth()}
        />
        <text
          class="axis-text"
          x={xScale(d.question) + xScale.bandwidth() / 2}
          y={yScale(d.count) - 5}
          text-anchor="middle">{Math.round(formatter(d.count))}</text
        >
      </g>
    {/each}

    <!-- axis labels -->
    <text
      class="chart-title"
      y={margin.top / 2 + 1}
      x={(width + margin.left) / 2}
      text-anchor="middle">Question Type: {emojiObj[$feedbackSelection]}</text
    >

    <line
      class="axis-line"
      x1={margin.left}
      x2={width - margin.right}
      y1={height - margin.bottom}
      y2={height - margin.bottom}
    />
    <line
      class="axis-line"
      x1={margin.left}
      x2={margin.left}
      y1={margin.top}
      y2={height - margin.bottom}
      opacity="0"
    />
  </svg>
</div>

<style>
  .chart-title {
    font-size: var(--smallText);
  }
  rect:hover {
    stroke: var(--black);
  }
  #stackedrect-holder {
    height: 100%;
    width: 100%;
  }
  .axis-line {
    stroke-width: 3;
    stroke: var(--black);
    fill: none;
  }
  .axis-tick {
    stroke-width: 1;
    fill: none;
    opacity: 0;
    font-size: 9px;
  }
  .axis-text {
    font-size: calc(var(--smallText) * 0.9);
  }
</style>
