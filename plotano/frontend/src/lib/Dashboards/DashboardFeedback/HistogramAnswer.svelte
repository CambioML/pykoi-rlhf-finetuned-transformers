<script>
  import { max, min, bin } from "d3-array";
  import { format } from "d3-format";
  import { scaleLinear, scaleBand, scaleOrdinal } from "d3-scale";
  import { feedbackSelection, chatLog } from "../../../store.js";
  import { interpolate } from "d3-interpolate";
  import { getQAWordFrequency } from "../../../utils";

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
    top: 15,
    bottom: 10,
    left: 25,
    right: 5,
  };

  const formatter = format(".1f");

  $: qadata =
    $feedbackSelection === "all"
      ? $chatLog
      : $chatLog.filter((d) => d.vote_status === $feedbackSelection);
  $: frequencyData = getQAWordFrequency(qadata);

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  let maxNum = 100;
  let thresholds = Array.from({ length: maxNum }, (_, i) => i);

  // Bin the data.
  $: bins = bin()
    .thresholds(thresholds)
    .value((d) => d.answer)(frequencyData);

  // $: console.log("--------- \nbins", bins);
  // $: console.log("frequencyData changed", frequencyData);

  // Declare the x (horizontal position) scale.
  $: xScale = scaleLinear()
    .domain([3, maxNum])
    .range([margin.left, width - margin.right]);

  // Declare the y (vertical position) scale.
  $: yScale = scaleLinear()
    .domain([0, max(bins, (d) => d.length)])
    .range([height - margin.bottom, margin.top]);

  $: maxValue = max(frequencyData, (d) => d.length);
  $: color = scaleLinear()
    .domain([0, maxValue])
    .range(["white", colorObj[$feedbackSelection]])
    .interpolate(interpolate);

  // $: console.log(bins);
</script>

<div
  class="histogram-container"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    <!-- x-ticks -->
    {#each xScale.ticks() as tick}
      <g
        transform={`translate(${xScale(tick) + (xScale(1) - xScale(0)) / 2} ${
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

    {#each bins as d}
      {@const barWidth = xScale(d.x1) - xScale(d.x0)}
      <g class="histogram-bin">
        <rect
          x={xScale(d.x0) + 1}
          width={barWidth}
          y={yScale(d.length)}
          height={yScale(0) - yScale(d.length)}
          fill={colorObj[$feedbackSelection]}
        />
        <!-- <text
            class="axis-text"
            x={xScale(d.x0) + barWidth / 2}
            y={yScale(d.length) - 5}
            text-anchor="middle">{Math.round(formatter(d.length))}</text
          > -->
      </g>
    {/each}

    <!-- axis labels -->
    <text
      class="chart-title"
      y={margin.top / 2 + 5}
      x={(width + margin.left) / 2}
      text-anchor="middle">Answer Length</text
    >

    <line
      class="axis-line"
      x1={margin.left}
      x2={width - margin.right}
      y1={height - margin.bottom}
      y2={height - margin.bottom}
    />
  </svg>
</div>

<style>
  * {
    transition: all 0.3s;
  }
  .chart-title {
    font-size: var(--smallText);
  }
  rect:hover {
    stroke: var(--black);
  }
  .histogram-container {
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
