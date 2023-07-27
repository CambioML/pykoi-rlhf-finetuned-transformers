<script>
  import { max, min } from "d3-array";
  import { format } from "d3-format";
  import { scaleLinear, scaleOrdinal, scaleBand } from "d3-scale";
  import { questionDistribution } from "./store.js";
  import { stack, stackOrderNone, stackOffsetNone } from "d3-shape";
  import { interpolate } from "d3-interpolate";

  let outerHeight = 300;
  let outerWidth = 300;

  let margin = {
    top: 10,
    bottom: 15,
    left: 35,
    right: 0,
  };

  const formatter = format(".0%");

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  // const color = scaleOrdinal().range(["#2074d5", "#fde24f"]);

  $: xScale = scaleBand()
    .rangeRound([margin.left, width - margin.right])
    .padding(0.05)
    .domain($questionDistribution.map((d) => d.question));

  $: yScale = scaleLinear()
    .rangeRound([height - margin.bottom, margin.top])
    .domain([0, max($questionDistribution, (d) => d.count)]);

  const dStack = stack()
    .keys(["count"])
    .order(stackOrderNone)
    .offset(stackOffsetNone);

  $: series = dStack($questionDistribution);

  $: minValue = min($questionDistribution, (d) => d.count);
  $: maxValue = max($questionDistribution, (d) => d.count);
  $: color = scaleLinear()
    .domain([minValue, maxValue])
    .range(["#FF5470", "#FF5470"])
    .interpolate(interpolate);
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
        <!-- <text class="axis-text" y="15" text-anchor="middle">{tick}</text> -->
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
        <text
          class="axis-text"
          x="-5"
          y="0"
          text-anchor="end"
          dominant-baseline="middle">{tick}</text
        >
      </g>
    {/each}
    <!-- stacked rects -->
    {#each series as serie}
      <g class="series">
        {#each serie as d}
          <rect
            x={xScale(d.data.question)}
            y={yScale(d[1])}
            height={yScale(d[0]) - yScale(d[1])}
            fill={color(d[1])}
            fill-opacity="0.95"
            width={xScale.bandwidth()}
          />
          <text
            x={xScale(d.data.question) + xScale.bandwidth() / 2}
            y={yScale(d[1]) - 5}
            text-anchor="middle"
          />
        {/each}
      </g>
    {/each}

    <!-- axis labels -->
    <text
      class="chart-title"
      y={margin.top / 2}
      x={(width + margin.left) / 2}
      text-anchor="middle"
    />
    <text
      class="chart-title"
      y={margin.top / 2 + 15}
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
  .axis-label,
  .chart-title {
    font-size: 12px;
  }
  rect:hover {
    stroke: black;
  }
  #stackedrect-holder {
    height: 100%;
    width: 100%;
  }
  .axis-line {
    stroke-width: 1;
    stroke: black;
    fill: none;
  }
  .axis-tick {
    stroke-width: 1;
    fill: none;
    opacity: 0.05;
    font-size: 9px;
  }
  .axis-text {
    font-family: Arial;
    font-size: 12px;
  }
</style>
