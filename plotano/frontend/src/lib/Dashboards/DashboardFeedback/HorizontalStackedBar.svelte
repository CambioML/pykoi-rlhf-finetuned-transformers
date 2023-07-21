<script>
  import { max } from "d3-array";
  import { format } from "d3-format";
  import { scaleLinear, scaleOrdinal, scaleBand } from "d3-scale";
  import { stack, stackOrderNone, stackOffsetNone } from "d3-shape";
  import { chatLog, stackedData, feedbackSelection } from "../../../store";
  import { onMount } from "svelte";

  let outerHeight = 300;
  let outerWidth = 300;
  const feedback2Highlight = {
    up: 2,
    down: 1,
    "n/a": 0,
    all: "all",
  };

  $: highlightedBar = feedback2Highlight[$feedbackSelection];

  const lineOffset = 5;

  const margin = {
    top: 10,
    bottom: 10,
    left: 25,
    right: 5,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  const color = scaleOrdinal().range([
    "var(--yellow)",
    "var(--red)",
    "var(--green)",
  ]);

  const formatter = format(".0%");
  $: valueSum = $stackedData.up + $stackedData.down + $stackedData["n/a"];

  $: yScale = scaleBand()
    .rangeRound([margin.top, height - margin.bottom])
    .padding(0)
    .domain(["a"]);

  $: xScale = scaleLinear()
    .rangeRound([margin.left, width - margin.right])
    .domain([0, valueSum]);

  $: dStack = stack()
    .keys(["n/a", "down", "up"])
    .order(stackOrderNone)
    .offset(stackOffsetNone);

  $: {
    $chatLog.forEach((example) => {
      $stackedData[example.vote_status]++;
    });
  }

  $: series = dStack([$stackedData]);

  function horizontalBarClick(i, val) {
    if (highlightedBar === i) {
      highlightedBar = "all";
    } else {
      highlightedBar = i;
    }
    $feedbackSelection = highlightedBar === "all" ? "all" : val;
  }
</script>

<div>
  <p>Feedback Distribution</p>
</div>
<div
  id="stackedrect-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    {#each series as serie, i}
      <g
        class="series"
        on:click={() => horizontalBarClick(i, serie.key)}
        on:keypress={() => horizontalBarClick(i, serie.key)}
        role="button"
        tabindex="0"
      >
        {#each serie as d}
          {@const value = $stackedData[serie.key]}
          <rect
            class="horizontal-bar-rect {highlightedBar === i
              ? 'selected'
              : 'unselected'}"
            y={yScale(d.data.xVal)}
            x={xScale(d[0])}
            stroke="var(--black)"
            width={xScale(d[1]) - xScale(d[0])}
            fill={color(serie.key)}
            opacity={highlightedBar === "all" || highlightedBar === i
              ? "1"
              : "0.25"}
            height={yScale.bandwidth()}
          />
          <line
            class="line"
            x1={xScale(d[0])}
            x2={xScale(d[0])}
            y1={yScale(d.data.xVal) - lineOffset}
            y2={yScale.bandwidth() + yScale(d.data.xVal) + lineOffset}
            stroke="black"
          />
          <text
            class="horizontal-bar-text"
            y={yScale.bandwidth() / 2}
            x={(xScale(d[0]) + xScale(d[1])) / 2}
            opacity={highlightedBar === "all" || highlightedBar === i
              ? "1"
              : "0.25"}
            text-anchor="middle"
          >
            {serie.key}
            <tspan
              x={(xScale(d[0]) + xScale(d[1])) / 2}
              dy="16"
              text-anchor="middle">{formatter(value / valueSum)}</tspan
            >
          </text>
        {/each}
        <line
          class="line"
          x1={width - margin.right}
          x2={width - margin.right}
          y1={yScale("A") - lineOffset}
          y2={yScale.bandwidth() + yScale("A") + lineOffset}
          stroke="black"
        />
      </g>
    {/each}
  </svg>
</div>

<style>
  .series:focus {
    outline: none;
  }
  .horizontal-bar-rect.selected {
    stroke: var(--black);
    stroke-width: 3;
  }
  #stackedrect-holder {
    height: 100%;
    width: 100%;
  }
  .line {
    stroke-width: 3;
    stroke: var(--black);
    fill: none;
  }
  .horizontal-bar-rect {
    transition: opacity 0.3s;
  }
  .horizontal-bar-rect:hover {
    stroke: var(--black);
    stroke-width: 3;
  }
  .horizontal-bar-text {
    transition: opacity 0.3s;
    stroke: var(--white);
    stroke-width: 4px;
    fill: var(--black);
    stroke-linejoin: round;
    paint-order: stroke fill;
    pointer-events: none;
    font-size: var(--smallText);
  }
</style>
