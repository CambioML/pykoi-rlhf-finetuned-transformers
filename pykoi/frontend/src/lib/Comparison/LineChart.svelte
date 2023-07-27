<script>
  import { extent } from "d3-array";
  import { scaleLinear } from "d3-scale";
  import { line, curveBasis, area } from "d3-shape";
  import { data } from "./data";

  let outerHeight;
  let outerWidth;

  let margin = {
    top: 10,
    bottom: 15,
    left: 35,
    right: 0,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  // scales
  $: xScale = scaleLinear()
    .domain(extent(data.map((d) => d.epoch)))
    .range([margin.left, width - margin.right]);

  $: yScale = scaleLinear()
    .domain(extent(data.map((d) => d.error)))
    .range([height - margin.bottom, margin.top]);

  // the path generator
  $: pathLine = line()
    .x((d) => xScale(d.epoch))
    .y((d) => yScale(d.error));
  // .curve(curveBasis);

  $: pathArea = area()
    .x((d) => xScale(d.epoch))
    .y0(height - margin.bottom)
    .y1((d) => yScale(d.error));
  // .curve(curveBasis);
</script>

<div
  class="chart-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    <defs>
      <linearGradient id="area-gradien2t" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" style="stop-color: var(--green); stop-opacity: 1" />
        <stop offset="100%" style="stop-color: white; stop-opacity: .2" />
      </linearGradient>
    </defs>

    <path class="area-path" d={pathArea(data)} fill="url(#area-gradien2t)" />
    <path class="outer-path" d={pathLine(data)} />
    <path class="inner-path" d={pathLine(data)} />

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
    />
    <!-- x-ticks -->
    {#each xScale.ticks() as tick}
      <g transform={`translate(${xScale(tick) + 0} ${height - margin.bottom})`}>
        <line
          class="axis-tick"
          x1="0"
          x2="0"
          y1={0}
          y2={-height + margin.bottom + margin.top}
          stroke="black"
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
          stroke-dasharray="4"
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

    <!-- axis labels -->
    <text
      class="chart-title"
      y={margin.top / 2}
      x={(width + margin.left) / 2}
      text-anchor="middle"
    />
    <text
      class="axis-label"
      y={height + margin.bottom + 2}
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
  </svg>
</div>

<style>
  .chart-holder {
    height: 100%;
    width: 100%;
  }
  .axis-line {
    stroke-width: 3;
    stroke: black;
    fill: none;
  }
  .axis-tick {
    stroke-width: 1;
    stroke: black;
    fill: none;
    opacity: 0.05;
  }
  .axis-text {
    font-family: Arial;
    font-size: 12px;
  }

  .inner-path {
    stroke: var(--green);
    stroke-width: 4;
    fill: none;
    stroke-linecap: round;
  }
  .outer-path {
    stroke: white;
    stroke-width: 5;
    opacity: 1;
    fill: none;
    stroke-linecap: round;
  }

  .area-path {
    opacity: 0.76; /* adjust for your preferred opacity */
  }
</style>
