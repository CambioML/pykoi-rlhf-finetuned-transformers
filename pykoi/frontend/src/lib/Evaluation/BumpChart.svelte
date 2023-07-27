<script>
  import { extent, max } from "d3-array";
  import { scaleLinear, scalePoint, scaleOrdinal } from "d3-scale";
  import { line } from "d3-shape";
  import { data } from "./data";

  const firstData = data
    .filter((d) => d.QID === 1)
    .map((d) => ({ model: d.model, rank: d.rank }));

  let models = Array.from(new Set(data.map((d) => d.model)));

  let outerHeight;
  let outerWidth;

  let margin = {
    top: 15,
    bottom: 15,
    left: 60,
    right: 10,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  // scales
  $: xScale = scalePoint()
    .domain(data.map((d) => d.QID))
    .padding(0.3)
    .range([margin.left, width - margin.right]);

  $: yScale = scalePoint()
    .domain(data.map((d) => d.rank))
    .padding(1)
    .range([margin.top, height - margin.bottom]);

  $: colorScale = scaleOrdinal()
    .domain(data.map((d) => d.model))
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F"]);

  // the path generator
  $: pathLine = line()
    .x((d) => xScale(d.QID))
    .y((d) => yScale(d.rank));

  $: modelData = models.map((model) => data.filter((d) => d.model === model));
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
    <!-- y-ticks -->
    {#each yScale.domain() as tick}
      <g transform={`translate(${margin.left} ${yScale(tick) + 0})`}>
        <text
          class="axis-text"
          x="-5"
          y="0"
          text-anchor="end"
          dominant-baseline="middle"
          >{firstData
            .filter((d) => d.rank == tick)
            .map((d) => d.model)[0]}</text
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
      <path class="model-path-outer" d={pathLine(d)} stroke="white" />
      <path class="model-path" d={pathLine(d)} stroke={colorScale(models[i])} />
    {/each}

    {#each data as d}
      <g transform={`translate(${xScale(d.QID)}, ${yScale(d.rank)})`}>
        <circle
          r={(d.answer.length / 2) * 0 + 12}
          fill={colorScale(d.model)}
          stroke="white"
        />
        <text
          class="bump-text"
          text-anchor="middle"
          alignment-baseline="middle"
          color="white"
          fill="white">{d.rank}</text
        >
      </g>
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
    opacity: 0.13;
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
    stroke-width: 15;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .model-path-outer {
    fill: none;
    stroke-width: 18;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
</style>
