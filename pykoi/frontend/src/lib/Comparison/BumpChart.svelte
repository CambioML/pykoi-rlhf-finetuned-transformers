<script>
  import { scalePoint, scaleOrdinal } from "d3-scale";
  import { line } from "d3-shape";
  import { comparisonData } from "./store";

  $: firstData = $comparisonData
    .filter((d) => d.qid === 3)
    .map((d) => ({ model: d.model, rank: d.rank }));

  $: console.log("firstData", $comparisonData);

  $: models = Array.from(new Set($comparisonData.map((d) => d.model)));
  $: console.log("models", models);

  let outerHeight;
  let outerWidth;

  let margin = {
    top: 35,
    bottom: 15,
    left: 10,
    right: 0,
  };

  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  // scales
  $: xScale = scalePoint()
    .domain($comparisonData.map((d) => d.qid))
    .padding(0.3)
    .range([margin.left, width - margin.right]);

  $: sortedRanks = $comparisonData.map((d) => d.rank).sort((a, b) => a - b);

  $: yScale = scalePoint()
    .domain(sortedRanks)
    .padding(1)
    .range([margin.top, height - margin.bottom]);

  $: colorScale = scaleOrdinal()
    .domain($comparisonData.map((d) => d.model))
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F", "red"]);

  // the path generator
  $: pathLine = line()
    .x((d) => xScale(d.qid))
    .y((d) => yScale(d.rank));
  // .curve(curveBasis)

  $: modelData = models.map((model) =>
    $comparisonData.filter((d) => d.model === model)
  );

  $: console.log("md", modelData);
  $: console.log(
    "ranks",
    $comparisonData.map((d) => d.rank)
  );

  $: xTickArray =
    xScale.domain().length > 10
      ? xScale.domain().filter((_, index) => index % 2 === 0)
      : xScale.domain();

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
    {#each xTickArray as tick}
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
        <!-- <text
          class="axis-text"
          x="-5"
          y="0"
          text-anchor="end"
          dominant-baseline="middle"
          >{firstData
            .filter((d) => d.rank == tick)
            .map((d) => d.model)[0]}</text
        > -->
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
    {#each modelData as d, i (i)}
      <path
        class="model-path-outer"
        d={pathLine(d)}
        stroke="white"
        data-model={models[i]}
      />
      <path
        class="model-path"
        d={pathLine(d)}
        role="img"
        stroke={colorScale(models[i])}
        on:mouseover={() => highLight(i)}
        on:focus={() => highLight(i)}
        on:mouseout={unHighlight}
        on:blur={unHighlight}
        data-model={models[i]}
      />
    {/each}

    {#each $comparisonData as d (d.model + d.qid)}
      <g transform={`translate(${xScale(d.qid)}, ${yScale(d.rank)})`}>
        <circle
          r={(d.answer.length / 2) * 0 + 12}
          fill={colorScale(d.model)}
          stroke="white"
          class="model-circle"
          role="img"
          on:mouseover={() => highLight(i)}
          on:focus={() => highLight(i)}
          on:mouseout={unHighlight}
          on:blur={unHighlight}
          data-model={d.model}
        />
        <text
          class="bump-text"
          text-anchor="middle"
          alignment-baseline="middle"
          color="white"
          fill="white"
        >
          {d.rank}
        </text>
      </g>
    {/each}

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
    pointer-events: none;
  }

  .model-path {
    fill: none;
    stroke-width: 5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .model-path-outer {
    fill: none;
    stroke-width: 8;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .chart-subtitle {
    font-size: 12px;
  }
</style>
