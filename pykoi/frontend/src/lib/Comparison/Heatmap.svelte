<script>
  import { scaleBand, scaleOrdinal, scaleSequential } from "d3-scale";
  import { interpolateRgb } from "d3-interpolate";
  import { min, max } from "d3-array";
  import { format } from "d3-format";
  import { comparisonData } from "./store";

  const formatter = format(".1f");

  function calculateDiffs(data) {
    // Group the data by qid
    let groupedByqid = data.reduce((acc, curr) => {
      if (!acc[curr.qid]) {
        acc[curr.qid] = [];
      }
      acc[curr.qid].push(curr);
      return acc;
    }, {});

    // Get unique models
    let models = Array.from(new Set(data.map((d) => d.model)));

    // Create the initial diffs object
    let diffs = models.reduce((acc, model) => {
      acc[model] = models.reduce((acc2, model2) => {
        acc2[model2] = { sum: 0, count: 0 };
        return acc2;
      }, {});
      return acc;
    }, {});

    // Populate the diffs object
    for (let qid in groupedByqid) {
      let group = groupedByqid[qid];
      for (let model1 of models) {
        for (let model2 of models) {
          let data1 = group.find((d) => d.model === model1);
          let data2 = group.find((d) => d.model === model2);
          if (data1 && data2) {
            // diffs[model1][model2].sum += Math.abs(data1.rank - data2.rank);
            diffs[model1][model2].sum += data2.rank - data1.rank;
            diffs[model1][model2].count++;
          }
        }
      }
    }

    // Calculate the averages
    for (let model1 in diffs) {
      for (let model2 in diffs[model1]) {
        if (diffs[model1][model2].count > 0) {
          diffs[model1][model2] =
            diffs[model1][model2].sum / diffs[model1][model2].count;
        } else {
          diffs[model1][model2] = 0;
        }
      }
    }

    return diffs;
  }

  $: diffs = calculateDiffs($comparisonData);

  $: models = Array.from(new Set($comparisonData.map((d) => d.model)));

  let outerHeight = 500;
  let outerWidth = 500;
  let margin = {
    top: 50,
    bottom: 0,
    left: 65,
    right: 25,
  };
  $: width = outerWidth - margin.left - margin.right;
  $: height = outerHeight - margin.top - margin.bottom;

  $: xScale = scaleBand().range([0, width]).domain(models).padding(0.05);
  $: yScale = scaleBand().range([0, height]).domain(models).padding(0.05);

  $: diffValues = Object.values(diffs).flatMap((obj) => Object.values(obj));
  $: diffMin = min(diffValues);
  $: diffMax = max(diffValues);

  $: colorScale = scaleOrdinal()
    .domain(models)
    .range(["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F"]);

  $: colorScales = models.reduce((acc, model) => {
    acc[model] = scaleSequential()
      .domain([diffMin, diffMax])
      .interpolator(interpolateRgb("white", colorScale(model)));
    return acc;
  }, {});

  function getFillColor(rowModel, colModel) {
    let value = diffs[rowModel][colModel];
    if (value === 0) {
      return "white";
    } else {
      return value >= 0 ? colorScale(rowModel) : colorScale(colModel);
    }
  }
</script>

<div
  id="heatmap-holder"
  bind:offsetWidth={outerWidth}
  bind:offsetHeight={outerHeight}
>
  <svg width={outerWidth} height={outerHeight}>
    <!-- labels -->
    {#each models as model, i}
      <!-- top labels -->
      <text
        class="axis-text"
        x={margin.left + xScale(model) + xScale.bandwidth() / 2}
        y={margin.top - 10}
        text-anchor="middle"
      >
        {model}
      </text>

      <!-- left labels -->
      <text
        class="axis-text"
        x={margin.left - 10}
        y={margin.top + yScale(model) + yScale.bandwidth() / 2}
        text-anchor="end"
        dominant-baseline="middle"
      >
        {model}
      </text>
    {/each}

    <!-- cells -->
    {#each models as rowModel}
      {#each models as colModel}
        <rect
          x={margin.left + xScale(colModel)}
          y={margin.top + yScale(rowModel)}
          width={xScale.bandwidth()}
          height={yScale.bandwidth()}
          fill={getFillColor(rowModel, colModel)}
          rx="4"
          ry="4"
          class="model-path"
          data-model={rowModel}
        />
        <text
          x={margin.left + xScale(colModel) + xScale.bandwidth() / 2}
          y={margin.top + yScale(rowModel) + yScale.bandwidth() / 2}
          text-anchor="middle"
          dominant-baseline="middle"
          class="model-path"
          data-model={rowModel}
        >
          {formatter(diffs[rowModel][colModel])}
        </text>
      {/each}
    {/each}

    <text
      class="chart-title"
      y={margin.top / 2}
      x={margin.left}
      text-anchor="start">Relative Performance</text
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
  #heatmap-holder {
    height: 100%;
    width: 100%;
  }
  .axis-text {
    font-size: 12px;
  }
  .chart-subtitle {
    font-size: 12px;
  }
</style>
