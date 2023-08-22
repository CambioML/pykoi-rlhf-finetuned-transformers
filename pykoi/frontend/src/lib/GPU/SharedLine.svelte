<!-- Line plot with shared tooltip -->
<script>
  import { extent } from "d3-array";
  import { scaleLinear, scaleTime, scaleOrdinal } from "d3-scale";
  import { timeFormat } from "d3-time-format";
  import { line, curveBasis, area } from "d3-shape";
  import { draw } from "svelte/transition";
  import { multiFormat } from "./timeUtils.js";
  import { onMount } from "svelte";
  import { bisector } from "d3-array";
  import { pointer, select, selectAll } from "d3-selection";
  import { hoveredIndexData, gpuData } from "./data.js";

  export let data;
  export let x = "myX";
  export let y = "myY";
  export let marginTop = 40;
  export let marginBottom = 70;
  export let marginLeft = 90;
  export let marginRight = 30;
  export let stroke = "#fde24f";
  export let tooltipColor = "red";
  export let tooltipFontSize = "12";
  export let strokeWidth = 2;
  export let gradient = true;
  export let xAxisText = "x-axis";
  export let yAxisText = "y-axis";
  export let xAxisLine = "true";
  export let yAxisLine = "true";
  export let title = "title";
  export let subtitle = "subtitle";
  export let xTicks = true;
  export let yTicks = false;
  export let tick_opacity = 1;
  export let background = "";
  export let colors = ["#FF5470", "#1B2D45", "#00EBC7", "#FDE24F"];
  export let timeScaleFormat = "";
  export let animateRender = "true";
  export let points = "true";

  $: width = 400;
  $: height = 400;

  let margin = {
    top: +marginTop,
    bottom: +marginBottom,
    left: +marginLeft,
    right: +marginRight,
  };

  $: innerWidth = +width - margin.left - margin.right;
  $: innerHeight = +height - margin.top - margin.bottom;

  const containerId = "plotano-line-" + Math.round(Math.random() * 1e8);

  function isValidDate(d) {
    if (typeof d === "string") {
      d = new Date(d);
    }
    return d instanceof Date && !isNaN(d.getTime());
  }

  $: isArrayY = Array.isArray(y);
  $: yValues = isArrayY ? y : [y];

  $: useTimeScale = data && data.length > 0 && isValidDate(data[0][x]);

  $: xAccessor = useTimeScale ? (d) => new Date(d[x]) : (d) => d[x];

  $: xScale = useTimeScale
    ? scaleTime()
        .domain(extent(data.map(xAccessor)))
        .range([0, innerWidth])
    : scaleLinear()
        .domain(extent(data.map(xAccessor)))
        .range([0, innerWidth]);

  function getCombinedExtent(data, yValues) {
    let minVals = yValues.map((y) => extent(data.map((d) => d[y]))[0]);
    let maxVals = yValues.map((y) => extent(data.map((d) => d[y]))[1]);
    return [Math.min(...minVals), Math.max(...maxVals)];
  }

  $: yScaleDomain = isArrayY
    ? getCombinedExtent(data, yValues)
    : extent(data.map((d) => d[y]));
  $: yScale = scaleLinear().domain(yScaleDomain).range([innerHeight, 0]);

  $: colorScale = scaleOrdinal().domain([1, 2]).range(colors);

  $: pathsLine = yValues.map(
    (yValue) =>
      line()
        .x((d) => xScale(useTimeScale ? new Date(d[x]) : d[x]))
        .y((d) => yScale(d[yValue]))
    //   .curve(curveBasis)
  );

  $: pathsArea = yValues.map(
    (yValue) =>
      area()
        .x((d) => xScale(useTimeScale ? new Date(d[x]) : d[x]))
        .y0(innerHeight)
        .y1((d) => yScale(d[yValue]))
    //   .curve(curveBasis)
  );

  let formatDate = timeScaleFormat ? timeFormat(timeScaleFormat) : multiFormat;
  let rendered = animateRender === "true" ? false : true;
  onMount(() => {
    rendered = true;
  });

  let m = { x: 0, y: 0 };

  function hideTooltip() {
    selectAll("#annotation-tooltip").style("opacity", 0);

    selectAll("#annotation-tooltip-text").style("opacity", 0);

    selectAll(".annotation-tooltip-circle").style("opacity", 0);
  }

  function showTooltip() {
    selectAll("#annotation-tooltip").style("opacity", 1);
    selectAll("#annotation-tooltip-text").style("opacity", 1);
    selectAll(".annotation-tooltip-circle").style("opacity", 1);
  }

  function handleMousemove(event) {
    showTooltip();
    m.x = event.clientX - event.currentTarget.getBoundingClientRect().left;
    m.y = event.clientY - event.currentTarget.getBoundingClientRect().top;
    if (m.x < margin.left) {
      m.x = margin.left;
      hideTooltip();
    }
    if (m.x > width - margin.right) {
      m.x = width - margin.right;
      hideTooltip();
    }
    if (m.y < margin.top) {
      m.y = margin.top;
      hideTooltip();
    }
    if (m.y > height - margin.bottom) {
      m.y = height - margin.bottom;
      hideTooltip();
    }
  }

  function handleMouseout(event) {
    hideTooltip();
  }

  function handleHover(event) {
    if ($gpuData.length > 0) {
      const pEvent = pointer(event);
      let [mouseX, mouseY] = pEvent;
      mouseX = mouseX + margin.left * 0;
      const xPos = xScale.invert(mouseX);
      const dataBisector = bisector(xAccessor).left;
      const bisectionIndex = dataBisector(data, xPos);
      hoveredIndexData.set(data[bisectionIndex]);
      tooltipText = $hoveredIndexData[y];
    }
  }

  function handleHoverOut(event) {
    // hideTooltip()
  }

  // $: tooltipX = 0;
  // $: tooltipY = 0;
  $: tooltipX = xScale(xAccessor($hoveredIndexData)) + margin.left;
  $: tooltipY = yScale($hoveredIndexData[y]) + margin.top;
  $: tooltipText = $hoveredIndexData[y];
</script>

<div
  id={containerId}
  class="line-chart-holder"
  bind:clientWidth={width}
  bind:clientHeight={height}
  style:background
>
  <svg
    {width}
    {height}
    on:touchmove={handleMousemove}
    on:mousemove={handleMousemove}
    on:mouseleave={handleMouseout}
    on:touchend={handleMouseout}
    role="img"
    aria-label="Interactive chart"
  >
    <g transform={`translate(${margin.left}, ${margin.top})`}>
      <defs>
        <linearGradient id="area-gradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" style={`stop-color: ${stroke}; stop-opacity: 1`} />
          <stop offset="70%" style="stop-color: white; stop-opacity: 1" />
        </linearGradient>
      </defs>

      {#each xScale.ticks() as tick}
        <g transform={`translate(${xScale(tick)}, ${innerHeight})`}>
          {#if xTicks === "true"}
            <line
              class="axis-tick"
              x1="0"
              x2={0}
              y1={0}
              y2={-innerHeight}
              stroke="black"
              stroke-dasharray="4"
              opacity={tick_opacity}
            />
          {/if}
          <text
            class="axis-text"
            y={useTimeScale ? "15" : "25"}
            x="0"
            text-anchor="end"
            transform={useTimeScale ? "rotate(0)" : ""}
          >
            {useTimeScale ? formatDate(tick) : tick}
          </text>
        </g>
      {/each}

      {#each yScale.ticks() as tick}
        <g transform={`translate(0, ${yScale(tick)})`}>
          {#if yTicks === "true"}
            <line
              class="axis-tick"
              x1={0}
              x2={innerWidth}
              y1="0"
              y2={0}
              stroke="red"
              stroke-dasharray="4"
              opacity={tick_opacity}
            />
          {/if}
          <text
            class="axis-text"
            x="-5"
            y="0"
            text-anchor="end"
            dominant-baseline="middle">{tick}</text
          >
        </g>
      {/each}

      {#each yValues as yValue, i}
        {#if gradient === "true"}
          <path
            class="area-path"
            d={pathsArea[i](data)}
            fill="url(#area-gradient)"
          />
        {/if}
        {#if rendered}
          <path
            transition:draw={{ duration: 2000 }}
            class="outer-path"
            stroke-width={+strokeWidth + 1}
            stroke={"white"}
            d={pathsLine[i](data)}
          />
          <path
            transition:draw={{ duration: 1000 }}
            class="inner-path"
            stroke-width={+strokeWidth}
            stroke={colorScale(i)}
            d={pathsLine[i](data)}
          />
        {/if}
      {/each}

      {#if points === "true"}
        {#each data as d}
          <circle
            cx={xScale(useTimeScale ? new Date(d[x]) : d[x])}
            cy={yScale(d[y])}
            r="4"
            fill="black"
          />
        {/each}
      {/if}

      {#if xAxisLine === "true"}
        <line
          class="axis-line x-axis-line"
          x1={0}
          x2={innerWidth}
          y1={innerHeight}
          y2={innerHeight}
        />
      {/if}
      {#if yAxisLine === "true"}
        <line
          class="axis-line y-axis-line"
          x1={0}
          x2={0}
          y1={0}
          y2={innerHeight}
        />
      {/if}

      <text
        class="axis-label"
        y={innerHeight + margin.bottom / 2}
        x={innerWidth / 2}
        text-anchor="middle">{xAxisText}</text
      >

      <!-- rect to track mouse event  -->
      <rect
        class="overlay-rect"
        width={innerWidth}
        height={innerHeight}
        fill="red"
        opacity="0"
        on:touchmove={handleHover}
        on:mousemove={handleHover}
        on:mouseleave={handleHoverOut}
        role="img"
        aria-label="Interactive overlay"
      />
    </g>
    <text
      class="axis-label"
      y={margin.left / 2}
      x={-(innerHeight / 2 + margin.top)}
      text-anchor="middle"
      transform="rotate(-90)">{yAxisText}</text
    >

    <text class="chart-title" y={15} x={margin.left} text-anchor="start"
      >{title}</text
    >
    <text
      class="chart-subtitle"
      y={15 * 2 + 2}
      x={margin.left}
      opacity=".6"
      text-anchor="start">{subtitle}</text
    >

    <line
      id="annotation-tooltip"
      x1={tooltipX}
      y1={height - margin.bottom}
      x2={tooltipX}
      y2={margin.top}
      stroke-width="2"
      stroke={tooltipColor}
    />
    <circle
      class="annotation-tooltip-circle"
      r="0"
      cx={tooltipX}
      cy={tooltipY}
      fill="red"
      stroke="white"
    />
    <text
      x={tooltipX}
      y={tooltipY}
      dy="0.3em"
      fill="black"
      stroke={tooltipColor}
      font-size={tooltipFontSize}
      opacity="1"
      id="annotation-tooltip-text"
    >
      {tooltipText}
    </text>
  </svg>
</div>

<style>
  :root {
    --black: black;
  }
  #annotation-tooltip {
    pointer-events: none;
    stroke-dasharray: 4;
    opacity: 0;
  }
  .annotation-tooltip-circle {
    pointer-events: none;
    opacity: 0;
  }
  #annotation-tooltip-text {
    pointer-events: none;
    stroke-linejoin: round;
    paint-order: stroke fill;
    stroke-width: 6px;
    fill: white;
    font-size: 12;
    opacity: 0;
  }
  .line-chart-holder {
    height: 100%;
    width: 100%;
  }
  .axis-line {
    stroke-width: 3;
    /* stroke: var(--black); */
    stroke: black;
    fill: none;
  }
  .axis-tick {
    stroke-width: 1;
    stroke: black;
    fill: none;
  }
  .axis-text {
    font-family: Work Sans;
    font-size: 12px;
  }

  .inner-path {
    fill: none;
    stroke-linecap: round;
  }
  .outer-path {
    opacity: 1;
    fill: none;
    stroke-linecap: round;
  }

  .area-path {
    opacity: 0.76;
  }
</style>
