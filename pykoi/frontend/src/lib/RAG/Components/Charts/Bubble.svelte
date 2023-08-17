<script>
  import { scale } from "svelte/transition";
  import { scaleLinear } from "d3-scale";
  import { forceSimulation, forceX, forceY, forceCollide } from "d3-force";
  import { extent, quantile } from "d3-array";
  import { formatBytes } from "../../../../utils.js";
  import { select, selectAll } from "d3-selection";
  import { uploadedFiles } from "../../../store.js";

  let width = 600;
  let height = 350;
  let hover = true;
  let hovered = false;

  const move = (x, y) => `transform: translate(${x}px, ${y}px)`;

  $: renderedData = $uploadedFiles.map((d) => ({
    ...d,
    x: width / 2,
    y: height / 2,
    tickCount: 0,
  }));

  const xDomain = [-4, 4];
  const yDomain = [-1, 1];

  $: console.log($uploadedFiles.map((d) => d.size).sort((a, b) => a - b));

  $: activeForceX = forceX().x(width / 2);
  $: activeForceY = forceY().y(height / 2);
  $: activeForceCollide = forceCollide()
    .radius((d) => radiusScale(d.size) + 1)
    .iterations(3);

  $: thirdQuantile = quantile(
    $uploadedFiles.map((d) => d.size).sort((a, b) => a - b),
    0.85
  );

  // scales
  $: xScale = scaleLinear().domain(xDomain).range([5, width]);

  $: yScale = scaleLinear().domain(yDomain).range([height, 5]);

  $: radiusScale = scaleLinear()
    .domain(extent($uploadedFiles, (d) => d.size))
    .range([12, 60]);

  $: simulation = forceSimulation()
    .nodes(renderedData)
    .on("tick", () => {
      renderedData = [...renderedData];
    });

  $: {
    simulation.force("x", activeForceX);
    simulation.force("y", activeForceY);
    simulation.force("collide", activeForceCollide);
    simulation.alpha(0.02);
    simulation.restart();
  }

  $: {
    radiusScale.domain(extent($uploadedFiles, (d) => d.size));
    renderedData.forEach((d) => (d.radius = radiusScale(d.size)));
    simulation.nodes(renderedData); // Update the nodes in the simulation
    activeForceCollide.radius((d) => d.radius + 0.5); // Update the collide force
  }

  function showTooltip() {
    hover = true;
    hovered = true;
    const g = select(this);
    selectAll(".file-circle-g").select("text").style("opacity", 0);
    g.raise();
    g.select("circle").attr("fill", "var(--red)");
    g.select("text").style("opacity", 1);
  }

  const hideTooltip = () => {
    const g = selectAll(".file-circle-g");
    g.select("circle").attr("fill", "var(--yellow)");
    g.select("text").style("opacity", 0);
    hovered = false;
    setTimeout(() => {
      hover = false;
    }, 1000);
  };

  $: if (hover === false) {
    if (!hovered) {
      selectAll(".large").select("text").style("opacity", 1);
    }
  }
</script>

<div id="network-chart" bind:offsetWidth={width} bind:offsetHeight={height}>
  <svg {width} {height}>
    {#each renderedData as d, i}
      <g
        class="file-circle-g {d.size > thirdQuantile ? 'large' : ''}"
        style={move(d.x, d.y)}
        on:mouseover={showTooltip}
        on:mouseout={hideTooltip}
        on:focus={showTooltip}
        on:blur={hideTooltip}
        role="img"
      >
        <circle
          transition:scale|local={{
            duration: 1000,
          }}
          r={radiusScale(d.size)}
          fill={"var(--primary"}
          stroke={"var(--white)"}
          stroke-width="1"
          fill-opacity={0.95}
        />
        <text
          class="bubble-file-text"
          text-anchor="middle"
          opacity={d.size > thirdQuantile ? 1 : 0}
          transition:scale|local={{
            duration: 1000,
          }}
          >{d.file}
          <tspan x="0" y="15">{formatBytes(d.size)}</tspan></text
        >
      </g>
    {/each}
  </svg>
</div>

<style>
  svg {
    background: conic-gradient(
        from 90deg at 1px 1px,
        #0000 90deg,
        rgba(0, 0, 0, 0.04) 0
      )
      0 0/20px 20px;
    border: var(--line);
  }
  .file-circle-g:hover {
    opacity: 1;
  }

  .bubble-file-text {
    font-size: calc(0.99 * var(--smallText));
    pointer-events: none;
    stroke: var(--white);
    stroke-width: 4px;
    fill: var(--black);
    stroke-linejoin: round;
    paint-order: stroke fill;
  }

  circle {
    transition: all 0.3s ease;
  }

  circle:hover {
    stroke: var(--black);
    stroke-width: 3;
  }
</style>
