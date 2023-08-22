<script>
  import { writable } from "svelte/store";
  import { onMount } from "svelte";
  import SharedLine from "./SharedLine.svelte";
  import { gpuData } from "./data.js";

  // let initialDeviceData = gpu_data["device_status"];
  // let initialData = initialDeviceData.map((d) => ({
  //   timestamp: d.timestamp,
  //   gpu: d.utilization.gpu,
  //   memory: d.utilization.memory,
  // }));

  // Define the writable store

  async function retrieveDBData() {
    const response = await fetch("/chat/comparator/db/retrieve");
    const data = await response.json();
    console.log("uploooo", data);

    // const data = await response.json();
    //     console.log("uploooo", data);

    //     // Assuming that the returned data has a similar structure to gpu_data.device_status
    //     const formattedData = data.device_status.map((d) => ({
    //       timestamp: d.timestamp,
    //       gpu: d.utilization.gpu,
    //       memory: d.utilization.memory,
    //     }));

    //     dataStore.set(formattedData);
  }

  let interval;

  let vals = [
    { y: "gpu", title: "GPU", yAxis: "GPU" },
    { y: "memory", title: "memory", yAxis: "memory" },
    { y: "temperature", title: "temperature", yAxis: "temperature" },
    { y: "fan_speed", title: "fan_speed", yAxis: "fan_speed" },
    {
      y: "current_graphics_clock",
      title: "current_graphics_clock",
      yAxis: "current_graphics_clock",
    },
    {
      y: "current_memory_clock",
      title: "current_memory_clock",
      yAxis: "current_memory_clock",
    },
  ];

  onMount(() => {
    interval = setInterval(() => {
      let newTimestamp = new Date().toISOString();
      let newGpuValue = Math.floor(Math.random() * 100);
      let newMemoryValue = Math.floor(Math.random() * 100);
      let newTemperature = Math.floor(Math.random() * 100);
      let newFanSpeed = Math.floor(Math.random() * 100);
      let newGraphicsClock = Math.floor(Math.random() * 100);
      let newMemoryClock = Math.floor(Math.random() * 100);

      // Update the store with the new data
      gpuData.update((data) => [
        ...data,
        {
          timestamp: newTimestamp,
          gpu: newGpuValue,
          memory: newMemoryValue,
          temperature: newTemperature,
          fan_speed: newFanSpeed,
          current_graphics_clock: newGraphicsClock,
          current_memory_clock: newMemoryClock,
        },
      ]);
    }, 500);

    // Cleanup the interval when component is destroyed
    return () => {
      clearInterval(interval);
    };
  });
</script>

<h3>GPU Monitoring</h3>

<div id="chart">
  {#each vals as d}
    <SharedLine
      data={$gpuData}
      x="timestamp"
      y={d.y}
      yAxisLine="false"
      xAxisText="Timestamp"
      yAxisText={d.yAxis}
      title={d.title}
      tooltipColor="black"
      tooltipFontSize="15"
      subtitle=""
      points="false"
    />
  {/each}
</div>

<style>
  #chart {
    display: grid;
    grid-template-columns: repeat(3, 33%);
    grid-template-rows: repeat(4, 40%);
    height: 100vh;
    width: 90vw;
    margin: 3rem auto;
    justify-content: center;
  }
</style>
