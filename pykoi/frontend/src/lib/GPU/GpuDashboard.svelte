<script>
  import { writable } from "svelte/store";
  import { onMount } from "svelte";
  import SharedLine from "./SharedLine.svelte";
  import { gpu_data } from "./data.js";

  let initialDeviceData = gpu_data["device_status"];
  let initialData = initialDeviceData.map((d) => ({
    timestamp: d.timestamp,
    gpu: d.utilization.gpu,
    memory: d.utilization.memory,
  }));

  // Define the writable store
  export const gpuData = writable([]);

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

  onMount(() => {
    interval = setInterval(() => {
      let newTimestamp = new Date().toISOString();
      let newGpuValue = Math.floor(Math.random() * 100);
      let newMemoryValue = Math.floor(Math.random() * 100);

      // Update the store with the new data
      gpuData.update((data) => [
        ...data,
        {
          timestamp: newTimestamp,
          gpu: newGpuValue,
          memory: newMemoryValue,
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
  <SharedLine
    data={$gpuData}
    x="timestamp"
    y="gpu"
    color="red"
    strokeWidth="4"
    gradient="false"
    tick_opacity=".04"
    yAxisLine="false"
    xAxisText="Timestamp"
    yAxisText="GPU"
    xTicks="true"
    yTicks="true"
    title="GPU Usage"
    subtitle=""
  />
  <SharedLine
    data={$gpuData}
    x="timestamp"
    y="memory"
    color="red"
    strokeWidth="4"
    gradient="false"
    tick_opacity=".04"
    yAxisLine="false"
    xAxisText="Timestamp"
    yAxisText="Memory"
    xTicks="true"
    yTicks="true"
    title="Memory Usage"
    subtitle=""
    points="false"
  />
  <SharedLine
    data={$gpuData}
    x="timestamp"
    y="memory"
    color="red"
    strokeWidth="4"
    gradient="false"
    tick_opacity=".04"
    yAxisLine="false"
    xAxisText="Timestamp"
    yAxisText="Memory"
    xTicks="true"
    yTicks="true"
    title="Memory Usage"
    subtitle=""
    points="false"
  />
  <SharedLine
    data={$gpuData}
    x="timestamp"
    y="gpu"
    color="red"
    strokeWidth="4"
    gradient="false"
    tick_opacity=".04"
    yAxisLine="false"
    xAxisText="Timestamp"
    yAxisText="Memory"
    xTicks="true"
    yTicks="true"
    title="Memory Usage"
    subtitle=""
    points="false"
  />
  <SharedLine
    data={$gpuData}
    x="timestamp"
    y="memory"
    color="red"
    strokeWidth="4"
    gradient="false"
    tick_opacity=".04"
    yAxisLine="false"
    xAxisText="Timestamp"
    yAxisText="Memory"
    xTicks="true"
    yTicks="true"
    title="Memory Usage"
    subtitle=""
    points="false"
  />
  <SharedLine
    data={$gpuData}
    x="timestamp"
    y="gpu"
    color="red"
    strokeWidth="4"
    gradient="false"
    tick_opacity=".04"
    yAxisLine="false"
    xAxisText="Timestamp"
    yAxisText="Memory"
    xTicks="true"
    yTicks="true"
    title="Memory Usage"
    subtitle=""
    points="false"
  />
</div>

<style>
  /* #chart {
    display: flex;
    height: 50vh;
    width: 50vw;
    margin: auto;
    justify-content: center;
  } */
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
