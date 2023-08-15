<script>
  import { writable } from "svelte/store";
  import { onMount } from "svelte";
  import Line from "./Line.svelte";
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
    // const dbRows = data["data"];
    // const formattedRows = dbRows.map((d) => ({
    //   model: d.model,
    //   answer: d.answer,
    //   qid: parseInt(d.qid),
    //   rank: parseInt(d.rank),
    // }));
    // $comparisonData = [...formattedRows];
  }

  let interval;

  onMount(() => {
    interval = setInterval(() => {
      let newTimestamp = new Date().toISOString();
      let newGpuValue = Math.floor(Math.random() * 100); // Generate a random value between 0 and 100 for the sake of this example
      let newMemoryValue = Math.floor(Math.random() * 100); // Generate a random value between 0 and 100

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

<div id="chart">
  <Line
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
    title="GPU Usage"
    subtitle="Measured every 4 seconds"
  />
</div>

<style>
  #chart {
    display: flex;
    height: 50vh;
    width: 50vw;
    margin: auto;
    justify-content: center;
  }
</style>
