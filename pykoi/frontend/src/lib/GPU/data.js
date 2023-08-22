import { writable } from "svelte/store";

export const tooltipX = writable(0);
export const hoveredIndexData = writable({});
export const gpuData = writable([]);

export const gpu_data = {
  device_status: [
    {
      timestamp: "2023-08-14T05:38:02.105572",
      utilization: {
        gpu: 0,
        memory: 0,
      },
    },
    {
      timestamp: "2023-08-14T05:38:17.105572",
      utilization: {
        gpu: 5,
        memory: 3,
      },
    },
    {
      timestamp: "2023-08-14T05:38:32.105572",
      utilization: {
        gpu: 10,
        memory: 7,
      },
    },
    {
      timestamp: "2023-08-14T05:38:47.105572",
      utilization: {
        gpu: 20,
        memory: 12,
      },
    },
    {
      timestamp: "2023-08-14T05:39:02.105572",
      utilization: {
        gpu: 25,
        memory: 15,
      },
    },
    {
      timestamp: "2023-08-14T05:39:17.105572",
      utilization: {
        gpu: 32,
        memory: 21,
      },
    },
    {
      timestamp: "2023-08-14T05:39:32.105572",
      utilization: {
        gpu: 40,
        memory: 30,
      },
    },
    {
      timestamp: "2023-08-14T05:39:47.105572",
      utilization: {
        gpu: 48,
        memory: 40,
      },
    },
    {
      timestamp: "2023-08-14T05:40:02.105572",
      utilization: {
        gpu: 55,
        memory: 43,
      },
    },
    {
      timestamp: "2023-08-14T05:40:17.105572",
      utilization: {
        gpu: 60,
        memory: 48,
      },
    },
    {
      timestamp: "2023-08-14T05:40:32.105572",
      utilization: {
        gpu: 62,
        memory: 52,
      },
    },
    {
      timestamp: "2023-08-14T05:40:47.105572",
      utilization: {
        gpu: 65,
        memory: 58,
      },
    },
    {
      timestamp: "2023-08-14T05:41:02.105572",
      utilization: {
        gpu: 70,
        memory: 60,
      },
    },
    {
      timestamp: "2023-08-14T05:41:17.105572",
      utilization: {
        gpu: 73,
        memory: 65,
      },
    },
    {
      timestamp: "2023-08-14T05:41:32.105572",
      utilization: {
        gpu: 80,
        memory: 70,
      },
    },
  ],
};
