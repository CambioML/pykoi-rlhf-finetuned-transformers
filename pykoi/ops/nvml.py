""" NVML (NVIDIA Management Library). """

from typing import Any, Dict, List

import pynvml


class DeviceStatus:
    """
    Device status.
    """

    def __init__(
        self, device_id: int, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> None:
        """
        Initialize device status.

        Args:
            device_id (int): Device ID.
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        """
        self.device_id = device_id
        self.ecc_errors = self.get_ecc_errors(handle)
        self.utilization = self.get_utilization(handle)
        self.processes = self.get_active_processes(handle)
        self.clocks_pstate = self.get_clocks_and_pstate(handle)
        self.temp_fan = self.get_temperature_and_fan(handle)
        self.power = self.get_power(handle)
        self.identification = self.get_identification(handle)

    def get_ecc_errors(
        self, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> Dict[str, int]:
        """
        Get ECC errors.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            dict: ECC errors.
        """
        ecc_mapping = {
            "single_bit_ecc_volatile_ecc": (
                pynvml.NVML_SINGLE_BIT_ECC,
                pynvml.NVML_VOLATILE_ECC,
            ),
            "single_bit_ecc_aggregate_ecc": (
                pynvml.NVML_SINGLE_BIT_ECC,
                pynvml.NVML_AGGREGATE_ECC,
            ),
            "double_bit_ecc_volatile_ecc": (
                pynvml.NVML_DOUBLE_BIT_ECC,
                pynvml.NVML_VOLATILE_ECC,
            ),
            "double_bit_ecc_aggregate_ecc": (
                pynvml.NVML_DOUBLE_BIT_ECC,
                pynvml.NVML_AGGREGATE_ECC,
            ),
        }

        return {
            key: pynvml.nvmlDeviceGetTotalEccErrors(handle, *value)
            for key, value in ecc_mapping.items()
        }

    def get_utilization(
        self, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> Dict[str, int]:
        """
        Get utilization.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            dict: Utilization.
        """
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {"gpu": utilization.gpu, "memory": utilization.memory}

    def get_active_processes(
        self, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> List[Dict[str, int]]:
        """
        Get active processes.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            list: Active processes.
        """
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        return [
            {
                "pid": process.pid,
                "name": process.name,
                "memory_used": process.usedGpuMemory,
            }
            for process in processes
        ]

    def get_clocks_and_pstate(
        self, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> Dict[str, int]:
        """
        Get clocks and performance state.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            dict: Clocks and performance state.
        """
        return {
            "current_graphics_clock": pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_GRAPHICS
            ),
            "current_memory_clock": pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            ),
            "max_graphics_clock": pynvml.nvmlDeviceGetMaxClockInfo(
                handle, pynvml.NVML_CLOCK_GRAPHICS
            ),
            "max_memory_clock": pynvml.nvmlDeviceGetMaxClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            ),
            "performance_state": pynvml.nvmlDeviceGetPerformanceState(handle),
        }

    def get_temperature_and_fan(
        self, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> Dict[str, int]:
        """
        Get temperature and fan speed.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            dict: Temperature and fan speed.
        """
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan_speed = "Not Supported"

        return {"temperature": temperature, "fan_speed": fan_speed}

    def get_power(self, handle: pynvml.nvmlDeviceGetHandleByIndex) -> Dict[str, int]:
        """
        Get power usage and power limit.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            dict: Power usage and power limit.
        """
        return {
            "power_usage": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "power_limit": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        }

    def get_identification(
        self, handle: pynvml.nvmlDeviceGetHandleByIndex
    ) -> Dict[str, str]:
        """
        Get identification.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): Device handle.

        Returns:
            dict: Identification.
        """
        return {
            "serial_number": pynvml.nvmlDeviceGetSerial(handle),
            "vbios_version": pynvml.nvmlDeviceGetVbiosVersion(handle),
            "name": pynvml.nvmlDeviceGetName(handle),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            dict: Dictionary.
        """
        return {
            "device_id": self.device_id,
            "ecc_errors": self.ecc_errors,
            "utilization": self.utilization,
            "processes": self.processes,
            "clocks_pstate": self.clocks_pstate,
            "temp_fan": self.temp_fan,
            "power": self.power,
            "identification": self.identification,
        }


class Nvml:
    """
    NVML.
    """

    def __init__(self):
        """Initialize NVML."""
        self._has_gpu = False
        try:
            pynvml.nvmlInit()
            self._has_gpu = True
        except pynvml.NVMLError:
            self._has_gpu = False
            # raise RuntimeError("Failed to initialize NVML library.")

    def get(self) -> List[Dict[str, Any]]:
        """
        Get device status.

        Returns:
            list: Device status.
        """
        if self._has_gpu:
            device_count = pynvml.nvmlDeviceGetCount()
            return [
                DeviceStatus(i, pynvml.nvmlDeviceGetHandleByIndex(i)).to_dict()
                for i in range(device_count)
            ]
        return []

    def __del__(self):
        """Shutdown NVML."""
        if self._has_gpu:
            pynvml.nvmlShutdown()
