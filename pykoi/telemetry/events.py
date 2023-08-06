"""This module contains telemetry events for PyKoi."""

import os
from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Any
import platform
import torch
import requests


def identify_cloud_provider():
    """
    Identify the cloud provider that the code is running on.
    """
    try:
        # Check for AWS
        # AWS instances provide metadata at this URL
        r = requests.get("http://169.254.169.254/latest/meta-data/", timeout=2)
        if r.status_code == 200:
            return "AWS"
        # Check each environment variable name
        for var in os.environ.values():
            if "sagemaker" in var.lower():
                return "AWS-SageMaker"

    except requests.exceptions.RequestException:
        pass

    try:
        # Check for GCP
        # GCP instances provide metadata at this URL
        r = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/",
            headers={"Metadata-Flavor": "Google"},
            timeout=2,
        )
        if r.status_code == 200:
            return "GCP"
    except requests.exceptions.RequestException:
        pass

    try:
        # Check for Azure
        # Azure instances provide metadata at this URL
        r = requests.get(
            "http://169.254.169.254/metadata/instance",
            headers={"Metadata": "True"},
            timeout=2,
        )
        if r.status_code == 200:
            return "Azure"
    except requests.exceptions.RequestException:
        pass

    return "Unknown"


@dataclass
class TelemetryEvent:
    """Represents a telemetry event.

    Attributes:
        name (ClassVar[str]): The name of the telemetry event.
    """

    name: ClassVar[str]

    @property
    def properties(self) -> Dict[str, Any]:
        """Returns the properties of the telemetry event.

        Returns:
            Dict[str, Any]: The properties of the telemetry event.
        """
        return asdict(self)


@dataclass
class AppStartEvent(TelemetryEvent):
    """
    A telemetry event that is triggered when the application starts.

    Attributes:
        name (str): The name of the event.
        start_time (float): The time when the application started.
        date_time (str): The date and time when the application started.
        gpu (bool): Whether or not a GPU is available.
        cloud_provider (str): The name of the cloud provider, if running on a cloud platform.
        system (str): The name of the operating system.
        release (str): The release version of the operating system.
    """

    name: ClassVar[str] = "app_start"
    start_time: float
    date_time: str
    gpu: bool = torch.cuda.is_available()
    cloud_provider: str = identify_cloud_provider()
    system: str = platform.system()
    release: str = platform.release()


@dataclass
class AppStopEvent(TelemetryEvent):
    """
    A telemetry event that is triggered when the application stops.

    Attributes:
        name (str): The name of the event.
        end_time (float): The time when the application stopped.
        date_time (str): The date and time when the application stopped.
        duration (str): The duration of the application.
    """

    name: ClassVar[str] = "app_end"
    end_time: float
    date_time: str
    duration: str
