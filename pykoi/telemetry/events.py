from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Any
import platform
import torch
import requests


def identify_cloud_provider():
    try:
        # Check for AWS
        # AWS instances provide metadata at this URL
        r = requests.get('http://169.254.169.254/latest/meta-data/', timeout=2)
        if r.status_code == 200:
            return 'AWS'
        # Check each environment variable name
        for var in os.environ.values():
            if 'sagemaker' in var.lower():
                return 'AWS-SageMaker'

    except requests.exceptions.RequestException:
        pass

    try:
        # Check for GCP
        # GCP instances provide metadata at this URL
        r = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/', 
                         headers={'Metadata-Flavor': 'Google'}, timeout=2)
        if r.status_code == 200:
            return 'GCP'
    except requests.exceptions.RequestException:
        pass

    try:
        # Check for Azure
        # Azure instances provide metadata at this URL
        r = requests.get('http://169.254.169.254/metadata/instance', 
                         headers={'Metadata': 'True'}, timeout=2)
        if r.status_code == 200:
            return 'Azure'
    except requests.exceptions.RequestException:
        pass

    return 'Unknown'


@dataclass
class TelemetryEvent:
    name: ClassVar[str]

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppStartEvent(TelemetryEvent):
    name: ClassVar[str] = "app_start"
    start_time: float
    date_time: str
    gpu: bool = torch.cuda.is_available()
    cloud_provider: str = identify_cloud_provider()
    system: str = platform.system()
    release: str = platform.release()


@dataclass
class AppStopEvent(TelemetryEvent):
    name: ClassVar[str] = "app_end"
    end_time: float
    date_time: str
    duration: str
