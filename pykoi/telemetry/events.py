from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Any


@dataclass
class TelemetryEvent:
    name: ClassVar[str]

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppStartEvent(TelemetryEvent):
    name: ClassVar[str] = "app_start"
