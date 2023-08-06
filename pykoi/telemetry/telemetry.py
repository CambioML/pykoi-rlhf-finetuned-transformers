"""This module contains telemetry for PyKoi."""
import os
import sys
import uuid
import logging

from typing import Dict, Any
from pathlib import Path
from posthog import Posthog

import pykoi
from pykoi.telemetry.events import TelemetryEvent


logger = logging.getLogger(__name__)


class Telemetry:
    """
    A class for capturing telemetry events and sending them to Posthog.

    Attributes:
        _user_id_path (str): The path to the file containing the user ID.
        _user_id (str): The user ID.
        _unknown (str): The default value for the user ID if it cannot be determined.
    """

    _user_id_path = str(Path.home() / ".pykoi" / "user_id")
    _user_id = None
    _unknown = "UNKNOWN"

    def __init__(self, enable_telemetry: bool = True) -> None:
        """
        Initializes a new instance of the Telemetry class.

        Args:
            enable_telemetry (bool): Whether to enable telemetry or not.
        """
        self._posthog = Posthog(
            project_api_key="phc_fpnBxwfQtbQfHf20NvAKeTsDxOysYjhBo9w4pDSvnEr",
            host="https://app.posthog.com",
        )

        if not enable_telemetry or "test" in sys.modules:
            self._posthog.disabled = True
        else:
            logger.info("telemetry enabled")

    def capture(self, event: TelemetryEvent) -> None:
        """
        Captures a telemetry event and sends it to Posthog.

        Args:
            event (TelemetryEvent): The telemetry event to capture.
        """
        try:
            self._posthog.capture(
                self.user_id,
                event.name,
                {**event.properties, "pykoi_context": self.context},
            )
        except Exception as e:
            logging.error(f"Failed to capture event {event.name}: {e}")

    @property
    def context(self) -> Dict[str, Any]:
        """
        Gets the context for the telemetry events.

        Returns:
            A dictionary containing the context for the telemetry events.
        """
        self._context = {
            "pykoi_version": pykoi.__version__,
        }
        return self._context

    @property
    def user_id(self) -> str:
        """
        Gets the user ID for the telemetry events.

        Returns:
            The user ID for the telemetry events.
        """
        # If _user_id is already set, return it
        if self._user_id:
            return self._user_id

        # Check if the user_id file exists
        if os.path.exists(self._user_id_path):
            self._read_user_id_from_file()
        else:
            self._create_user_id_file()

        # If something went wrong and _user_id is still None, set it to _unknown
        if self._user_id is None:
            self._user_id = self._unknown

        return self._user_id

    def _read_user_id_from_file(self) -> None:
        """
        Reads the user ID from the file.
        """
        try:
            with open(self._user_id_path, "r") as f:
                self._user_id = f.read()
        except Exception:
            pass

    def _create_user_id_file(self) -> None:
        """
        Creates a new user ID file.
        """
        try:
            os.makedirs(os.path.dirname(self._user_id_path), exist_ok=True)
            with open(self._user_id_path, "w") as f:
                new_user_id = str(uuid.uuid4())
                f.write(new_user_id)
            self._user_id = new_user_id
        except Exception:
            pass
