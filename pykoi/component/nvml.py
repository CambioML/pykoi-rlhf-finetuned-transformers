"""nvml component"""


from pykoi.ops.nvml import Nvml as Nv
from pykoi.component.base import Component


class Nvml(Component):
    """
    Nvml class represents a nvml component.

    Attributes:
        nv (Nv): The nvml component.
    """

    def __init__(self, **kwargs):
        """
        Initialize a new instance of Nvml.

        Args:
            kwargs: Additional properties for the component.
        """
        super().__init__(None, "Nvml", **kwargs)
        self.nvml = Nv()
