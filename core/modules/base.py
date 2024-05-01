from abc import ABC, abstractmethod
from core.utils.logger import logger
import time

class BaseModule(ABC):
    def __init__(self) -> None:
        self.name = "Base"

    def exit(self, data):
        data["exit"] = True
        return data
    
    @abstractmethod
    def _forward(self, **kwargs) -> dict:
        pass

    def forward(self, **kwargs) -> dict:
        if "exit" in kwargs:
            return kwargs

        logger.debug(f"Forwarding {self.name} module")
        logger.debug(f"Input           : {kwargs}")

        start = time.time()
        data = self._forward(**kwargs)
        logger.debug(f"Forwarding time : {time.time() - start} s")

        kwargs.update(data)
        return kwargs