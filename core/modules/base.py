from abc import ABC, abstractmethod
from core.utils.logging import logger
import time

class BaseModule(ABC):
    def __init__(self) -> None:
        self.name = "Base"
    
    @abstractmethod
    def _forward(self, **kwargs) -> dict:
        pass

    def forward(self, **kwargs) -> dict:
        logger.debug(f"Forwarding {self.name} module")
        logger.debug(f"Input           : {kwargs}")

        start = time.time()
        data = self._forward(**kwargs)
        logger.debug(f"Forwarding time : {time.time() - start} s")
        
        kwargs.update(data)
        return kwargs