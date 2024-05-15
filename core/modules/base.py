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
    async def _forward(self, **kwargs) -> dict:
        pass

    async def forward(self, **kwargs) -> dict:
        logger.debug(f"Forwarding {self.name} module")
        logger.debug(f"Input           : {kwargs}")

        start = time.time()

        data = kwargs
        if "exit" not in kwargs:
            data = await self._forward(**kwargs)
            
        logger.debug(f"Forwarding time : {time.time() - start} s")

        kwargs.update(data)
        return kwargs