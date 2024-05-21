from abc import ABC, abstractmethod
from core.utils.logger import logger
import time

class BaseModule(ABC):
    def __init__(self) -> None:
        """
         @brief Initializes the object. This is called by __init__ and should not be called directly. The name is set to " Base ".
         @return A reference to the object to be used for chaining or None if no call is required ( for example if self. is_valid () is False
        """
        self.name = "Base"

    def exit(self, data):
        """
         @brief Exit the test. This is a no - op in case we don't have a test to run
         @param data data passed to the test
         @return data with exit flag set to True in order to be able to test the result of the test without
        """
        data["exit"] = True
        return data
    
    @abstractmethod
    async def _forward(self, **kwargs) -> dict:
        """
         @brief Forward method to be implemented by subclasses. This method is called in response to an incoming request from the client.
         @return The response to the request as a dictionary with keys ` ` name ` ` and ` ` type ` `
        """
        pass

    async def forward(self, **kwargs) -> dict:
        """
         @brief Forward data to module. This is a coroutine. If you don't want to call it in a coroutine you can do so by passing exit = True in the kwargs.
         @return dict with data to pass to module's _forward method. It is updated with the data returned
        """
        logger.debug(f"Forwarding {self.name} module")
        logger.debug(f"Input           : {kwargs}")

        start = time.time()

        data = kwargs
        # Forward the process to the remote server.
        if "exit" not in kwargs:
            data = await self._forward(**kwargs)
            
        logger.debug(f"Forwarding time : {time.time() - start} s")

        kwargs.update(data)
        return kwargs