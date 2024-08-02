from core.modules.base import BaseModule

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document

from core.utils.logger import logger

class CacheModule(BaseModule):
    def __init__(self, index_db, threshold=0.95):
        """
         @brief Initialize the cache. This is called by __init__ and should not be called directly. You should use
         @param index_db The database to use for caching
         @param threshold The threshold to use for
        """
        self.name = "Cache"
        self.storage_context = index_db.get_or_create_storage_context("cache")
        self.index = VectorStoreIndex(nodes = [], storage_context = self.storage_context)

        self.retriever = self.index.as_retriever(similarity_top_k=1)
        self.threshold = threshold

    async def get_response(self, text):
        """
         @brief Get response from cache. This is a wrapper around retriever. aretrieve to avoid hitting nodes that don't have a score > threshold
         @param text Text to look up.
         @return Response or None if not found or no response to look up. Note that it's possible to get a response that is less than threshold
        """
        nodes = await self.retriever.aretrieve(text)

        # Returns the node or None if there are no nodes.
        if len(nodes) == 0:
            return None

        node = nodes[0]

        # Returns the response from the cache.
        if node.score >= self.threshold:
            logger.info(f"Cache hit: {text} -> {node.score} -> {node.text}")
            return node.metadata["response"]

        logger.info(f"Cache miss: {text} -> {node.score} -> {node.text}")
        
        return None
    
    async def _forward(self, **kwargs):
        """
         @brief Forward to index if it's not cached. This is a hack to avoid hitting the index every time we get a response
         @return kwargs with the response
        """
        # Returns kwargs for cache hit or not.
        if kwargs.get("cache_hit", False) or kwargs.get("domain") == "out" or kwargs.get("intent_score") < 0.9:
            return kwargs
        
        text = kwargs.get("text")
        response = kwargs.get("response")

        # Inserts a response to the index.
        if response is not None:
            self.index.insert(
                Document(
                    text = text,
                    metadata = {
                        "response": response
                    },
                    excluded_embed_metadata_keys=["response"]
                )
            )
            return kwargs
        
        response = await self.get_response(text)

        # Set the response to the response.
        if response is not None:
            kwargs["response"] = response
            kwargs["cache_hit"] = True

        return kwargs