from core.modules.base import BaseModule

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document

from core.utils.logger import logger

class CacheModule(BaseModule):
    def __init__(self, index_db, threshold=0.95):
        self.name = "Cache"
        self.storage_context = index_db.get_or_create_storage_context("cache")
        self.index = VectorStoreIndex(nodes = [], storage_context = self.storage_context)

        self.retriever = self.index.as_retriever(similarity_top_k=1)
        self.threshold = threshold

    async def get_response(self, text):
        nodes = await self.retriever.aretrieve(text)

        if len(nodes) == 0:
            return None

        node = nodes[0]

        if node.score >= self.threshold:
            logger.info(f"Cache hit: {text} -> {node.score} -> {node.text}")
            return node.metadata["response"]

        logger.info(f"Cache miss: {text} -> {node.score} -> {node.text}")
        
        return None
    
    async def _forward(self, **kwargs):
        if kwargs.get("cache_hit", False) or kwargs.get("domain") == "out" or kwargs.get("intent_score") < 0.9:
            return kwargs
        
        text = kwargs.get("text")
        response = kwargs.get("response")

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

        if response is not None:
            kwargs["response"] = response
            kwargs["cache_hit"] = True

        return kwargs