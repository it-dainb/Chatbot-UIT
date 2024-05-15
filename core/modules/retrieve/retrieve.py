from llama_index.core import VectorStoreIndex

from llama_index.core.retrievers import RecursiveRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from core.config.config import get_config

class RetrieveModule:
    def __init__(self, index_db, top_vector: int = 3, top_bm25: int = None, verbose=None):
        super().__init__()

        self.index_db = index_db
        
        self.retrievers = {}

        self.top_vector = top_vector

        if top_bm25 is None:
            top_bm25 = top_vector

        self.top_bm25 = top_bm25

        if verbose is None:
            verbose = get_config("Debug", "verbose")
        
        self.verbose = verbose

    def get_retriever_by_intent(self, intent, **kargs):
        if self.index_db.storages == {}:
            self.retrievers = {}
        
        if intent not in self.retrievers:
            storage_context = self.index_db.get_or_create_storage_context(intent)

            index = VectorStoreIndex(
                nodes = [],
                storage_context = storage_context
            )

            vector_retriever = RecursiveRetriever(
                "root",
                retriever_dict={"root": index.as_retriever(similarity_top_k=self.top_vector, **kargs)},
                node_dict=self.index_db.nodes_dict[intent],
                verbose=self.verbose,
            )

            bm25_retriever = BM25Retriever.from_defaults(similarity_top_k=self.top_bm25, docstore=storage_context.docstore, verbose=True)

            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=None,
                num_queries=1,
                use_async=True,
                verbose=self.verbose,
            )

            self.retrievers[intent] = retriever

        return self.retrievers[intent]