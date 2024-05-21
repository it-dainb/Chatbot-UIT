from llama_index.core import VectorStoreIndex

from llama_index.core.retrievers import RecursiveRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from core.config.config import get_config

class RetrieveModule:
    def __init__(self, index_db, top_vector: int = 3, top_bm25: int = None, verbose=None):
        """
         @brief Initialize the retriever. This is the base class for all class methods. It should be called from the __init__ method of the parent class
         @param index_db The path to the index database
         @param top_vector The number of top vectors to be used in the training set. Default is 3. A value of 3 will result in a 3x3 matrix of size ( n_samples n
         @param top_bm25 The number of top BM25 vectors
         @param verbose The level of verbosity to
        """
        super().__init__()

        self.index_db = index_db
        
        self.retrievers = {}

        self.top_vector = top_vector

        # Set the top vector to the vector.
        if top_bm25 is None:
            top_bm25 = top_vector

        self.top_bm25 = top_bm25

        # If verbose is not set the verbose flag to true.
        if verbose is None:
            verbose = get_config("Debug", "verbose")
        
        self.verbose = verbose

    def get_retriever_by_intent(self, intent, **kargs):
        """
         @brief Get retriever by intent. This is a wrapper around VectorStoreIndex. as_retriever () and BM25Retriever. from_defaults ()
         @param intent intent to look up.
         @return retriever ( : class : ` pyfasstools. retriever. Retriever ` )
        """
        # If the index_db is storages and retrieval is not in the index_db. storages this method will be called when the index_db is storages.
        if self.index_db.storages == {}:
            self.retrievers = {}
        
        # Create a new instance of the appropriate retriever for the given intent.
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