from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from transformers import AutoTokenizer
import chromadb

from llama_index.core.retrievers import RecursiveRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from core.config.config import get_config

from unidecode import unidecode
import joblib
from os import path

class RetrieveModule:
    def __init__(self, model: str, path_data: str, top_vector: int = 3, top_bm25: int = None, max_length:int = 256, verbose=None):
        super().__init__()

        self.set_embed(model, max_length)
        
        self.path_data = path_data
        
        self.chroma_client = chromadb.PersistentClient(path_data)
        self.nodes_dict = joblib.load(path.join(path_data, "nodes_dict.pkl"))
        
        self.storages = {}
        self.retrievers = {}

        self.top_vector = top_vector

        if top_bm25 is None:
            top_bm25 = top_vector

        self.top_bm25 = top_bm25

        if verbose is None:
            verbose = get_config("Debug", "verbose")
        
        self.verbose = verbose

    def set_embed(self, model, max_length):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

        Settings.embed_model = OptimumEmbedding(
            folder_name=model,
            tokenizer=tokenizer,
            pooling='mean',
            max_length=max_length
        )

    def get_storage_context(self, intent, **kargs):
        intent = unidecode(intent.lower())
        
        if intent not in self.storages:
            path_data = path.join(self.path_data, intent)
            
            storage_context = StorageContext.from_defaults(
                vector_store=ChromaVectorStore(
                    chroma_collection=self.chroma_client.get_or_create_collection(intent)
                ),
                persist_dir=path_data,
            )
            
            self.storages[intent] = storage_context

        return self.storages[intent]

    def get_retriever_by_intent(self, intent, **kargs):
        if intent not in self.retrievers:
            storage_context = self.get_storage_context(intent, **kargs)

            index = VectorStoreIndex(
                nodes = [],
                storage_context = storage_context
            )

            vector_retriever = RecursiveRetriever(
                "root",
                retriever_dict={"root": index.as_retriever(similarity_top_k=self.top_vector, **kargs)},
                node_dict=self.nodes_dict[intent],
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