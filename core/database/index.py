from transformers import AutoTokenizer

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, IndexNode, Document

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import RefDocInfo

from .database import Database

from core.modules import ClsModule, PreprocessingModule, LanguageModule
from core.config import get_config
from core.utils.logger import logger

from unidecode import unidecode
import joblib, pickle
from os import path
import chromadb
import os

from typing import Tuple, Optional


class CustomDocumentStore(SimpleDocumentStore):
    def _get_kv_pairs_for_insert(

        self, node: BaseNode, ref_doc_info: Optional[RefDocInfo], store_text: bool
    ) -> Tuple[
        Optional[Tuple[str, dict]],
        Optional[Tuple[str, dict]],
        Optional[Tuple[str, dict]],
    ]:  
        """
         @brief Get key value pairs for insert. This is overridden to handle IndexNodes as well as non - index nodes
         @param node Node to get data for
         @param ref_doc_info Reference doc info for this node
         @param store_text Whether to store text in the node
         @return Tuple of ( key value ) or ( None None None ) depending on store_text ( True ) or
        """
        # Return None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None None
        if isinstance(node, IndexNode):
            return None, None, None
        
        return super()._get_kv_pairs_for_insert(node, ref_doc_info, store_text)

class IndexDatabase:
    def __init__(self, path_data, model, max_length) -> None:
        """
         @brief Initialize Chroma embed. This is the entry point for the class. It sets up the chromadb client database preprocessing pipeline and parser
         @param path_data Path to the data directory
         @param model Name of the model to use ( ex : " cambiar " )
         @param max_length Maximum length of the embeddings ( ex : 100 )
         @return None if initialization failed otherwise the instance of ChromadbEmbed that was created will be returned in __
        """
        self.set_embed(model, max_length)
        
        self.path_data = path_data
        
        self.chroma_client = chromadb.PersistentClient(path_data)
        
        self.database = Database(
            path_indomain=get_config("Path", "indomain"),
            path_outdomain=get_config("Path", "outdomain")
        )

        self.language = LanguageModule()
        
        self.classification = ClsModule(
            model_inout=os.path.join(get_config("Path", "model"), get_config("Model", "inout")),
            model_intent=os.path.join(get_config("Path", "model"), get_config("Model", "intent")),
        )
        self.preprocessing = PreprocessingModule(
            database=self.database
        )

        self.pipeline = [self.language, self.preprocessing, self.classification]
        self.parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        
        self.reload()

    def get_or_create_storage_context(self, intent):
        """
         @brief Get or create a storage context for the given intent. If there is no storage context for the intent it will be created.
         @param intent The intent for which to create a storage context.
         @return The : class : ` StorageContext ` associated with the intent. Note that this is a singleton and will return an instance of
        """
        intent = unidecode(intent.lower())
        
        # Store the given intent in the storages.
        if intent not in self.storages:
            path_data = path.join(self.path_data, intent)

            # If path_data is not exist then the file is stored in the local storage directory.
            if not path.exists(path_data):
                storage_context = StorageContext.from_defaults(
                    vector_store=ChromaVectorStore(
                        chroma_collection=self.chroma_client.get_or_create_collection(intent)
                    ),
                )

                storage_context.persist(path_data)
            else:
                storage_context = StorageContext.from_defaults(
                    vector_store=ChromaVectorStore(
                        chroma_collection=self.chroma_client.get_or_create_collection(intent)
                    ),
                    persist_dir=path_data,
                )
            
            self.storages[intent] = storage_context

        return self.storages[intent]

    def reload(self):
        """
         @brief Reload data from disk. This is called after a change to the configuration or the node_dict is saved
        """
        self.storages = {}
        self.nodes_dict = joblib.load(path.join(self.path_data, "nodes_dict.pkl"))

    async def insert(self, files_path):
        """
         @brief Insert messages into LLM. This is the main method for the insertion process. It takes a list of files that are to be inserted and returns a list of messages that have been inserted.
         @param files_path The path to the files that are to be inserted.
         @return A list of messages that have been inserted and their intent ( s ). If there are no messages in the list an empty list is returned
        """
        # If files_path is a string it will be a list of files_path.
        if isinstance(files_path, str):
            files_path = [files_path]

        el = "\n"
        msg_file = '\n- '.join(files_path)
        logger.info(f"Inserting{el}- {msg_file}")
        
        reader = SimpleDirectoryReader(input_files=files_path)
        data = reader.load_data()
        documents = []

        # Add a Document to the list of documents.
        for i in data:
            # Add a Document to the document list.
            for line in i.text.split("\n"):
                documents.append(
                    Document(
                        text=line,
                    )
                )
        
        nodes = self.parser.get_nodes_from_documents(documents)

        logger.info(f"Total nodes: {len(nodes)}")
        
        nodes_intent = {}
        decline = 0
        # This method is called by the pipeline to forward the nodes to the pipeline.
        for node in nodes:
            data = {"text": node.get_content("llm")}

            # forward all modules in pipeline.
            for module in self.pipeline:
                data = await module.forward(**data)

            # Decines decline of the domain out.
            if data.get("domain", "out") == "out":
                decline += 1
                continue

            intent = data["intent"]

            # Add an intent to the list of nodes that have been added to the graph.
            if intent not in nodes_intent:
                nodes_intent[intent] = []

            node.metadata["pattern"] = intent
            node.excluded_llm_metadata_keys=["pattern"]

            index_node = IndexNode(
                text=intent,
                index_id=node.node_id,
            )
            
            nodes_intent[intent].append(node)
            nodes_intent[intent].append(index_node)
            
            self.nodes_dict[intent].update({node.node_id: node})
            self.nodes_dict[intent].update({index_node.node_id: index_node})

        # Return true if the decline is the number of nodes.
        if decline == len(nodes):
            return True
        
        logger.info(f"Decline: {decline/len(nodes)*100:.2f}% nodes")

        # print intent len nodes 2
        for intent, nodes in nodes_intent.items():
            logger.info(f"{intent:<25}: {len(nodes/2)} nodes")
        
        # Create a vector store for each node in the nodes_intent.
        for intent, nodes in nodes_intent.items():
            intent = unidecode(intent.lower())
            
            chroma_collection = self.chroma_client.get_or_create_collection(intent)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            vector_store.stores_text = False
            
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=CustomDocumentStore(),
            )

            index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=True, insert_batch_size=256)
            index.insert_nodes(nodes)
            
            storage_context.persist(path.join(get_config("Path", "Index"), intent))

        joblib.dump(self.nodes_dict, path.join(get_config("Path", "Index"), "nodes_dict.pkl"), protocol=pickle.HIGHEST_PROTOCOL)

        self.reload()
        
        return True

    def set_embed(self, model, max_length):
        """
         @brief Set embeddings for training. This is a wrapper around OptimumEmbedding to be used in tests
         @param model Path to pretrained model to use
         @param max_length Maximum length of the embeddings ( int
        """
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

        Settings.embed_model = OptimumEmbedding(
            folder_name=model,
            tokenizer=tokenizer,
            pooling='mean',
            max_length=max_length,
            device='cpu'
        )