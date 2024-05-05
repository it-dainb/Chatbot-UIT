from core.modules import PreprocessingModule, ClsModule, GenerateModule, RetrieveModule, OptimumRerank, LanguageModule
from core.database import Database
from core.database.index import IndexDatabase
from core.config import get_config

from llama_index.llms.openai import OpenAI

import colorlog
import os

colorlog.getLogger("CHATBOT_UIT").setLevel(get_config("Debug", "log_level"))

llm = OpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.0,
    api_key=os.getenv("api_key"),
    api_version=os.getenv("api_version"),
)

database = Database(
    path_indomain=get_config("Path", "indomain"),
    path_outdomain=get_config("Path", "outdomain")
)

index_database = IndexDatabase(
    path_data=get_config("Retriever", "data"),
    model=os.path.join(get_config("Path", "model"), get_config("Model", "embed")),
    max_length=get_config("Model", "max_length")
)

preprocessing = PreprocessingModule(
    database=database, 
    accent_path=os.path.join(get_config("Path", "model"), get_config("Model", "accent")),
    check_accent=get_config("Chat", "check_accent"),
    threshold=get_config("Chat", "accent_threshold")
)

language_detect = LanguageModule(
    threshold=get_config("Chat", "language_threshold")
)

classification = ClsModule(
    model_inout=os.path.join(get_config("Path", "model"), get_config("Model", "inout")),
    model_intent=os.path.join(get_config("Path", "model"), get_config("Model", "intent")),
)

generation = GenerateModule(
    llm=llm,
    max_tokens_memory=get_config("Chat", "max_memory_token"),
    retriever=RetrieveModule(
        index_db=index_database,
        top_vector=get_config("Retriever", "top_vectors"),
        top_bm25=get_config("Retriever", "top_bm25"),
    ),
    rerank=OptimumRerank(
        model=os.path.join(get_config("Path", "model"), get_config("Model", "rerank")),
        max_length=get_config("Model", "max_length"),
        top_n=get_config("Retriever", "top_rerank"),
    )
)

pipelines = [language_detect, preprocessing, classification, generation]