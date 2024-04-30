from core.database.database import Database
from core.config.config import get_config
import time
from core.modules import PreprocessingModule, ClsModule, GenerateModule, RetrieveModule, OptimumRerank

import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.openai import OpenAI

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

preprocessing = PreprocessingModule(
    database=database, 
    accent_path=os.path.join(get_config("Path", "model"), get_config("Model", "accent")),
    check_accent=get_config("Chat", "check_accent"),
    threshold=get_config("Chat", "accent_threshold")
)

classification = ClsModule(
    model_inout=os.path.join(get_config("Path", "model"), get_config("Model", "inout")),
    model_intent=os.path.join(get_config("Path", "model"), get_config("Model", "intent")),
)

generation = GenerateModule(
    llm=llm,
    max_tokens_memory=get_config("Chat", "max_memory_token"),
    retriever=RetrieveModule(
        model=os.path.join(get_config("Path", "model"), get_config("Model", "embed")),
        path_data=get_config("Retriever", "data"),
        top_vector=get_config("Retriever", "top_vectors"),
        top_bm25=get_config("Retriever", "top_bm25"),
        max_length=get_config("Model", "max_length")
    ),
    rerank=OptimumRerank(
        model=os.path.join(get_config("Path", "model"), get_config("Model", "rerank")),
        max_length=get_config("Model", "max_length"),
        top_n=get_config("Retriever", "top_rerank"),
    )
)

pipeline = [preprocessing, classification, generation]

while(True):
    query = input("Nhập truy vấn: ")
    id = input("Nhập id: ")

    data = {
        "text": query
    }
    if id != "":
        data["chat_id"] = id
        
    for module in pipeline:
        start = time.time()
        data = module.forward(**data)
        print(data)
        print("Time Inference: ", time.time() - start)