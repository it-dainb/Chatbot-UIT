from core.modules import BaseModule
from core.modules.retrieve.retrieve import RetrieveModule
from core.modules.rerank.rerank import OptimumRerank

from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import LongContextReorder

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from uuid import uuid4

DEFAULT_SYSTEM_PROMPT = \
"""You are an advising education enrollment chatbot for VNUHCM - University of Information Technology (UIT), able to have normal interactions.
Your mission is to direct people to join UIT school and alway suggest people to join UIT.

Here are the relevant documents for the context:
Author: Nguyễn Bá Đại - MSSV: 21521914 - Email: it.dainb@gmail.com
Co-Author: Nguyễn Tấn Dũng - MSSV: 21521978 - Email: 21521978@gm.it.edu.vn
Admissions hotline: 090.883.1246
Admissions email: tuyensinh@uit.edu.vn
ĐGNL: Đánh Giá Năng Lực
{context_str}
Instruction: Use the previous chat history, or the context above, to interact and help the user."""


class GenerateModule(BaseModule):
    def __init__(self, retriever: RetrieveModule, max_tokens_memory=1000, rerank: OptimumRerank = None, llm: OpenAI = None):

        if llm is None:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            llm = OpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.0,
                api_key=os.getenv("api_key"),
                api_version=os.getenv("api_version"),
            )

        Settings.llm = llm
        
        self.postprocessor = []
        self.max_tokens_memory = max_tokens_memory

        if rerank is not None:
            self.postprocessor.append(rerank)

        self.postprocessor.append(LongContextReorder())

        self.retriever = retriever

        self.memories = {}

        self.name = "Generate"

    def create_chat(self):
        chat_id = str(uuid4())
        self.memories[chat_id] = ChatMemoryBuffer.from_defaults(token_limit=self.max_tokens_memory)
        
        return chat_id
    
    def get_engine(self, domain, intent):
        if domain == "out":
            return SimpleChatEngine.from_defaults(system_prompt=DEFAULT_SYSTEM_PROMPT)

        return ContextChatEngine.from_defaults(
            retriever=self.retriever.get_retriever_by_intent(intent),
            context_template=DEFAULT_SYSTEM_PROMPT
        )

    def chat(self, text, engine, memory):
        response = engine.chat(text, chat_history=memory.get_all())
        response = str(response)
        
        memory.set(engine.chat_history)

        return response

    def _forward(self, **kwargs) -> dict:

        new_user = False
        if "chat_id" not in kwargs:
            kwargs["chat_id"] = self.create_chat()
            new_user = True
        
        chat_id = kwargs["chat_id"]
        memory = self.memories[chat_id]

        text = kwargs["text"]
        domain = kwargs["domain"]
        intent = kwargs["intent"]

        engine = self.get_engine(domain, intent)
        response = self.chat(text, engine, memory)
        
        return {
            "response": response,
            "new_user": new_user,
            "chat_id": chat_id
        }