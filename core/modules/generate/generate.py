from core.modules import BaseModule
from core.modules.retrieve.retrieve import RetrieveModule
from core.modules.rerank.rerank import OptimumRerank

from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import LongContextReorder

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from uuid import uuid4, UUID

DEFAULT_SYSTEM_PROMPT = \
"""## INSTRUCTION
You are an advising education enrollment chatbot for VNUHCM - University of Information Technology (UIT), able to have normal interactions.
Your mission is to direct people to join UIT school and alway suggest people to join UIT.

Chatbot's Author: Nguyễn Bá Đại - MSSV: 21521914 - Email: it.dainb@gmail.com
Chatbot's Co-Author: Nguyễn Tấn Dũng - MSSV: 21521978 - Email: 21521978@gm.uit.edu.vn
Admissions hotline: 090.883.1246
Admissions email: tuyensinh@uit.edu.vn

## CONTEXT
Here are the relevant documents of the UIT schools for the context:
ĐGNL: Đánh Giá Năng Lực
{context_str}
Instruction: Use the previous chat history, or the context above, to interact and help the user."""


class GenerateModule(BaseModule):
    def __init__(self, retriever: RetrieveModule, max_tokens_memory=1000, rerank: OptimumRerank = None, llm: OpenAI = None):
        """
        @brief Initialize the LLLM. This is the entry point for the LLLM generator. You can pass in a custom retriever and it will be used to generate tokens.
        @param retriever The retriever to use for generating tokens
        @param max_tokens_memory The maximum memory to use for token re - ordering
        @param rerank The optimum rerank to use
        @param llm The openAI instance to use for l
        """

        # OpenAI object for the OpenAI class
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

        # Append rerank to the postprocessor.
        if rerank is not None:
            self.postprocessor.append(rerank)

        self.postprocessor.append(LongContextReorder())

        self.retriever = retriever

        self.memories = {}

        self.name = "Generate"

    def create_chat(self, chat_id):
        """
         @brief Creates a chat in the database. If chat_id is None or'' a new id will be generated
         @param chat_id The id of the chat
         @return The id of the chat in the database or None if no id was given ( for testing purposes not to be confused with id
        """
        # If chat_id is not None or empty string it will be used as a unique chat_id.
        if chat_id is None or chat_id == '':
            chat_id = str(uuid4())
        
        self.memories[chat_id] = ChatMemoryBuffer.from_defaults(token_limit=self.max_tokens_memory)
        
        return chat_id

    def get_or_create_memory(self, chat_id = None):
        """
         @brief Get or create memory. If chat_id is None or empty will create a new chat. Otherwise will return the memories with the given chat_id
         @param chat_id Unique identifier for the chat
         @return Tuple of memory and
        """
        # Create a new chat if not already created.
        if chat_id is None or chat_id.strip() not in self.memories or chat_id == '':
            chat_id = self.create_chat(chat_id)

        return self.memories[chat_id], chat_id
    
    def get_engine(self, domain, intent) -> BaseChatEngine:
        """
         @brief Get chat engine for given domain and intent. This is used to determine which retriever to use and the context template to use.
         @param domain Domain to get engine for. Can be " out "
         @param intent Intent to look up.
         @return BaseChatEngine that can be used to interact with the retriever and context of the given intent. In the case of domain " out " the system prompt is
        """
        # SimpleChatEngine for the domain out.
        if domain == "out":
            return SimpleChatEngine.from_defaults(system_prompt=DEFAULT_SYSTEM_PROMPT)

        return ContextChatEngine.from_defaults(
            retriever=self.retriever.get_retriever_by_intent(intent),
            context_template=DEFAULT_SYSTEM_PROMPT
        )

    async def chat(self, text, engine: BaseChatEngine, memory):        
        """
         @brief Send a chat message to the chat engine. This is a wrapper around achat that handles history of messages sent to the chat engine
         @param text The text to send to the engine
         @param engine The chat engine that is used to send messages
         @param memory The memory to use for this chat
         @return The response from the chat engine or None if there was an error sending the message ( in which case the error will be logged
        """
        response = await engine.achat(text, chat_history=memory.get_all())
        response = str(response)
        
        memory.set(engine.chat_history)

        return response

    async def _forward(self, **kwargs) -> dict:
        """
        @brief Forward to chat. Args : chat_id : Unique identifier for the chat. The id will be generated if not provided.
        @return dict with keys " response " and " chat_id ". Response is sent to the user if any
        """

        chat_id = kwargs.get("chat_id")
        memory, chat_id = self.get_or_create_memory(chat_id)

        # Cache the chat message.
        if kwargs.get("cache_hit", False):
            memory.put(
                ChatMessage(
                    content=kwargs["chat_text"],
                    role=MessageRole.USER
                )
            )

            memory.put(
                ChatMessage(
                    content=kwargs["response"],
                    role=MessageRole.ASSISTANT
                )
            )

            return {
                "response": kwargs["response"],
                "chat_id": chat_id
            }

        text = kwargs["chat_text"]
        domain = kwargs["domain"]
        intent = kwargs["intent"]

        engine: BaseChatEngine = self.get_engine(domain, intent)
        response = await self.chat(text, engine, memory)
        
        return {
            "response": response,
            "chat_id": chat_id
        }
