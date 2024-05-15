from pydantic import BaseModel
from uuid import UUID

class ChatResponse(BaseModel):
    response: str
    chat_id: UUID | None