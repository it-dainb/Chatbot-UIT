from pydantic import BaseModel
from uuid import UUID

class ChatResponse(BaseModel):
    """
    Represents a chat response containing a message and an optional chat ID.

    Attributes:
        response (str): The chat message.
        chat_id (UUID, optional): An optional chat ID associated with the response.
    """

    response: str
    chat_id: UUID | None