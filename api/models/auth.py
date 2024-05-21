from pydantic import BaseModel
from .user import UserRole
from enum import Enum

class Token(BaseModel):
    """
    Represents an access token used for authentication.

    Attributes:
        access_token (str): The actual access token.
        token_type (str): The type of the token (e.g., "Bearer").
    """
    access_token: str
    token_type: str

class AuthCode(Enum):
    """
    Enumeration for authentication response codes.

    Attributes:
        SUCCESS (int): Authentication succeeded.
        USER_NOT_EXIST (int): User does not exist.
        USER_EXIST (int): User already exists.
        WRONG_PASSWORD (int): Incorrect password.
    """
    SUCCESS = 0
    USER_NOT_EXIST = 1
    USER_EXIST = 2
    WRONG_PASSWORD = 3