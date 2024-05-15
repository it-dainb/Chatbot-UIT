from pydantic import BaseModel
from .user import UserRole
from enum import Enum

class Token(BaseModel):
    access_token: str
    token_type: str

class AuthCode(Enum):
    SUCCESS = 0
    USER_NOT_EXIST = 1
    USER_EXIST = 2
    WRONG_PASSWORD = 3