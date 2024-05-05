from pydantic import BaseModel
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    username: str
    role: UserRole = UserRole.USER

class UserInDB(User):
    hashed_password: str