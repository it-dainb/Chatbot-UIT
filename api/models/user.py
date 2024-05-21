from pydantic import BaseModel
from enum import Enum

class UserRole(str, Enum):
    """
    Enumeration for user roles.

    Attributes:
        ADMIN (str): Represents an admin user role.
        USER (str): Represents a regular user role.
    """
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    """
    Represents a user.

    Attributes:
        username (str): The user's username.
        role (UserRole, optional): The user's role (default is USER).
    """ 
    username: str
    role: UserRole = UserRole.USER

class UserInDB(User):
    """
    Represents a user stored in the database.

    Attributes:
        hashed_password (str): The hashed password associated with the user.
    """

    hashed_password: str