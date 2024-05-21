from fastapi import APIRouter, Form, status, HTTPException
from typing import Annotated, Optional

from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.models.user import UserRole
from api.models.auth import AuthCode, Token

from .auth import get_token, mongo

user = APIRouter(prefix="/user", tags=["user"])

@user.post("/register", response_model=Token)
async def register(
    username : Annotated[str, Form()], 
    password: Annotated[str, Form()], 
    role : Optional[Annotated[UserRole, Form()]] = UserRole.USER
) -> Token:
    """_summary_

    Args:
        username (Annotated[str, Form): _description_
        password (Annotated[str, Form): _description_
        role (Optional[Annotated[UserRole, Form, optional): _description_. Defaults to UserRole.USER.

    Raises:
        HTTPException: _description_

    Returns:
        Token: _description_
    """
    auth_code: AuthCode = mongo.create_user(
        username=username,
        password=password,
        role=role
    )

    if auth_code == AuthCode.USER_EXIST:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exist",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token: Token = await get_token(OAuth2PasswordRequestForm(username=username, password=password))
    
    return token

@user.post("/login", response_model=Token)
async def register(
    username : Annotated[str, Form()], 
    password: Annotated[str, Form()]
) -> Token:
    """_summary_

    Args:
        username (Annotated[str, Form): _description_
        password (Annotated[str, Form): _description_

    Raises:
        HTTPException: _description_

    Returns:
        Token: _description_
    """
    auth_code: AuthCode = mongo.authenticate_user(
        username=username,
        password=password
    )

    if auth_code in [AuthCode.USER_NOT_EXIST, AuthCode.WRONG_PASSWORD]:
        if auth_code == AuthCode.USER_NOT_EXIST:
            message = "User not exist please register"
        elif auth_code == AuthCode.WRONG_PASSWORD:
            message = "Wrong password please try again"
            
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            headers={"WWW-Authenticate": "Bearer"},
        )

    token: Token = await get_token(OAuth2PasswordRequestForm(username=username, password=password))
    
    return token