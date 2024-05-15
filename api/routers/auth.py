from typing import Annotated

from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from api.models.auth import Token
from api.models.user import User, UserRole

from core.database import MongoDatabase
import os

auth = APIRouter(prefix="/auth", tags=["auth"])

mongo = MongoDatabase(os.getenv("MONGO_URI"), os.getenv("SECRET_KEY"), os.getenv("ALGORITHM"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not mongo.valid_access_token(token):
        raise credentials_exception

    payload = mongo.decode_access_token(token)

    username: str = payload.get("username")

    if username is None:
        raise credentials_exception

    user = mongo.get_user_by_username(username=username)
    
    if user is None:
        raise credentials_exception
    
    return user

async def is_admin(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> bool:
    user = await get_current_user(token)
    
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You do not have permission to access this resource",
            headers={"WWW-Authenticate": "Bearer"},
        )

@auth.post("/token")
async def get_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    access_token = mongo.create_access_token(form_data.username, expires = 10)

    return Token(access_token=access_token, token_type="bearer")