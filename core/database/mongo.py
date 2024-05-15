from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError

from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from calendar import timegm

from api.models.user import UserRole, User
from api.models.auth import AuthCode


import logging
logging.getLogger('passlib').setLevel(logging.ERROR)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class MongoDatabase:
    def __init__(self, uri: str, secret_key: str = "SECRET", algorithm: str = "HS256"):
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client["chatbot"]
        self.users = self.db["users"]

        self.secret_key = secret_key
        self.algorithm = algorithm

    def get_current_time(self, return_int = False):
        current_time = datetime.now(timezone.utc)

        if return_int:
            return timegm(current_time.utctimetuple())

        return current_time

    def get_hashed_password(self, password: str):
        return pwd_context.hash(password)

    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, username, expires: int = 10):
        user = self.get_user_by_username(username)
        
        to_encode = {
            "username": user.username,
            "role": user.role
        }

        expire = self.get_current_time() + timedelta(minutes=expires)

        to_encode.update({"exp": expire})

        print(to_encode)
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        return encoded_jwt

    def create_user(self, username: str, password: str, role: UserRole = UserRole.USER):
        try:
            self.users.insert_one({
                "_id": username,
                "password": self.get_hashed_password(password),
                "role": role.value
            })
        except DuplicateKeyError:
            return AuthCode.USER_EXIST

        return AuthCode.SUCCESS

    def get_user_by_username(self, username: str):
        c = self.users.find_one({"_id": username})

        if c is None:
            return None
        
        user = User(
            username=c["_id"],
            role=UserRole(c["role"]),
            hashed_password=c["password"]
        )

        return user
        

    def authenticate_user(self, username: str, password: str) -> AuthCode:
        user = self.get_user_by_username(username)
        
        if user is None:
            return AuthCode.USER_NOT_EXIST
        
        if not self.verify_password(password, user.hashed_password):
            return AuthCode.WRONG_PASSWORD

        return AuthCode.SUCCESS

    def decode_access_token(self, token: str):
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

    def valid_access_token(self, token: str):
        try:
            data = self.decode_access_token(token)

            if data["exp"] <= self.get_current_time(return_int=True):
                return False
        except JWTError:
            return False
        
        return True