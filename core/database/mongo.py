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
        """
         @brief Initialize the chatbot client. This is the constructor for the class. You need to call this before any other methods are called
         @param uri The URI of the server
         @param secret_key The secret key to use
         @param algorithm The algorithm to use ( HS256 HS384
        """
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client["chatbot"]
        self.users = self.db["users"]

        self.secret_key = secret_key
        self.algorithm = algorithm

    def get_current_time(self, return_int = False):
        """
         @brief Get the current time. This is used to determine if we are in the past or not. If it's the past we'll return the number of seconds since January 1 1970
         @param return_int Whether to return int or float
         @return A datetime object or a float depending on return_int. Default is False which means it's the
        """
        current_time = datetime.now(timezone.utc)

        # Return the current time in UTC.
        if return_int:
            return timegm(current_time.utctimetuple())

        return current_time

    def get_hashed_password(self, password: str):
        """
         @brief Hashes and returns the password. This is used to verify the password before logging in. The password is hashed using the : func : ` pwd_context. hash ` function.
         @param password The password to hash. Must be a string.
         @return The hashed password as a string. Note that it is possible to have different hashes depending on the platform
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password, hashed_password):
        """
         @brief Verify a password against a hashed password. This is a wrapper around pwd_context. verify () to avoid having to re - hash the password every time it is called
         @param plain_password plain password to be verified
         @param hashed_password hashed password that should be verified
         @return True if the password
        """
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, username, expires: int = 10):
        """
         @brief Creates an access token for a user. The token is valid for 10 minutes. If you want to create a new token use : py : meth : ` get_access_token `
         @param username The username of the user
         @param expires The number of minutes after which the token will expire
         @return A JSON Web Token ( JWT ) that can be used to access the user's data in a
        """
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
        """
         @brief Creates a user in the database. This is a convenience method for users that don't have a username and / or password.
         @param username The username of the user to create. If this is a username it will be hashed using : func : ` get_hashed_password ` before
         @param password
         @param role
        """
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
        """
         @brief Get a user by username. This is used to get information about a user in the user table.
         @param username The username of the user. It must be unique among all users in the database.
         @return The user or None if not found. Note that the role is set to UserRole. USER_
        """
        c = self.users.find_one({"_id": username})

        # Return None if c is None.
        if c is None:
            return None
        
        user = User(
            username=c["_id"],
            role=UserRole(c["role"]),
            hashed_password=c["password"]
        )

        return user
        

    def authenticate_user(self, username: str, password: str) -> AuthCode:
        """
         @brief Authenticate a user by username and password. This is a low level method for authenticating a user.
         @param username The username of the user to authenticate. It must be unique among all users in the system.
         @param password The password that will be used to authenticate the user.
         @return The AuthCode indicating success or failure of the authentication process. If authentication was successful the return value will be one of the AUTH_CODE_ * constants
        """
        user = self.get_user_by_username(username)
        
        # Returns AuthCode. USER_NOT_EXIST if user is not None.
        if user is None:
            return AuthCode.USER_NOT_EXIST
        
        # Returns the authentication code for the user.
        if not self.verify_password(password, user.hashed_password):
            return AuthCode.WRONG_PASSWORD

        return AuthCode.SUCCESS

    def decode_access_token(self, token: str):
        """
         @brief Decodes an access token. Decoding is done using the secret key and the algorithm specified in the constructor.
         @param token The access token to decode. Must be a JSON Web Token ( JWT ).
         @return The decoded token as a Python object. >>> token ='abcdefghijklmnopqrstuvwxyz '
        """
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

    def valid_access_token(self, token: str):
        """
         @brief Check if token is valid. This is used to validate access tokens that are generated by OAuth 2. 0.
         @param token The token to check. Must be a JSON Web Token
         @return True if the token is
        """
        try:
            data = self.decode_access_token(token)

            # Return True if exp is less than current time
            if data["exp"] <= self.get_current_time(return_int=True):
                return False
        except JWTError:
            return False
        
        return True