import nest_asyncio
nest_asyncio.apply()

import os
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.routers import engine, user, auth

app = FastAPI(
    title = "Chatbot UIT"
)

app.include_router(user)
app.include_router(engine)
app.include_router(auth)

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST"), port=int(os.getenv("API_PORT", 8080)), reload=False, use_colors=True)