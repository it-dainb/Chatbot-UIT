from fastapi import APIRouter, Depends, Form, UploadFile, File, status, HTTPException

from .auth import get_current_user, is_admin

from api.models.engine import ChatResponse

from ..pipelines import pipelines, index_database
from typing import Annotated
from uuid import UUID

import os
import shutil
from datetime import datetime, timezone

from passlib.hash import hex_sha256

engine = APIRouter(prefix="/engine", tags=["engine"], dependencies=[Depends(get_current_user)])

@engine.post("/chat", response_model=ChatResponse)
async def chat(
    text: Annotated[str, Form(..., description="Input text to chat")], 
    chat_id: Annotated[UUID, Form(..., description="Chat ID for tracking conversation features")] = None
) -> ChatResponse:

    data = {
        "text": text,
        "chat_id": str(chat_id) if chat_id is not None else None,
    }

    for module in pipelines:
        data = await module.forward(**data)
    
    return ChatResponse(
        response=data["response"],
        chat_id=data["chat_id"],
    )

@engine.post("/upload", dependencies=[Depends(is_admin)])
async def upload(
    file: Annotated[UploadFile, File()]
):

    file_type = file.filename.split(".")[-1]
    file_name = ".".join(file.filename.split(".")[:-1])
    
    filename = f"{file_name}_{datetime.now(timezone.utc).timestamp()}"
    filename = hex_sha256.hash(filename)    
    filename = f"{filename}.{file_type}"
    

    os.makedirs('uploaded', exist_ok=True)

    try:
        file_path = f'uploaded/{filename}'
        
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        await index_database.insert(file_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded file {file_name}.{file_type}"}