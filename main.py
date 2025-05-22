from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import uuid
from datetime import datetime
import os
from typing import List, Optional

app = FastAPI(
    title="Nisme AI API",
    description="Nisme AI chat uchun backend API",
    version="1.0.0"
)

# CORS sozlamalari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI konfiguratsiyasi
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ma'lumotlar bazasi simulyatsiyasi (asl loyihada PostgreSQL/MongoDB ishlatish yaxshiroq)
chat_sessions = {}

# Pydantic modellari
class Message(BaseModel):
    text: str
    is_user: bool
    timestamp: str

class ChatSession(BaseModel):
    session_id: str
    created_at: str
    title: str
    messages: List[Message]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Foydalanuvchi xabarini qabul qilib, AI javobini qaytaradi"""
    try:
        # Yangi sessiya yoki mavjud sessiyani tekshirish
        if not request.session_id or request.session_id not in chat_sessions:
            session_id = str(uuid.uuid4())
            title = request.message[:30] + ("..." if len(request.message) > 30 else "")
            chat_sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "title": title,
                "messages": []
            }
        else:
            session_id = request.session_id
        
        # Foydalanuvchi xabarini tarixga qo'shish
        user_message = {
            "text": request.message,
            "is_user": True,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[session_id]["messages"].append(user_message)
        
        # AI javobini olish
        ai_response = await generate_ai_response(request.message, session_id)
        
        # AI javobini tarixga qo'shish
        ai_message = {
            "text": ai_response,
            "is_user": False,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[session_id]["messages"].append(ai_message)
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Xatolik yuz berdi: {str(e)}")

async def generate_ai_response(message: str, session_id: str) -> str:
    """OpenAI yordamida AI javobini generatsiya qilish"""
    try:
        # Sessiya tarixini olish
        session = chat_sessions.get(session_id, {})
        messages_history = session.get("messages", [])
        
        # OpenAI uchun konvertatsiya qilish
        messages_for_openai = [
            {"role": "system", "content": "Siz Nisme AI siz. Foydalanuvchi bilan do'stona va professional muloqot qiling."}
        ]
        
        # So'nggi 6 ta xabarni kontekst uchun yuborish
        for msg in messages_history[-6:]:
            role = "user" if msg["is_user"] else "assistant"
            messages_for_openai.append({"role": role, "content": msg["text"]})
        
        # OpenAI API chaqiruvi
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages_for_openai,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except openai.error.OpenAIError as e:
        return f"AI xizmatida xatolik: {str(e)}"
    except Exception as e:
        return f"Kutilmagan xatolik: {str(e)}"

@app.get("/api/sessions", response_model=List[ChatSession])
async def get_all_sessions():
    """Barcha chat sessiyalarini qaytaradi"""
    return [
        ChatSession(
            session_id=session_id,
            created_at=session["created_at"],
            title=session["title"],
            messages=session["messages"]
        )
        for session_id, session in chat_sessions.items()
    ]

@app.get("/api/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """ID bo'yicha bitta sessiyani qaytaradi"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Sessiya topilmadi")
    return ChatSession(
        session_id=session_id,
        created_at=chat_sessions[session_id]["created_at"],
        title=chat_sessions[session_id]["title"],
        messages=chat_sessions[session_id]["messages"]
    )

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Sessiyani o'chiradi"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"message": "Sessiya o'chirildi"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)