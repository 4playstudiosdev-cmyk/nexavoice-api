import asyncio
import json
import uuid
import os
import base64
import requests
import tempfile
import io
from aiohttp import web
import aiohttp
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS
import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
load_dotenv()

# Cloud Deployment settings
PORT = int(os.environ.get("PORT", 8080))

# API Keys
GROQ_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "ON8VbOMsYaufJcoM")

client = Groq(api_key=GROQ_KEY)

active_agents = {}

def tts_to_base64(text):
    """ElevenLabs TTS for Urdu (Premium Voice)"""
    if not ELEVEN_KEY:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
    headers = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text, 
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.50, "similarity_boost": 0.75}
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode("utf-8")
    except:
        pass
    return None

def english_tts_to_base64(text):
    """Free Google TTS for English (Fast & Free)"""
    try:
        tts = gTTS(text=text, lang='en', tld='com')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode("utf-8")
    except:
        return None

def transcribe_audio(audio_bytes, mime_type="audio/webm"):
    """Whisper STT (Fast Transcription)"""
    ext = "webm" if "webm" in mime_type else "wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(f"audio.{ext}", f, mime_type),
                model="whisper-large-v3",
                response_format="text"
            )
        os.unlink(tmp_path)
        return result.strip() if isinstance(result, str) else result.text.strip()
    except:
        return None

def to_roman_urdu(text):
    """Strict Roman Urdu Conversion"""
    prompt = f"""Convert to Roman Urdu. Strictly NO Urdu script.
    - Keep it short.
    - Use Pakistani style (Assalam o Alaikum, Shukriya).
    Text: "{text}" """
    try:
        r = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.1-8b-instant", max_tokens=60, temperature=0.1)
        return r.choices[0].message.content.strip()
    except:
        return text

async def create_agent(request):
    try:
        data = await request.json()
        agent_id = str(uuid.uuid4())
        active_agents[agent_id] = {
            "name": data.get('company_name', 'Company'),
            "type": data.get('use_case', 'customer_care'),
            "knowledge": data.get('knowledge_base', '')
        }
        return web.json_response({"status": "success", "agent_id": agent_id})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def websocket_handler(request):
    agent_id = request.match_info.get('agent_id')
    if agent_id not in active_agents:
        return web.Response(status=404, text="Agent not found")

    agent_data = active_agents[agent_id]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # State management for current call
    session = {
        "stage": "language_selection", 
        "language": None, 
        "history": [],
        "processing": False # To handle interrupts
    }

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                # Signal to stop current speech (Interrupt Logic)
                if data.get("type") == "interrupt":
                    session["processing"] = False
                    print("[Interrupt] User started speaking, stopping AI...")
                    continue

                if data.get("type") == "start_call":
                    greeting = f"Assalam o Alaikum! Welcome to {agent_data['name']}. Please select English or Urdu."
                    audio = tts_to_base64(greeting)
                    await ws.send_json({"type": "agent_message", "text": greeting, "audio": audio})
                
                elif data.get("type") == "audio_chunk":
                    session["processing"] = True
                    raw_user_text = transcribe_audio(base64.b64decode(data.get("audio", "")))
                    
                    if not raw_user_text or not session["processing"]: 
                        continue
                        
                    await ws.send_json({"type": "user_message", "text": raw_user_text})
                    
                    if session["stage"] == "language_selection":
                        prompt = f"Detect language: '{raw_user_text}'. Reply ONLY 'English' or 'Urdu'."
                        r = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant", temperature=0)
                        detected_lang = r.choices[0].message.content.strip().lower()
                        
                        if "english" in detected_lang:
                            session["language"] = "English"
                            base_prompt = f"""Professional AI for {agent_data['name']}. 
                            RULES: 
                            - Reply ONLY in English. 
                            - Short answers only (10-15 words max). 
                            - DO NOT read full descriptions of menu items. Just mention 'Item Name' and 'Price'.
                            - Be extremely fast.
                            Knowledge: {agent_data['knowledge']}"""
                            reply = "English selected. How can I help you today?"
                            audio = english_tts_to_base64(reply)
                        else:
                            session["language"] = "Urdu"
                            base_prompt = f"""Tu {agent_data['name']} ka AI assistant hai. 
                            RULES: 
                            - Strictly Roman Urdu. No Urdu script.
                            - Pakistani Islamic greetings. 
                            - Menu items ki lambi details NA PARHO. Sirf Item ka naam aur price batao.
                            - Bohat mukhtasir (short) jawab do.
                            Knowledge: {agent_data['knowledge']}"""
                            reply = "Theek hai, Urdu mein baat karte hain. Batayye main kya madad karoon?"
                            audio = tts_to_base64(reply)
                        
                        session["history"].append({"role": "system", "content": base_prompt})
                        session["history"].append({"role": "assistant", "content": reply})
                        session["stage"] = "conversation"
                        await ws.send_json({"type": "agent_message", "text": reply, "audio": audio})

                    elif session["stage"] == "conversation":
                        llm_input = to_roman_urdu(raw_user_text) if session["language"] == "Urdu" else raw_user_text
                        session["history"].append({"role": "user", "content": llm_input})
                        
                        # Fetch response from Groq (Fast Model)
                        r = client.chat.completions.create(
                            messages=session["history"], 
                            model="llama-3.1-8b-instant",
                            max_tokens=80, # Keep it short for speed
                            temperature=0.6
                        )
                        reply = r.choices[0].message.content.strip()
                        
                        if not session["processing"]: continue # Check interrupt again

                        session["history"].append({"role": "assistant", "content": reply})
                        
                        # Generate Audio based on selected language
                        audio = english_tts_to_base64(reply) if session["language"] == "English" else tts_to_base64(reply)
                        await ws.send_json({"type": "agent_message", "text": reply, "audio": audio})
    finally:
        pass
    return ws

app = web.Application()
import aiohttp_cors
cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
cors.add(app.router.add_post('/api/create_agent', create_agent))
app.router.add_get('/ws/{agent_id}', websocket_handler)

if __name__ == '__main__':
    print(f"🚀 NexaVoice Backend Live on port {PORT}")
    web.run_app(app, host='0.0.0.0', port=PORT)