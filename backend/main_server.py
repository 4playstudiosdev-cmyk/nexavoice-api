# -*- coding: utf-8 -*-
import asyncio, json, uuid, os, base64, io, time, re, tempfile, pathlib
import warnings, logging, datetime, httpx
warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
from aiohttp import web
import aiohttp, aiohttp_cors
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

PORT         = int(os.environ.get("PORT", 8080))
GROQ_KEY     = os.getenv("GROQ_API_KEY")
ELEVEN_KEY   = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
SMTP_USER    = os.getenv("SMTP_USER")
SMTP_PASS    = os.getenv("SMTP_PASS")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
ELEVEN_VOICE_UR = os.getenv("ELEVENLABS_URDU_VOICE_ID",   "EXAVITQu4vr4xnSDxMaL")
ELEVEN_VOICE_AR = os.getenv("ELEVENLABS_ARABIC_VOICE_ID", "IKne3meq5aSn9XLyUdCD")
client        = Groq(api_key=GROQ_KEY)
active_agents = {}
# Railway: /app/backend/main_server.py -> frontend is ../frontend
# Local:   D:/Awaz_AI_Project/backend  -> frontend is ../frontend
FRONTEND_DIR  = str(pathlib.Path(__file__).parent.parent / "frontend")
LANG_CODE     = {"English": "en", "Urdu": "ur", "Arabic": "ar"}

# ── STT ──────────────────────────────────────────────────────
print("Loading STT (GPU if available, Groq API fallback)...")
try:
    from faster_whisper import WhisperModel
    stt_model = WhisperModel("small", device="cuda", compute_type="float16")
    print("STT small GPU ready!")
except Exception as e:
    stt_model = None
    print(f"STT Groq fallback: {e}")

def transcribe(audio_bytes, lang_code=None):
    t = time.time()
    if stt_model:
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes); path = f.name
            segs, info = stt_model.transcribe(
                path, beam_size=1, language=lang_code,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=200),
                condition_on_previous_text=False)
            os.unlink(path)
            text = " ".join(s.text for s in segs).strip()
            detected = getattr(info, "language", lang_code or "auto")
            print(f"STT {int((time.time()-t)*1000)}ms [{detected}] -> '{text[:60]}'")
            return text or None
        except Exception as e:
            print(f"STT GPU err: {e}")
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes); path = f.name
        fh = open(path, "rb")
        kw = {"file": ("audio.webm", fh, "audio/webm"),
              "model": "whisper-large-v3", "response_format": "text"}
        if lang_code:
            kw["language"] = lang_code
        r = client.audio.transcriptions.create(**kw)
        fh.close(); os.unlink(path)
        text = (r if isinstance(r, str) else r.text).strip()
        print(f"STT {int((time.time()-t)*1000)}ms [Groq] -> '{text[:60]}'")
        return text or None
    except Exception as e:
        print(f"STT Groq err: {e}"); return None

# ── TTS ──────────────────────────────────────────────────────
try:
    import edge_tts
    HAS_EDGE = True
    print("TTS edge-tts ready!")
except Exception:
    HAS_EDGE = False
    print("pip install edge-tts")

async def _edge(text, voice, lbl):
    if not HAS_EDGE or not text.strip():
        return None
    # Retry up to 3 times — Railway network can be flaky to Microsoft servers
    for attempt in range(3):
        try:
            t = time.time()
            comm = edge_tts.Communicate(text.strip(), voice=voice, rate="+15%")
            buf = io.BytesIO()
            async for c in comm.stream():
                if c["type"] == "audio":
                    buf.write(c["data"])
            buf.seek(0); data = buf.read()
            if not data:
                print(f"TTS-{lbl} edge-tts returned empty audio (attempt {attempt+1})")
                await asyncio.sleep(0.2)
                continue
            b64 = base64.b64encode(data).decode()
            print(f"TTS-{lbl} edge-tts OK {int((time.time()-t)*1000)}ms bytes={len(data)} b64_start={b64[:6]}")
            return b64
        except Exception as e:
            print(f"TTS-{lbl} edge-tts attempt {attempt+1} failed: {e}")
            await asyncio.sleep(0.3)
    print(f"TTS-{lbl} edge-tts FAILED after 3 attempts")
    return None

async def _eleven(text, voice_id, lbl):
    if not ELEVEN_KEY or not text.strip():
        return None
    try:
        t = time.time()
        async with httpx.AsyncClient(timeout=15) as h:
            r = await h.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                json={"text": text.strip(), "model_id": "eleven_flash_v2_5",
                      "voice_settings": {"stability": 0.5, "similarity_boost": 0.7,
                                         "style": 0.0, "use_speaker_boost": False}},
                headers={"xi-api-key": ELEVEN_KEY,
                         "Content-Type": "application/json", "Accept": "audio/mpeg"})
        if r.status_code == 200:
            print(f"TTS-{lbl} {int((time.time()-t)*1000)}ms ElevenLabs Flash")
            return base64.b64encode(r.content).decode()
        # fallback to multilingual
        async with httpx.AsyncClient(timeout=15) as h:
            r = await h.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json={"text": text.strip(), "model_id": "eleven_multilingual_v2",
                      "voice_settings": {"stability": 0.55, "similarity_boost": 0.75,
                                         "style": 0.0, "use_speaker_boost": False}},
                headers={"xi-api-key": ELEVEN_KEY,
                         "Content-Type": "application/json", "Accept": "audio/mpeg"})
        if r.status_code == 200:
            print(f"TTS-{lbl} {int((time.time()-t)*1000)}ms ElevenLabs v2")
            return base64.b64encode(r.content).decode()
        print(f"ElevenLabs {r.status_code}: {r.text[:100]}")
        return None
    except Exception as e:
        print(f"TTS-{lbl} err: {e}"); return None

async def speak(text, language="English"):
    if not text or not text.strip():
        return None
    if language == "Urdu":
        return await _eleven(text, ELEVEN_VOICE_UR, "UR") or await _edge(text, "ur-PK-UzmaNeural", "UR-fb")
    if language == "Arabic":
        return await _eleven(text, ELEVEN_VOICE_AR, "AR") or await _edge(text, "ar-SA-HamedNeural", "AR-fb")
    return await _edge(text, "en-US-GuyNeural", "EN")

# ── SUPABASE ─────────────────────────────────────────────────
async def db_insert(table, data):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as h:
            r = await h.post(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers={"apikey": SUPABASE_KEY,
                         "Authorization": f"Bearer {SUPABASE_KEY}",
                         "Content-Type": "application/json",
                         "Prefer": "return=representation"},
                content=json.dumps(data))
        if r.status_code in (200, 201):
            rows = r.json()
            row = rows[0] if isinstance(rows, list) and rows else rows
            print(f"DB insert id={str(row.get('id','?'))[:8]}")
            return row
        print(f"DB insert error {r.status_code}: {r.text[:150]}")
        return None
    except Exception as e:
        print(f"DB error: {e}"); return None

async def db_update(table, row_id, data):
    if not SUPABASE_URL or not SUPABASE_KEY or not row_id:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as h:
            r = await h.patch(
                f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}",
                headers={"apikey": SUPABASE_KEY,
                         "Authorization": f"Bearer {SUPABASE_KEY}",
                         "Content-Type": "application/json"},
                content=json.dumps(data))
        if r.status_code in (200, 204):
            print(f"DB update id={row_id[:8]}")
        else:
            print(f"DB update error {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"DB update error: {e}")

# ── SUMMARIZER ───────────────────────────────────────────────
EXT = {
    "restaurant": '"order_type":"","customer_name":"","phone":"","address":"","items":[],"payment_method":"","total":0,"delivery_time":""',
    "kelectric":  '"intent":"","account_number":"","phone":"","complaint_id":"","area":"","issue":"","status":"","resolution_time":""',
    "loanapp":    '"intent":"","customer_name":"","loan_amount":0,"due_date":"","status":"","response":"","risk_level":"medium","payment_plan":false',
    "telecom":    '"intent":"","phone":"","package_type":"","package_name":"","status":"","amount":0,"sim_action":""',
    "sales":      '"intent":"sales_call","customer_name":"","product":"","interest_level":"medium","objection":"","final_status":"","amount":0',
    "custom":     '"main_topic":"","action_taken":"","follow_up_needed":false',
}

async def summarize_call(agent_type, agent_name, transcript, language):
    txt = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in transcript if m.get("role") != "system")
    if not txt.strip():
        return _empty_summary(language, 0)
    turns = len([m for m in transcript if m.get("role") == "user"])
    ext   = EXT.get(agent_type, EXT["custom"])
    prompt = (
        f"Analyze this {agent_name} call. Return ONLY valid JSON, no markdown.\n"
        f"TRANSCRIPT:\n{txt[:2500]}\n"
        f'Return exactly: {{"summary":"2-3 sentence summary","sentiment":"positive",'
        f'"resolved":true,"key_actions":["action"],"extracted_data":{{{ext}}}}}\n'
        f"sentiment=positive|neutral|negative resolved=true|false"
    )
    try:
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", max_tokens=600, temperature=0.0)
        raw = r.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        parsed = json.loads(raw)
        parsed["language_used"]  = language
        parsed["duration_turns"] = turns
        return parsed
    except Exception as e:
        print(f"Summarizer: {e}")
        return _empty_summary(language, turns)

def _empty_summary(lang, turns):
    return {"summary": "Call completed.", "sentiment": "neutral",
            "language_used": lang, "duration_turns": turns,
            "resolved": False, "key_actions": [], "extracted_data": {}}

# ── EMAIL ────────────────────────────────────────────────────
async def send_email(agent_name, agent_type, summary, call_id):
    if not all([SMTP_USER, SMTP_PASS, NOTIFY_EMAIL]):
        print("Email skipped — set SMTP vars"); return
    # Quick credential check before building HTML
    import smtplib as _test
    try:
        with _test.SMTP(SMTP_HOST, SMTP_PORT) as _s:
            _s.ehlo(); _s.starttls(); _s.ehlo(); _s.login(SMTP_USER, SMTP_PASS)
    except Exception as cred_err:
        print(f"Email disabled — bad credentials: {cred_err}")
        print("Fix: Gmail -> Google Account -> Security -> App Passwords -> create one")
        return
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    se  = {"positive": "Good", "neutral": "Neutral", "negative": "Poor"}.get(
          summary.get("sentiment", "neutral"), "Neutral")
    rt  = "Resolved" if summary.get("resolved") else "Unresolved"
    lf  = {"English": "EN", "Urdu": "UR", "Arabic": "AR"}.get(
          summary.get("language_used", "English"), "EN")
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    ext = summary.get("extracted_data", {}) or {}
    rows = "".join(
        f"<tr><td style='padding:6px 12px;color:#888'>{k.replace('_',' ').title()}</td>"
        f"<td style='padding:6px 12px;font-weight:600;color:#eee'>"
        f"{', '.join(str(i) for i in v) if isinstance(v, list) else str(v)}</td></tr>"
        for k, v in ext.items() if v not in (None, "", [], False, 0))
    acts = "".join(f"<li style='color:#ccc;margin-bottom:4px'>{a}</li>"
                   for a in (summary.get("key_actions") or []))
    html = (
        "<html><body style='margin:0;background:#0A0A0F;font-family:sans-serif'>"
        "<div style='max-width:560px;margin:0 auto;padding:24px 16px'>"
        "<div style='background:#111;border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:12px'>"
        "<div style='font-size:10px;color:#4F8EF7;letter-spacing:2px;font-weight:700;margin-bottom:6px'>NEXAVOICE CALL REPORT</div>"
        f"<div style='font-size:20px;font-weight:800;color:#fff;margin-bottom:3px'>{agent_name}</div>"
        f"<div style='font-size:11px;color:#555'>{now} - ID:{str(call_id)[:8]}</div></div>"
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px'>"
        + "".join(
            f"<div style='background:#111;border:1px solid #222;border-radius:10px;padding:12px;text-align:center'>"
            f"<div style='font-size:9px;color:#555;margin-bottom:4px;text-transform:uppercase'>{lbl}</div>"
            f"<div style='font-size:13px;font-weight:700;color:{col}'>{val}</div></div>"
            for lbl, col, val in [
                ("Sentiment", "#fff", se),
                ("Turns", "#4F8EF7", str(summary.get("duration_turns", 0))),
                ("Language", "#fff", lf),
                ("Status", "#22D07A" if summary.get("resolved") else "#F5A623", rt)])
        + "</div>"
        + f"<div style='background:#111;border:1px solid #222;border-radius:10px;padding:16px;margin-bottom:10px'>"
          f"<div style='font-size:9px;color:#4F8EF7;letter-spacing:2px;font-weight:700;margin-bottom:8px'>SUMMARY</div>"
          f"<div style='color:#ccc;font-size:13px;line-height:1.7'>{summary.get('summary', 'No summary.')}</div></div>"
        + (f"<div style='background:#111;border:1px solid #222;border-radius:10px;padding:16px;margin-bottom:10px'>"
           f"<div style='font-size:9px;color:#22D07A;letter-spacing:2px;font-weight:700;margin-bottom:8px'>KEY ACTIONS</div>"
           f"<ul style='margin:0;padding-left:16px'>{acts}</ul></div>" if acts else "")
        + (f"<div style='background:#111;border:1px solid #222;border-radius:10px;padding:16px;margin-bottom:10px'>"
           f"<div style='font-size:9px;color:#F5A623;letter-spacing:2px;font-weight:700;margin-bottom:8px'>CALL DATA</div>"
           f"<table style='width:100%;border-collapse:collapse'>{rows}</table></div>" if rows else "")
        + "<div style='text-align:center;color:#333;font-size:10px;margin-top:10px'>NexaVoice AI</div>"
          "</div></body></html>"
    )
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Call Report - {agent_name} - {rt} [{lf}]"
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_EMAIL
        msg.attach(MIMEText(html, "html"))
        loop = asyncio.get_event_loop()
        def _send():
            import smtplib
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
                srv.ehlo(); srv.starttls(); srv.ehlo()
                srv.login(SMTP_USER, SMTP_PASS)
                srv.send_message(msg)
        await loop.run_in_executor(None, _send)
        print(f"Email sent to {NOTIFY_EMAIL}")
    except Exception as e:
        print(f"Email error: {e}")

# ── AGENT SCRIPTS ────────────────────────────────────────────
# All strings use normal quotes — no triple-quote/dict-key conflicts

_REST_EN = (
    "You are a restaurant voice agent. Be fast. Listen exactly.\n"
    "NEVER substitute items. If customer says X, add X exactly as said.\n"
    "FLOW:\n"
    "1) Ask: Delivery or pickup?\n"
    "2) Ask: What would you like to order?\n"
    "3) Add each item exactly. After each: Anything else?\n"
    "4) When done ordering, read back full order. Ask: Is that correct?\n"
    "5) If delivery: ask address. If pickup: ask time.\n"
    "6) Payment is cash only. Confirm: Paying cash on delivery?\n"
    "7) Ask name and phone number.\n"
    "8) Say: Order confirmed! Arrives in 30-45 mins. Thank you.\n"
    "RULES: MAX 12 words per reply. One question only. Never guess items."
)
_REST_UR = (
    "Tu restaurant ka voice agent hai. SIRF Roman Urdu.\n"
    "Jo customer bole EXACTLY wohi add karo. Kabhi substitute mat karo.\n"
    "FLOW:\n"
    "1) Delivery ya pickup?\n"
    "2) Kya order karein ge?\n"
    "3) Har item exactly add karo. Har baar: Kuch aur?\n"
    "4) Jab ho jaye: poora order repeat. Theek hai?\n"
    "5) Delivery=address lo. Pickup=time lo.\n"
    "6) Payment sirf cash hai. Cash on delivery confirm karo.\n"
    "7) Naam aur number lo.\n"
    "8) Order confirm! 30-45 min mein aayega. Shukriya.\n"
    "RULES: MAX 10 alfaaz. Ek sawaal. Substitute kabhi nahi."
)
_REST_AR = (
    "Anta wakeel mataem sawti. Redd bialArabiyyah faqat.\n"
    "Sujjil TAMAMAN ma yaqul alzibon. La tabdil abadan.\n"
    "FLOW:\n"
    "1) Tawseel am istilam?\n"
    "2) Maza tureed?\n"
    "3) Sujjil kull binad TAMAMAN. Bad kull: Hal tureed shayan akhar?\n"
    "4) Karrir altalab. Hal sahih?\n"
    "5) Tawseel=unwan. Istilam=waqt.\n"
    "6) Aldaf naqdan faqat. Takkid.\n"
    "7) Alisme + alhatif.\n"
    "8) Tamma altalab! 30-45 daqiqah. Shukran.\n"
    "RULES: MAX 12 kalmah. Sual wahid. La tabdil."
)

_KE_EN = (
    "You are K-Electric AI agent. Be concise and fast.\n"
    "INTENTS: bill inquiry, power outage, complaint, new connection, load shedding.\n"
    "OUTAGE: get account or meter or phone, give complaint ID KE-XXXXXX, ETA 4-6hrs.\n"
    "BILL: get account, give amount and due date, payment options Easypaisa JazzCash bank.\n"
    "COMPLAINT: describe issue, get account, give ID KE-XXXXXX, team will call.\n"
    "NEW CONNECTION: need CNIC and property docs, visit nearest center.\n"
    "LOAD SHEDDING: get area, give schedule.\n"
    "RULES: MAX 15 words. Always give complaint ID. Angry: I understand, resolving now."
)
_KE_UR = (
    "Tu K-Electric ka AI agent hai. SIRF Roman Urdu mein jawab de.\n"
    "OUTAGE: account ya phone lo, ID KE-XXXXXX do, 4-6 ghante.\n"
    "BILL: account lo, amount aur due date batao, payment options.\n"
    "COMPLAINT: masla lo, account lo, ID KE-XXXXXX do, team contact karegi.\n"
    "LOAD SHEDDING: area lo, schedule batao.\n"
    "RULES: MAX 12 alfaaz. Roman Urdu ONLY. ID hamesha do."
)
_KE_AR = (
    "Anta mosaaid K-Electric dhaki. Redd bialArabiyyah faqat.\n"
    "Inqita: hesab aw hatif, KE-XXXXXX, 4-6 saat.\n"
    "Fatorah: hesab, almablagh watarikh alhiqaq, aldaf.\n"
    "Shakwa: wasf almasalah, hesab, raqam shakwa KE-XXXXXX, alfareq saitasil.\n"
    "RULES: MAX 15 kalmah. Ati raqam alshakwa daiman."
)

_LOAN_EN = (
    "You are Kamran CashEasy ethical AI agent. Be fast and direct.\n"
    "VERIFICATION: confirm CNIC or DOB, employment and income, loan confirmation, under review.\n"
    "APPROVAL: congratulate, amount and transfer time, repayment date, late fee warning.\n"
    "REMINDER soft: due date, avoid late fees.\n"
    "REMINDER strong: overdue, pay now, penalties.\n"
    "Cannot pay: offer extension or payment plan.\n"
    "Did not take loan: register dispute and escalate.\n"
    "SUPPORT: App Pay Now Easypaisa JazzCash bank. Complaint CE-XXXXXX.\n"
    "RULES: MAX 15 words. NEVER threaten. Verify identity FIRST. Payment plan before escalating."
)
_LOAN_UR = (
    "Tu Kamran hai, CashEasy ka ethical agent. SIRF Roman Urdu mein jawab de.\n"
    "Verification: CNIC confirm karo, income lo, loan confirm karo.\n"
    "Reminder: due date aur late charges batao.\n"
    "Nahi de sakta: extension ya plan offer karo.\n"
    "Loan nahi liya: dispute register karo.\n"
    "Support: EasyPaisa JazzCash, complaint CE-XXXXXX.\n"
    "RULES: MAX 12 alfaaz. Roman Urdu ONLY. Kabhi threat nahi. Identity pehle verify karo."
)
_LOAN_AR = (
    "Anta Kamran wakeel CashEasy alakhlaqi. Redd bialArabiyyah faqat.\n"
    "Tahaqquq: CNIC aw DOB, dakhil, taakkid alqard.\n"
    "Tadhkeer: mawid alistihqaq, rasm altaakhir.\n"
    "La yastatioo aldaf: khuttah sidad.\n"
    "Lam yakhudh qard: tasjeel niza.\n"
    "RULES: MAX 15 kalmah. La tahdeed. Tahaqquq awwalan."
)

_TEL_EN = (
    "You are a telecom AI agent for Jazz Telenor Ufone Zong. Be fast and direct.\n"
    "BALANCE: get mobile number, give balance, upsell package.\n"
    "PACKAGE: daily 1GB Rs50, weekly 5GB Rs200, monthly 20GB Rs600, activate, SMS confirm.\n"
    "SIM LOST: get CNIC, block SIM, nearest franchise.\n"
    "COMPLAINT: get area, register, 24-48hrs resolution.\n"
    "UNSUBSCRIBE: confirm, deactivate.\n"
    "RECHARGE: Easypaisa JazzCash scratch card.\n"
    "After EVERY action: Would you like a bundle to save more.\n"
    "RULES: MAX 15 words. One question at a time."
)
_TEL_UR = (
    "Tu Pakistani telecom AI agent hai. SIRF Roman Urdu mein jawab de.\n"
    "Balance: number lo, upsell karo.\n"
    "Package: daily weekly monthly, confirm karo.\n"
    "SIM lost: CNIC lo, block karo, franchise.\n"
    "Complaint: area lo, 24-48 ghante.\n"
    "Recharge: EasyPaisa JazzCash.\n"
    "RULES: MAX 12 alfaaz. Roman Urdu ONLY. Har kaam ke baad upsell karo."
)
_TEL_AR = (
    "Anta wakeel sharikat ittisalat. Redd bialArabiyyah faqat.\n"
    "Raseed: raqam alhatif, iqtirah baqah.\n"
    "Baqah: yawmi weekly shahri, tafil.\n"
    "Shari'ah mafqudah: CNIC, hajb, aqrab far.\n"
    "Shakwa: almintaqah, 24-48 saa.\n"
    "RULES: MAX 15 kalmah. Iqtirah baqah bad kull talab."
)

_SALES_EN = (
    "You are an elite AI sales agent. Be punchy and fast.\n"
    "FLOW: 1 Get permission 30 seconds. 2 Identify pain point. "
    "3 Pitch solution and benefit simply. 4 Price and urgency today only. "
    "5 Close - Shall I activate this right now.\n"
    "OBJECTIONS: Not interested - Price or timing. "
    "Expensive - Saves you more overall. "
    "Think about it - Offer ends today reserve it. "
    "Angry - I will be brief this genuinely helps.\n"
    "RULES: MAX 15 words. Never quit on first objection. Follow up: When should I call back."
)
_SALES_UR = (
    "Tu elite AI sales agent hai. SIRF Roman Urdu mein jawab de. Fast aur punchy.\n"
    "FLOW: 1 Permission lo 30 second. 2 Pain identify karo. "
    "3 Solution aur benefit batao. 4 Price aur urgency. 5 Close - Abhi activate kar doon.\n"
    "OBJECTIONS: Nahi chahiye - Price ya timing. "
    "Mehnga - Isliye save hoga. "
    "Sochta hoon - Aaj tak hai offer.\n"
    "RULES: MAX 12 alfaaz. Roman Urdu ONLY. Pehli objection pe mat chhodna."
)
_SALES_AR = (
    "Anta wakeel mabi'at mutaqaddim. Redd bialArabiyyah faqat. Kun sari'an.\n"
    "1 Idhn 30 thaniyah. 2 Almushkilah. 3 Alhal walfaidah. "
    "4 Alsi'r walilhah. 5 Alghalq - Hal ofa'iloh alaan.\n"
    "Itiradat: Ghayr muhtamm - Alsi'r am alwaqt. "
    "Ghali - Yuwaffir akhar. "
    "Sa'ufakkir - Alard yantahi alyawm.\n"
    "RULES: MAX 15 kalmah. La tastaslem ind awwal itirad."
)

_CUSTOM_EN = "You are a professional AI voice assistant. MAX 15 words per reply. Use the knowledge base. Unknown: I will check and get back to you."
_CUSTOM_UR = "Tu professional AI assistant hai. SIRF Roman Urdu. MAX 12 alfaaz. Pata nahi: Check karke batata hoon."
_CUSTOM_AR = "Anta mosa'id sawti dhaki. Redd bialArabiyyah. MAX 15 kalmah. Majhool: Satahaqqaq wa a'ood ilayk."

SCRIPTS = {
    "restaurant": {
        "name": "Restaurant Order Agent", "color": "#FF6B35", "icon": "🍔",
        "description": "Orders, delivery & pickup",
        "en_greeting": "Hello! Thank you for calling. Would you like delivery or pickup today?",
        "ur_greeting": "Assalam o Alaikum! Shukriya call karne ka. Delivery chahiye ya pickup?",
        "ar_greeting": "Ahlan! Shukran littisalak. Tawseel am istilam?",
        "en_system": _REST_EN, "ur_system": _REST_UR, "ar_system": _REST_AR,
    },
    "kelectric": {
        "name": "K-Electric", "color": "#0066CC", "icon": "⚡",
        "description": "Power, billing & complaints",
        "en_greeting": "Assalam-o-Alaikum! Thank you for calling K-Electric. How may I assist you?",
        "ur_greeting": "Assalam o Alaikum! K-Electric mein shukriya. Kya masla hai?",
        "ar_greeting": "Assalamu alaikum! Shukran littisalak bil K-Electric. Kayfa asaaduk?",
        "en_system": _KE_EN, "ur_system": _KE_UR, "ar_system": _KE_AR,
    },
    "loanapp": {
        "name": "CashEasy Loan App", "color": "#00A651", "icon": "💰",
        "description": "Recovery, verification & support",
        "en_greeting": "Assalam-o-Alaikum! This is CashEasy. Am I speaking with the account holder?",
        "ur_greeting": "Assalam o Alaikum! CashEasy se bol raha hoon. Account holder hain aap?",
        "ar_greeting": "Assalamu alaikum! Ana Kamran min CashEasy. Hal atakallam ma sahib alhisab?",
        "en_system": _LOAN_EN, "ur_system": _LOAN_UR, "ar_system": _LOAN_AR,
    },
    "telecom": {
        "name": "Telecom Support", "color": "#E91E8C", "icon": "📡",
        "description": "Balance, packages, SIM & complaints",
        "en_greeting": "Assalam-o-Alaikum! Thank you for calling. How may I help you?",
        "ur_greeting": "Assalam o Alaikum! Kya madad kar sakta hoon?",
        "ar_greeting": "Assalamu alaikum! Shukran littisalak. Kayfa asaaduk?",
        "en_system": _TEL_EN, "ur_system": _TEL_UR, "ar_system": _TEL_AR,
    },
    "sales": {
        "name": "AI Sales Agent", "color": "#FFD700", "icon": "🧠",
        "description": "Human-like conversion & upselling",
        "en_greeting": "Assalam-o-Alaikum! I will be very quick. I have something that can genuinely help you. Is this a good time for 30 seconds?",
        "ur_greeting": "Assalam o Alaikum! Bilkul quick hoon. Aapke liye kuch useful hai. 30 second ka waqt hai?",
        "ar_greeting": "Assalamu alaikum! Sa'akun sari'an jiddan. Ladayya shay mufeed lak. Hal ladayk 30 thaniyah?",
        "en_system": _SALES_EN, "ur_system": _SALES_UR, "ar_system": _SALES_AR,
    },
    "custom": {
        "name": "Custom Agent", "color": "#7C3AED", "icon": "🤖",
        "description": "Your own knowledge base",
        "en_greeting": "Hello! How can I assist you today?",
        "ur_greeting": "Assalam o Alaikum! Kya madad kar sakta hoon?",
        "ar_greeting": "Marhaban! Kayfa yumkinuni musa'adatak alyawm?",
        "en_system": _CUSTOM_EN, "ur_system": _CUSTOM_UR, "ar_system": _CUSTOM_AR,
    },
}

# ── PIPELINE ─────────────────────────────────────────────────
async def pipeline(ws, session, audio_bytes):
    loop = asyncio.get_event_loop()
    t0   = time.time()
    lc   = LANG_CODE.get(session.get("language")) if session["stage"] == "conversation" else None
    text = await loop.run_in_executor(None, transcribe, audio_bytes, lc)
    if not text or not session["processing"]:
        return
    await ws.send_json({"type": "user_message", "text": text})
    if session["stage"] == "language_selection":
        await handle_language(ws, session, text)
        return

    lang    = session["language"]
    history = session["history"]
    history.append({"role": "user", "content": text})

    # Keep only last 6 turns — minimum context = maximum speed
    if len(history) > 7:
        history[1:] = history[-6:]

    t_llm = time.time(); buf = ""; full = ""; first = True
    split = re.compile(r'(?<=[.!?])\s+|(?<=[,])\s+')

    # Groq production models — April 2026 (auto-fallback on any API error)
    # openai/gpt-oss-20b     = 1000 t/sec  fastest
    # llama-3.1-8b-instant   =  560 t/sec  reliable free tier
    # llama-3.3-70b-versatile =  280 t/sec  most capable
    models = ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

    for model in models:
        try:
            stream = client.chat.completions.create(
                messages=history, model=model,
                max_tokens=40,        # 40 tokens = ~30 words, enough for short voice replies
                temperature=0.1,      # near-zero = instant deterministic output
                stream=True)
            for chunk in stream:
                if not session["processing"]: break
                d = chunk.choices[0].delta.content
                if not d: continue
                buf += d; full += d
                parts = split.split(buf)
                if len(parts) > 1:
                    to_speak = " ".join(parts[:-1]).strip(); buf = parts[-1]
                    if to_speak:
                        if first:
                            print(f"LLM first {int((time.time()-t_llm)*1000)}ms [{model}]")
                        audio = await speak(to_speak, lang)
                        if audio and session["processing"]:
                            await ws.send_json({"type": "agent_message", "text": to_speak,
                                                "audio": audio, "is_first": first})
                            if first:
                                print(f"First audio {int((time.time()-t0)*1000)}ms")
                                first = False
            if buf.strip() and session["processing"]:
                audio = await speak(buf.strip(), lang)
                await ws.send_json({"type": "agent_message", "text": buf.strip(),
                                    "audio": audio, "is_first": first})
            history.append({"role": "assistant", "content": full.strip()})
            print(f"Done {int((time.time()-t0)*1000)}ms")
            break  # success — exit model loop

        except Exception as e:
            err = str(e).lower()
            # Try next model for: rate limits, decommissioned, 400, 404, any API error
            skip_codes = ["rate", "429", "limit", "decommission", "deprecated",
                          "not found", "invalid_request", "400", "404", "model"]
            if any(code in err for code in skip_codes):
                print(f"Model {model} unavailable ({str(e)[:60]}), trying next...")
                await asyncio.sleep(0.05)
                buf = ""; full = ""; first = True
                continue
            print(f"Pipeline err [{model}]: {e}")
            break
    else:
        # All models exhausted — send fallback so frontend re-enables mic
        fallback = "Sorry, I had a connection issue. Please repeat that."
        audio = await speak(fallback, lang)
        await ws.send_json({"type": "agent_message", "text": fallback,
                            "audio": audio, "is_first": True})
        history.append({"role": "assistant", "content": fallback})

async def handle_language(ws, session, text):
    try:
        r = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": f"Classify language. Reply ONLY one word: English, Urdu, or Arabic.\nText: {text}"}],
            model="llama-3.1-8b-instant", max_tokens=3, temperature=0)
        lang = r.choices[0].message.content.strip().lower()
    except Exception:
        lang = "english"

    s  = session["script"]
    kb = session["knowledge"]
    if "arabic" in lang or "arab" in lang:
        session["language"] = "Arabic"
        sysprompt = s["ar_system"]; reply = s["ar_greeting"]
    elif "urdu" in lang:
        session["language"] = "Urdu"
        sysprompt = s["ur_system"]; reply = s["ur_greeting"]
    else:
        session["language"] = "English"
        sysprompt = s["en_system"]; reply = s["en_greeting"]

    if kb:
        sysprompt += (
            f"\n\nMENU / KNOWLEDGE BASE (use ONLY items from this list when taking orders):\n"
            f"{kb}\n"
            f"IMPORTANT: Only accept orders for items listed above. "
            f"If customer asks for something not listed, say it is not available and suggest from the list."
        )
    session["history"] = [
        {"role": "system",    "content": sysprompt},
        {"role": "assistant", "content": reply},
    ]
    session["stage"] = "conversation"
    print(f"Language: {session['language']}")
    if session.get("call_db_id"):
        await db_update("calls", session["call_db_id"], {"language": session["language"]})
    audio = await speak(reply, session["language"])
    await ws.send_json({"type": "agent_message", "text": reply, "audio": audio})

async def finalize_call(session):
    history = session.get("history", [])
    if len([m for m in history if m.get("role") != "system"]) <= 1:
        return
    atype = session.get("agent_type", "custom")
    aname = session["script"]["name"]
    lang  = session.get("language", "English")
    cid   = session.get("call_db_id")
    print(f"Summarizing: {aname}...")
    summary = await summarize_call(atype, aname, history, lang)
    clean = [{"role": m["role"], "content": m["content"]}
             for m in history if m.get("role") != "system"]
    data = {
        "agent_type":     atype, "agent_name": aname, "language": lang,
        "transcript":     clean, "summary": summary.get("summary", ""),
        "sentiment":      summary.get("sentiment", "neutral"),
        "resolved":       bool(summary.get("resolved", False)),
        "key_actions":    summary.get("key_actions", []),
        "extracted_data": summary.get("extracted_data", {}),
        "turn_count":     summary.get("duration_turns", 0),
        "ended_at":       datetime.datetime.utcnow().isoformat(),
    }
    if cid:
        await db_update("calls", cid, data)
    else:
        row = await db_insert("calls", data)
        cid = str(row.get("id", "?")) if row else "?"
    await send_email(aname, atype, summary, str(cid))

# ── HTTP ROUTES ──────────────────────────────────────────────
async def health(req):
    return web.json_response({
        "status": "ok", "gpu_stt": stt_model is not None,
        "edge_tts": HAS_EDGE, "elevenlabs": bool(ELEVEN_KEY),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
        "email": bool(SMTP_USER and SMTP_PASS and NOTIFY_EMAIL),
        "languages": ["English", "Urdu", "Arabic"],
    })

async def get_agent_types(req):
    return web.json_response({
        k: {"name": v["name"], "color": v["color"],
            "icon": v["icon"], "description": v["description"]}
        for k, v in SCRIPTS.items()
    })

async def create_agent(req):
    d     = await req.json()
    atype = d.get("agent_type", "custom")
    kb    = d.get("knowledge_base", "")
    cn    = d.get("company_name", "")
    s     = dict(SCRIPTS.get(atype, SCRIPTS["custom"]))
    if atype == "custom" and cn:
        s["name"] = cn
    aid = str(uuid.uuid4())
    active_agents[aid] = {"script": s, "knowledge": kb, "agent_type": atype}
    print(f"Agent created: {s['name']} ({atype})")
    return web.json_response({"status": "success", "agent_id": aid, "name": s["name"]})

async def get_calls(req):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return web.json_response({"calls": [], "error": "Supabase not configured"})
    try:
        atype = req.rel_url.query.get("agent_type", "")
        url   = f"{SUPABASE_URL}/rest/v1/calls?order=started_at.desc&limit=100"
        if atype:
            url += f"&agent_type=eq.{atype}"
        async with httpx.AsyncClient(timeout=10) as h:
            r = await h.get(url, headers={"apikey": SUPABASE_KEY,
                                          "Authorization": f"Bearer {SUPABASE_KEY}"})
        return web.json_response({"calls": r.json()})
    except Exception as e:
        return web.json_response({"calls": [], "error": str(e)})

async def serve_index(req):
    f = os.path.join(FRONTEND_DIR, "dashboard.html")
    return web.FileResponse(f) if os.path.exists(f) else web.Response(text="dashboard.html missing", status=404)

async def serve_integration(req):
    f = os.path.join(FRONTEND_DIR, "integration.html")
    return web.FileResponse(f) if os.path.exists(f) else web.Response(text="integration.html missing", status=404)

async def serve_file(req):
    f = os.path.join(FRONTEND_DIR, req.match_info["filename"])
    return web.FileResponse(f) if os.path.exists(f) else web.Response(text="Not found", status=404)

# ── WEBSOCKET ────────────────────────────────────────────────
async def ws_handler(req):
    aid = req.match_info["agent_id"]
    if aid not in active_agents:
        return web.Response(status=404)
    agent = active_agents[aid]
    ws    = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(req)
    call_row = await db_insert("calls", {
        "agent_type": agent["agent_type"], "agent_name": agent["script"]["name"],
        "language": "unknown", "started_at": datetime.datetime.utcnow().isoformat(),
        "transcript": [], "summary": "", "sentiment": "neutral",
        "resolved": False, "key_actions": [], "extracted_data": {}, "turn_count": 0,
    })
    cid = str(call_row.get("id")) if call_row else None
    print(f"Call started: {agent['script']['name']} | DB:{(cid or 'none')[:8]}")
    session = {
        "stage": "language_selection", "language": "English",
        "history": [], "processing": False,
        "script": agent["script"], "knowledge": agent["knowledge"],
        "agent_type": agent["agent_type"], "call_db_id": cid,
    }
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                d = json.loads(msg.data)
                if d.get("type") == "interrupt":
                    session["processing"] = False
                elif d.get("type") == "start_call":
                    session["processing"] = True
                    g = (f"Welcome to {agent['script']['name']}! "
                         "Please say English, Urdu, or Arabic to begin.")
                    audio = await speak(g, "English")
                    await ws.send_json({"type": "agent_message", "text": g, "audio": audio})
                elif d.get("type") == "audio_chunk":
                    raw = d.get("audio", "")
                    if not raw: continue
                    session["processing"] = True
                    asyncio.ensure_future(pipeline(ws, session, base64.b64decode(raw)))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    except Exception as e:
        print(f"WS err: {e}")
    print(f"Call ended: {agent['script']['name']}")
    asyncio.ensure_future(finalize_call(session))
    return ws

# ── APP ──────────────────────────────────────────────────────
app  = web.Application()
cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(
    allow_credentials=True, expose_headers="*",
    allow_headers="*", allow_methods=["GET", "POST", "OPTIONS"])})
cors.add(app.router.add_get("/health",            health))
cors.add(app.router.add_get("/api/agent_types",   get_agent_types))
cors.add(app.router.add_post("/api/create_agent", create_agent))
cors.add(app.router.add_get("/api/calls",         get_calls))
app.router.add_get("/ws/{agent_id}",  ws_handler)
app.router.add_get("/",               serve_index)
app.router.add_get("/integration",    serve_integration)
app.router.add_get("/{filename}",     serve_file)

async def keep_alive():
    """Ping self every 10 min so Railway never sleeps"""
    await asyncio.sleep(30)  # wait for server to start first
    while True:
        try:
            async with httpx.AsyncClient(timeout=5) as h:
                await h.get(f"http://localhost:{PORT}/health")
            print("Keep-alive ping OK")
        except Exception:
            pass
        await asyncio.sleep(600)  # every 10 minutes

async def on_startup(app):
    asyncio.ensure_future(keep_alive())

app.on_startup.append(on_startup)

if __name__ == "__main__":
    stt_s  = "small-GPU beam=1 [OK]" if stt_model else "Groq fallback [WARN]"
    ur_s   = "ElevenLabs Flash [OK]" if ELEVEN_KEY else "edge-tts fallback [WARN]"
    db_s   = "Supabase [OK]" if (SUPABASE_URL and SUPABASE_KEY) else "NOT SET [WARN]"
    mail_s = "Configured [OK]" if (SMTP_USER and NOTIFY_EMAIL) else "NOT SET [WARN]"
    print("=" * 52)
    print("  NexaVoice v8.1  SPEED OPTIMIZED")
    print("=" * 52)
    print(f"  STT  : {stt_s}")
    print(f"  EN   : edge-tts +15% rate [OK]")
    print(f"  UR/AR: {ur_s}")
    print(f"  LLM  : gpt-oss-20b -> 8b-instant -> 70b [OK]")
    print(f"  DB   : {db_s}")
    print(f"  Mail : {mail_s}")
    print(f"  Port : {PORT}")
    print("=" * 52)
    web.run_app(app, host="0.0.0.0", port=PORT)