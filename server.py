"""
동남아시아 + 중앙아시아 동시통역 서버
STT: faster-whisper (로컬, 무료) 우선 → OpenAI Whisper 폴백 → Web Speech API
번역: deep-translator (Google Translate 무료, API 키 불필요)
"""

import asyncio
import os
import secrets
import tempfile
import time
from datetime import datetime

# 회사 네트워크 SSL 검사 우회 (DISABLE_SSL_VERIFY=1 환경변수가 있을 때만 활성화)
if os.environ.get("DISABLE_SSL_VERIFY") == "1":
    import ssl
    import urllib3
    import requests
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_request = requests.Session.request
    def _patched_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, *args, **kwargs)
    requests.Session.request = _patched_request
    try:
        import httpx
        _hi = httpx.Client.__init__
        def _phi(self, *a, **kw): kw.setdefault("verify", False); _hi(self, *a, **kw)
        httpx.Client.__init__ = _phi
        _hai = httpx.AsyncClient.__init__
        def _phai(self, *a, **kw): kw.setdefault("verify", False); _hai(self, *a, **kw)
        httpx.AsyncClient.__init__ = _phai
    except Exception:
        pass

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Cookie, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from deep_translator import GoogleTranslator

# ── 번역 전용 스레드 풀 ───────────────────────────────────────────────────
_translate_pool = ThreadPoolExecutor(max_workers=24)

# ── 재사용 HTTP 세션 (TCP 연결 풀링으로 속도 향상) ───────────────────────
import requests as _requests
_http_session = _requests.Session()
_http_adapter = _requests.adapters.HTTPAdapter(
    pool_connections=24, pool_maxsize=24, max_retries=1
)
_http_session.mount("https://", _http_adapter)
_http_session.mount("http://", _http_adapter)

# ── 초대 코드 설정 ────────────────────────────────────────────────────────
# 환경변수 INVITE_CODES로 덮어쓰기 가능 (형식: "코드1:이름1,코드2:이름2")
def _load_invite_codes() -> dict:
    env = os.environ.get("INVITE_CODES", "")
    if env:
        result = {}
        for item in env.split(","):
            parts = item.strip().split(":", 1)
            if len(parts) == 2:
                result[parts[0].strip()] = parts[1].strip()
        if result:
            return result
    return {
        "KPC2026A": "사용자A",
        "KPC2026B": "사용자B",
        "KPC2026C": "사용자C",
        "KPC2026D": "사용자D",
        "KPC2026E": "사용자E",
    }

INVITE_CODES: dict[str, str] = _load_invite_codes()
ADMIN_CODE: str = os.environ.get("ADMIN_CODE", "KPCADMIN2026")
MAX_CONCURRENT: int = int(os.environ.get("MAX_CONCURRENT", "5"))
SESSION_TIMEOUT: int = 60 * 60  # 1시간 비활동 시 세션 만료

# ── 세션 저장소 (메모리) ──────────────────────────────────────────────────
# {token: {code, name, is_admin, login_time, last_active}}
_sessions: dict[str, dict] = {}
# {code: {trans_count, char_count, total_sec, last_access, name}}
_user_stats: dict[str, dict] = {}


def _now() -> float:
    return time.time()

def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"

def _cleanup_sessions():
    expired = [t for t, s in _sessions.items() if _now() - s["last_active"] > SESSION_TIMEOUT]
    for t in expired:
        s = _sessions.pop(t)
        # 누적 시간 저장
        code = s["code"]
        elapsed = _now() - s["login_time"]
        if code in _user_stats:
            _user_stats[code]["total_sec"] += elapsed

def _active_sessions() -> list[dict]:
    _cleanup_sessions()
    return list(_sessions.values())

def _get_session(token: str | None) -> dict | None:
    if not token or token not in _sessions:
        return None
    s = _sessions[token]
    if _now() - s["last_active"] > SESSION_TIMEOUT:
        _sessions.pop(token, None)
        return None
    s["last_active"] = _now()
    return s

def _require_session(token: str | None) -> dict:
    s = _get_session(token)
    if not s:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return s

def _require_admin(token: str | None) -> dict:
    s = _require_session(token)
    if not s.get("is_admin"):
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return s

def _update_stats(code: str, char_count: int = 0):
    if code not in _user_stats:
        _user_stats[code] = {"trans_count": 0, "char_count": 0, "total_sec": 0, "last_access": 0,
                              "name": INVITE_CODES.get(code, code)}
    st = _user_stats[code]
    st["trans_count"] += 1
    st["char_count"] += char_count
    st["last_access"] = _now()


# ── faster-whisper 로컬 모델 ──────────────────────────────────────────────
_fw_model = None

def _fw_available() -> bool:
    try:
        import faster_whisper  # noqa
        return True
    except ImportError:
        return False

FASTER_WHISPER = _fw_available()

def _load_fw_model():
    global _fw_model
    from faster_whisper import WhisperModel
    model_size = os.environ.get("WHISPER_MODEL", "base")
    print(f"[Whisper] 모델 로딩 중: {model_size}", flush=True)
    cpu_threads = int(os.environ.get("WHISPER_THREADS", "2"))
    _fw_model = WhisperModel(
        model_size, device="cpu", compute_type="int8",
        cpu_threads=cpu_threads, num_workers=1,
    )
    print("[Whisper] 모델 로드 완료 — 준비됨", flush=True)

def _get_fw_model():
    return _fw_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    if FASTER_WHISPER:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_fw_model)
    yield

app = FastAPI(title="동시통역 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

LANGUAGES = {
    "ko": {"name": "한국어",           "flag": "🇰🇷", "region": "동북아시아", "gt": "ko"},
    "en": {"name": "English",          "flag": "🇺🇸", "region": "국제",       "gt": "en"},
    "zh": {"name": "中文(简体)",        "flag": "🇨🇳", "region": "동북아시아", "gt": "zh-CN"},
    "th": {"name": "ภาษาไทย",          "flag": "🇹🇭", "region": "동남아시아", "gt": "th"},
    "vi": {"name": "Tiếng Việt",       "flag": "🇻🇳", "region": "동남아시아", "gt": "vi"},
    "id": {"name": "Bahasa Indonesia",  "flag": "🇮🇩", "region": "동남아시아", "gt": "id"},
    "ms": {"name": "Bahasa Melayu",    "flag": "🇲🇾", "region": "동남아시아", "gt": "ms"},
    "tl": {"name": "Filipino",         "flag": "🇵🇭", "region": "동남아시아", "gt": "tl"},
    "my": {"name": "မြန်မာဘာသာ",       "flag": "🇲🇲", "region": "동남아시아", "gt": "my"},
    "uz": {"name": "O'zbek tili",      "flag": "🇺🇿", "region": "중앙아시아", "gt": "uz"},
    "si": {"name": "සිංහල",             "flag": "🇱🇰", "region": "남아시아",   "gt": "si"},
    "mn": {"name": "Монгол хэл",        "flag": "🇲🇳", "region": "동북아시아", "gt": "mn"},
}


# ── 인증 API ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    code: str

@app.post("/api/login")
async def login(req: LoginRequest, response: Response, request: Request):
    code = req.code.strip()
    is_admin = (code == ADMIN_CODE)
    if not is_admin and code not in INVITE_CODES:
        raise HTTPException(status_code=401, detail="유효하지 않은 초대 코드입니다.")

    # 클라이언트 IP (프록시/Render 환경 고려)
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.headers.get("x-real-ip", "")
        or (request.client.host if request.client else "unknown")
    )

    # 동시 접속 제한 (관리자 제외)
    if not is_admin:
        _cleanup_sessions()
        active_user_sessions = [s for s in _sessions.values() if not s["is_admin"]]
        # 같은 코드로 이미 접속 중이면 기존 세션 교체 허용
        same_code = [s for s in active_user_sessions if s["code"] == code]
        other_sessions = [s for s in active_user_sessions if s["code"] != code]
        if not same_code and len(other_sessions) >= MAX_CONCURRENT:
            raise HTTPException(
                status_code=429,
                detail=f"동시 접속자 수({MAX_CONCURRENT}명)를 초과했습니다. 잠시 후 다시 시도해주세요."
            )
        # 같은 코드 기존 세션 제거
        for old_token, s in list(_sessions.items()):
            if s["code"] == code:
                _sessions.pop(old_token, None)

    name = "관리자" if is_admin else INVITE_CODES[code]
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "code": code, "name": name, "is_admin": is_admin,
        "login_time": _now(), "last_active": _now(),
        "ip": client_ip,
    }
    if not is_admin and code not in _user_stats:
        _user_stats[code] = {"trans_count": 0, "char_count": 0, "total_sec": 0,
                              "last_access": 0, "name": name}

    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=SESSION_TIMEOUT)
    return {"name": name, "is_admin": is_admin}

@app.post("/api/logout")
async def logout(response: Response, session: str | None = Cookie(default=None)):
    if session and session in _sessions:
        s = _sessions.pop(session)
        code = s["code"]
        if code in _user_stats:
            _user_stats[code]["total_sec"] += _now() - s["login_time"]
    response.delete_cookie("session")
    return {"ok": True}

@app.get("/api/me")
async def me(session: str | None = Cookie(default=None)):
    s = _get_session(session)
    if not s:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    active = len([x for x in _sessions.values() if not x["is_admin"]])
    return {"name": s["name"], "is_admin": s["is_admin"], "active_users": active, "max_users": MAX_CONCURRENT}

@app.get("/api/admin/stats")
async def admin_stats(session: str | None = Cookie(default=None)):
    _require_admin(session)
    _cleanup_sessions()

    active_tokens = {s["code"] for s in _sessions.values() if not s["is_admin"]}
    active_count = len(active_tokens)

    # 현재 접속자 IP 맵
    active_ip_map: dict[str, str] = {}
    for s in _sessions.values():
        if not s["is_admin"]:
            active_ip_map[s["code"]] = s.get("ip", "-")

    stats = []
    for code, st in _user_stats.items():
        current_sec = 0
        for s in _sessions.values():
            if s["code"] == code:
                current_sec = _now() - s["login_time"]
        total_sec = st["total_sec"] + current_sec
        stats.append({
            "code": code,
            "name": st["name"],
            "online": code in active_tokens,
            "ip": active_ip_map.get(code, "-"),
            "trans_count": st["trans_count"],
            "char_count": st["char_count"],
            "total_min": round(total_sec / 60, 1),
            "last_access": _fmt(st["last_access"]),
        })
    stats.sort(key=lambda x: (-x["online"], -x["trans_count"]))

    return {
        "active_count": active_count,
        "max_concurrent": MAX_CONCURRENT,
        "stats": stats,
    }


# ── 기존 API (세션 필요) ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    whisper_ok = FASTER_WHISPER or bool(os.environ.get("OPENAI_API_KEY"))
    whisper_mode = "local" if FASTER_WHISPER else ("openai" if os.environ.get("OPENAI_API_KEY") else "none")
    return {"status": "ok", "whisper": whisper_ok, "whisper_mode": whisper_mode}


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_langs: list[str] = list(LANGUAGES.keys())


class TranslateResponse(BaseModel):
    source_text: str
    source_lang: str
    translations: dict[str, str]
    elapsed_ms: int


def _translate_one(text: str, src: str, tgt_gt: str) -> str:
    translator = GoogleTranslator(source=src, target=tgt_gt)
    translator.session = _http_session
    return translator.translate(text)


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, session: str | None = Cookie(default=None)):
    s = _require_session(session)
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="번역할 텍스트를 입력하세요.")

    _update_stats(s["code"], len(req.text))

    start = time.time()
    src = "auto" if req.source_lang == "auto" else LANGUAGES.get(req.source_lang, {}).get("gt", "auto")

    loop = asyncio.get_event_loop()
    codes = [code for code in req.target_langs if code in LANGUAGES]
    tasks = [
        loop.run_in_executor(_translate_pool, _translate_one, req.text, src, LANGUAGES[code]["gt"])
        for code in codes
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    translations = {
        code: (r if isinstance(r, str) else "")
        for code, r in zip(codes, results)
    }

    return TranslateResponse(
        source_text=req.text,
        source_lang=req.source_lang,
        translations=translations,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), session: str | None = Cookie(default=None)):
    s = _require_session(session)
    audio_bytes = await audio.read()
    content_type = audio.content_type or "audio/webm"

    _update_stats(s["code"])

    if FASTER_WHISPER:
        try:
            return await _transcribe_local(audio_bytes, content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"로컬 Whisper 오류: {str(e)}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=(audio.filename or "recording.webm", audio_bytes, content_type),
                response_format="verbose_json",
            )
            return {
                "text": transcript.text,
                "language": transcript.language,
                "segments": [{"text": s.text, "start": s.start, "end": s.end}
                              for s in (transcript.segments or [])],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI Whisper 오류: {str(e)}")

    raise HTTPException(
        status_code=503,
        detail="STT 불가: faster-whisper 미설치 + OPENAI_API_KEY 미설정."
    )


async def _transcribe_local(audio_bytes: bytes, content_type: str) -> dict:
    ext = ".webm"
    if "mp4" in content_type: ext = ".mp4"
    elif "wav" in content_type: ext = ".wav"
    elif "ogg" in content_type: ext = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        loop = asyncio.get_event_loop()
        def do_transcribe():
            model = _get_fw_model()
            segments, info = model.transcribe(
                tmp_path, beam_size=1, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False, temperature=0,
            )
            text = "".join(s.text for s in segments).strip()
            return text, info.language
        text, lang = await loop.run_in_executor(None, do_transcribe)
        return {"text": text, "language": lang, "segments": []}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/languages")
async def get_languages():
    return LANGUAGES


# ── 페이지 라우팅 ──────────────────────────────────────────────────────────

@app.get("/admin")
async def admin_page(session: str | None = Cookie(default=None)):
    s = _get_session(session)
    if not s or not s["is_admin"]:
        return FileResponse("login.html")
    return FileResponse("admin.html")

@app.get("/")
async def root(session: str | None = Cookie(default=None)):
    s = _get_session(session)
    if not s:
        return FileResponse("login.html")
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
