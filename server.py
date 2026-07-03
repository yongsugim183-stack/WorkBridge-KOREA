"""
동남아시아 + 중앙아시아 동시통역 서버
STT: faster-whisper (로컬, 무료) 우선 → OpenAI Whisper 폴백 → Web Speech API
번역: deep-translator (Google Translate 무료, API 키 불필요)
"""

import asyncio
import os
import tempfile
import time

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

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    print("[Whisper] 모델 로드 완료 - 준비됨", flush=True)

def _get_fw_model():
    return _fw_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 모델 미리 로드 (첫 요청 지연 제거)
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
    translator.session = _http_session  # 세션 재사용
    return translator.translate(text)


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="번역할 텍스트를 입력하세요.")

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
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    content_type = audio.content_type or "audio/webm"

    # ① faster-whisper 로컬 모델 (API 키 불필요)
    if FASTER_WHISPER:
        try:
            return await _transcribe_local(audio_bytes, content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"로컬 Whisper 오류: {str(e)}")

    # ② OpenAI Whisper API (OPENAI_API_KEY 설정 시 폴백)
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
        detail="STT 불가: faster-whisper 미설치 + OPENAI_API_KEY 미설정. 브라우저 음성인식 사용 중."
    )


async def _transcribe_local(audio_bytes: bytes, content_type: str) -> dict:
    ext = ".webm"
    if "mp4" in content_type:
        ext = ".mp4"
    elif "wav" in content_type:
        ext = ".wav"
    elif "ogg" in content_type:
        ext = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        loop = asyncio.get_event_loop()

        def do_transcribe():
            model = _get_fw_model()
            segments, info = model.transcribe(
                tmp_path,
                beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                temperature=0,
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


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/board")
async def board():
    return FileResponse("board.html")


# ── 게시판 ────────────────────────────────────────────────────────────────────
BOARD_FILE = Path("board_data.json")
BOARD_ADMIN_PW = os.environ.get("BOARD_ADMIN_PW", "0101")
_board_lock = asyncio.Lock()

# JSONBin.io 외부 영구 저장소 (환경변수 설정 시 사용)
_JSONBIN_KEY    = os.environ.get("JSONBIN_KEY", "")
_JSONBIN_BIN_ID = os.environ.get("JSONBIN_BIN_ID", "")
_USE_JSONBIN    = bool(_JSONBIN_KEY and _JSONBIN_BIN_ID)


def _load_board_local() -> dict:
    if BOARD_FILE.exists():
        try:
            d = json.loads(BOARD_FILE.read_text(encoding="utf-8"))
            if "emergency" not in d:
                d["emergency"] = []
            if "culture" not in d:
                d["culture"] = []
            return d
        except Exception:
            pass
    return {"posts": [], "emergency": [], "culture": []}


def _save_board_local(data: dict):
    try:
        BOARD_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


async def _load_board() -> dict:
    if _USE_JSONBIN:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.get(
                    f"https://api.jsonbin.io/v3/b/{_JSONBIN_BIN_ID}/latest",
                    headers={"X-Master-Key": _JSONBIN_KEY},
                )
                if res.status_code == 200:
                    data = res.json().get("record", {"posts": []})
                    _save_board_local(data)   # 로컬 백업
                    return data
        except Exception as e:
            print(f"[JSONBin] load error: {e}", flush=True)
    return _load_board_local()


async def _save_board(data: dict):
    _save_board_local(data)   # 로컬 항상 저장
    if _USE_JSONBIN:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                await client.put(
                    f"https://api.jsonbin.io/v3/b/{_JSONBIN_BIN_ID}",
                    json=data,
                    headers={"X-Master-Key": _JSONBIN_KEY, "Content-Type": "application/json"},
                )
        except Exception as e:
            print(f"[JSONBin] save error: {e}", flush=True)


class BoardPostRequest(BaseModel):
    author_name: str
    lang: str
    title: str = ""
    text: str


class BoardReplyRequest(BaseModel):
    text: str
    admin_password: str = ""
    author_name: str = ""
    lang: str = "en"


@app.get("/api/board/posts")
async def get_board_posts():
    async with _board_lock:
        data = await _load_board()
    return data["posts"]


@app.post("/api/board/posts")
async def create_board_post(req: BoardPostRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")
    if req.lang not in LANGUAGES:
        raise HTTPException(status_code=400, detail="지원하지 않는 언어입니다.")

    src_gt = LANGUAGES[req.lang]["gt"]
    loop = asyncio.get_event_loop()

    # 제목·내용 동시 번역
    translate_tasks = []
    has_title = bool(req.title.strip())
    if has_title:
        translate_tasks.append(loop.run_in_executor(_translate_pool, _translate_one, req.title, src_gt, "ko"))
    if req.lang != "ko":
        translate_tasks.append(loop.run_in_executor(_translate_pool, _translate_one, req.text, src_gt, "ko"))

    results = await asyncio.gather(*translate_tasks, return_exceptions=True) if translate_tasks else []

    idx = 0
    if has_title:
        title_ko = results[idx] if isinstance(results[idx], str) else req.title
        idx += 1
    else:
        title_ko = ""

    korean_text = results[idx] if idx < len(results) and isinstance(results[idx], str) else req.text

    post = {
        "id": str(uuid.uuid4()),
        "author_name": req.author_name.strip() or "익명",
        "lang": req.lang,
        "lang_name": LANGUAGES[req.lang]["name"],
        "flag": LANGUAGES[req.lang]["flag"],
        "title": req.title.strip(),
        "title_ko": title_ko,
        "original_text": req.text,
        "korean_text": korean_text,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "replies": [],
    }

    async with _board_lock:
        data = await _load_board()
        data["posts"].insert(0, post)
        await _save_board(data)

    return post


@app.post("/api/board/posts/{post_id}/reply")
async def create_board_reply(post_id: str, req: BoardReplyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    is_admin = req.admin_password == BOARD_ADMIN_PW
    loop = asyncio.get_event_loop()

    async with _board_lock:
        data = await _load_board()
        post = next((p for p in data["posts"] if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

        if is_admin:
            # 관리자: 한국어 작성 → 게시글 작성자 언어로 번역
            post_lang = post["lang"]
            if post_lang == "ko":
                translated_text = req.text
            else:
                tgt_gt = LANGUAGES[post_lang]["gt"]
                translated_text = await loop.run_in_executor(
                    _translate_pool, _translate_one, req.text, "ko", tgt_gt
                )
            reply = {
                "id": str(uuid.uuid4()),
                "is_admin": True,
                "author_name": "Admin (관리자)",
                "lang": "ko",
                "flag": "👑",
                "lang_name": "관리자",
                "original_text": req.text,
                "korean_text": req.text,
                "translated_text": translated_text,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            }
        else:
            # 일반 이용자: 자신의 언어로 작성 → 한국어로 번역
            if req.lang not in LANGUAGES:
                raise HTTPException(status_code=400, detail="지원하지 않는 언어입니다.")
            src_gt = LANGUAGES[req.lang]["gt"]
            if req.lang == "ko":
                korean_text = req.text
            else:
                korean_text = await loop.run_in_executor(
                    _translate_pool, _translate_one, req.text, src_gt, "ko"
                )
            reply = {
                "id": str(uuid.uuid4()),
                "is_admin": False,
                "author_name": req.author_name.strip() or "익명",
                "lang": req.lang,
                "flag": LANGUAGES[req.lang]["flag"],
                "lang_name": LANGUAGES[req.lang]["name"],
                "original_text": req.text,
                "korean_text": korean_text,
                "translated_text": req.text,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            }

        post["replies"].append(reply)
        await _save_board(data)

    return reply


@app.delete("/api/board/posts/{post_id}")
async def delete_board_post(post_id: str, pw: str = ""):
    if pw != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 비밀번호가 틀렸습니다.")
    async with _board_lock:
        data = await _load_board()
        before = len(data["posts"])
        data["posts"] = [p for p in data["posts"] if p["id"] != post_id]
        if len(data["posts"]) == before:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        await _save_board(data)
    return {"ok": True}


class BoardEditRequest(BaseModel):
    text: str
    admin_password: str = ""


@app.put("/api/board/posts/{post_id}")
async def edit_board_post(post_id: str, req: BoardEditRequest):
    if req.admin_password != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 비밀번호가 틀렸습니다.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    loop = asyncio.get_event_loop()

    async with _board_lock:
        data = await _load_board()
        post = next((p for p in data["posts"] if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

        post_lang = post["lang"]
        src_gt = LANGUAGES.get(post_lang, {}).get("gt", "auto")
        if post_lang == "ko":
            korean_text = req.text
        else:
            korean_text = await loop.run_in_executor(
                _translate_pool, _translate_one, req.text, src_gt, "ko"
            )
        post["original_text"] = req.text
        post["korean_text"] = korean_text
        post["edited"] = True
        await _save_board(data)

    return post


@app.delete("/api/board/posts/{post_id}/replies/{reply_id}")
async def delete_board_reply(post_id: str, reply_id: str, pw: str = ""):
    if pw != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 비밀번호가 틀렸습니다.")
    async with _board_lock:
        data = await _load_board()
        post = next((p for p in data["posts"] if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        before = len(post["replies"])
        post["replies"] = [r for r in post["replies"] if r["id"] != reply_id]
        if len(post["replies"]) == before:
            raise HTTPException(status_code=404, detail="답글을 찾을 수 없습니다.")
        await _save_board(data)
    return {"ok": True}


@app.put("/api/board/posts/{post_id}/replies/{reply_id}")
async def edit_board_reply(post_id: str, reply_id: str, req: BoardEditRequest):
    if req.admin_password != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 비밀번호가 틀렸습니다.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    loop = asyncio.get_event_loop()

    async with _board_lock:
        data = await _load_board()
        post = next((p for p in data["posts"] if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        reply = next((r for r in post["replies"] if r["id"] == reply_id), None)
        if not reply:
            raise HTTPException(status_code=404, detail="답글을 찾을 수 없습니다.")

        if reply.get("is_admin", True):
            # 관리자 답글: 한국어 → 게시글 작성자 언어로 번역
            post_lang = post["lang"]
            if post_lang == "ko":
                translated_text = req.text
            else:
                tgt_gt = LANGUAGES[post_lang]["gt"]
                translated_text = await loop.run_in_executor(
                    _translate_pool, _translate_one, req.text, "ko", tgt_gt
                )
            reply["original_text"] = req.text
            reply["korean_text"] = req.text
            reply["translated_text"] = translated_text
        else:
            # 이용자 답글: 자국어 → 한국어 번역
            reply_lang = reply.get("lang", "en")
            src_gt = LANGUAGES.get(reply_lang, {}).get("gt", "auto")
            if reply_lang == "ko":
                korean_text = req.text
            else:
                korean_text = await loop.run_in_executor(
                    _translate_pool, _translate_one, req.text, src_gt, "ko"
                )
            reply["original_text"] = req.text
            reply["korean_text"] = korean_text
            reply["translated_text"] = req.text

        reply["edited"] = True
        await _save_board(data)

    return reply


@app.post("/api/board/admin/verify")
async def verify_admin(body: dict):
    if body.get("password") == BOARD_ADMIN_PW:
        return {"ok": True}
    raise HTTPException(status_code=403, detail="비밀번호가 틀렸습니다.")


# ── 긴급 연락망 ───────────────────────────────────────────────────────────────
@app.get("/emergency")
async def emergency_page():
    return FileResponse("emergency.html")


class EmergencyPostRequest(BaseModel):
    title: str
    text: str
    admin_password: str


@app.get("/api/emergency/posts")
async def get_emergency_posts():
    async with _board_lock:
        data = await _load_board()
    return data.get("emergency", [])


async def _translate_emergency(title: str, text: str) -> tuple[dict, dict]:
    """제목·내용을 12개 언어로 병렬 번역하여 (title_trans, text_trans) 반환."""
    loop = asyncio.get_event_loop()
    lang_codes = [c for c in LANGUAGES if c != "ko"]
    title_tasks = [
        loop.run_in_executor(_translate_pool, _translate_one, title, "ko", LANGUAGES[c]["gt"])
        for c in lang_codes
    ]
    text_tasks = [
        loop.run_in_executor(_translate_pool, _translate_one, text, "ko", LANGUAGES[c]["gt"])
        for c in lang_codes
    ]
    all_results = await asyncio.gather(*title_tasks, *text_tasks, return_exceptions=True)
    title_results = all_results[:len(lang_codes)]
    text_results  = all_results[len(lang_codes):]

    title_trans = {"ko": title}
    text_trans  = {"ko": text}
    for code, tr, tx in zip(lang_codes, title_results, text_results):
        title_trans[code] = tr if isinstance(tr, str) else title
        text_trans[code]  = tx if isinstance(tx, str) else text
    return title_trans, text_trans


@app.post("/api/emergency/posts")
async def create_emergency_post(req: EmergencyPostRequest):
    if req.admin_password != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    title_trans, text_trans = await _translate_emergency(req.title.strip(), req.text)

    post = {
        "id": str(uuid.uuid4()),
        "title": req.title.strip(),
        "title_translations": title_trans,
        "original_text": req.text,
        "translations": text_trans,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "edited": False,
    }

    async with _board_lock:
        data = await _load_board()
        data.setdefault("emergency", []).insert(0, post)
        await _save_board(data)

    return post


@app.put("/api/emergency/posts/{post_id}")
async def edit_emergency_post(post_id: str, req: EmergencyPostRequest):
    if req.admin_password != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    title_trans, text_trans = await _translate_emergency(req.title.strip(), req.text)

    async with _board_lock:
        data = await _load_board()
        post = next((p for p in data.get("emergency", []) if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        post["title"] = req.title.strip()
        post["title_translations"] = title_trans
        post["original_text"] = req.text
        post["translations"] = text_trans
        post["edited"] = True
        await _save_board(data)

    return post


@app.delete("/api/emergency/posts/{post_id}")
async def delete_emergency_post(post_id: str, pw: str = ""):
    if pw != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    async with _board_lock:
        data = await _load_board()
        before = len(data.get("emergency", []))
        data["emergency"] = [p for p in data.get("emergency", []) if p["id"] != post_id]
        if len(data["emergency"]) == before:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        await _save_board(data)
    return {"ok": True}


# ── 한국 직장문화 안내 ────────────────────────────────────────────────────────
@app.get("/culture")
async def culture_page():
    return FileResponse("culture.html")


class CulturePostRequest(BaseModel):
    title: str = ""
    text: str
    admin_password: str


@app.get("/api/culture/posts")
async def get_culture_posts():
    async with _board_lock:
        data = await _load_board()
    return data.get("culture", [])


@app.post("/api/culture/posts")
async def create_culture_post(req: CulturePostRequest):
    if req.admin_password != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    title_trans, text_trans = await _translate_emergency(req.title.strip(), req.text)

    post = {
        "id": str(uuid.uuid4()),
        "title": req.title.strip(),
        "title_translations": title_trans,
        "original_text": req.text,
        "translations": text_trans,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "edited": False,
    }

    async with _board_lock:
        data = await _load_board()
        data.setdefault("culture", []).insert(0, post)
        await _save_board(data)

    return post


@app.put("/api/culture/posts/{post_id}")
async def edit_culture_post(post_id: str, req: CulturePostRequest):
    if req.admin_password != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="내용을 입력하세요.")

    title_trans, text_trans = await _translate_emergency(req.title.strip(), req.text)

    async with _board_lock:
        data = await _load_board()
        post = next((p for p in data.get("culture", []) if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        post.update({
            "title": req.title.strip(),
            "title_translations": title_trans,
            "original_text": req.text,
            "translations": text_trans,
            "edited": True,
        })
        await _save_board(data)
    return post


@app.delete("/api/culture/posts/{post_id}")
async def delete_culture_post(post_id: str, pw: str = ""):
    if pw != BOARD_ADMIN_PW:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    async with _board_lock:
        data = await _load_board()
        before = len(data.get("culture", []))
        data["culture"] = [p for p in data.get("culture", []) if p["id"] != post_id]
        if len(data["culture"]) == before:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        await _save_board(data)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

