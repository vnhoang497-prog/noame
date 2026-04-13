import asyncio
import io
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
import google.generativeai as genai
from groq import Groq
from mistralai import Mistral
from PIL import Image
from telegram import Update
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
Application, CommandHandler, ConversationHandler,
MessageHandler, ContextTypes, filters,
)

# ==================================================

# DIEN TOKEN VAO DAY

# ==================================================

BOT1_TOKEN = "8631318177:AAF88vTOUV2ezJGzi9A3DWkvupt5jQLQl8o"   # token Telegram Bot1
GROUP_ID   = -1003719387951               # ID group nhan ket qua

# ==================================================

TOKEN_FILE  = Path("token.txt")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Conversation states

S_INPUT = 1   # cho nhan cau hoi / anh
S_OK    = 2   # da nhan input, cho /ok

SPINNER = ["|", "/", "-", "\\"]

logging.basicConfig(
format="%(asctime)s | %(levelname)s | %(message)s",
level=logging.INFO,
)
log = logging.getLogger(__name__)

# ==================================================

# DOC token.txt

# ==================================================

def _load(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

_T = _load(TOKEN_FILE)

GEMINI_KEYS  = [v for k, v in _T.items() if k.startswith("GEMINI_KEY")]
GROQ_KEYS    = [v for k, v in _T.items() if k.startswith("GROQ_KEY")]

# Mistral: lay key dau tien (chi 1 key)
MISTRAL_KEY  = next((v for k, v in _T.items() if k.startswith("MISTRAL_KEY")), "")

_gi = _gri = 0

def _next_gemini() -> str:
    global _gi
    k = GEMINI_KEYS[_gi % len(GEMINI_KEYS)]
    _gi += 1
    return k

def _next_groq() -> str:
    global _gri
    k = GROQ_KEYS[_gri % len(GROQ_KEYS)]
    _gri += 1
    return k

log.info(f"Loaded: {len(GEMINI_KEYS)} Gemini | {len(GROQ_KEYS)} Groq | Mistral={'OK' if MISTRAL_KEY else 'MISSING'}")

# ==================================================

# NEN ANH  (max 1600px, quality 75)

# ==================================================

def compress_image(data: bytes, max_w: int = 1600, quality: int = 75) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    result = buf.getvalue()
    log.info(f"compress: {len(data)//1024}KB -> {len(result)//1024}KB")
    return result

# ==================================================

# SPINNER  (edit_message moi 4s, tranh rate-limit)

# ==================================================

async def run_spinner(msg, prefix: str, stop: asyncio.Event):
    i = 0
    while not stop.is_set():
        try:
            await msg.edit_text(f"{SPINNER[i % 4]} {prefix}")
        except (BadRequest, RetryAfter):
            pass
        except Exception:
            pass
        i += 1
        await asyncio.sleep(4)

# ==================================================

# GOI GEMINI  (xoay vong key, retry 429)

# ==================================================

async def call_gemini(question: str, image_bytes: bytes = None) -> str:
    for attempt in range(len(GEMINI_KEYS) * 2):
        key = _next_gemini()
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-3-flash-preview")
            parts = []
            if image_bytes:
                parts.append(Image.open(io.BytesIO(image_bytes)))
            parts.append(question)
            r = await asyncio.to_thread(model.generate_content, parts)
            return r.text
        except Exception as e:
            log.warning(f"Gemini [{attempt+1}] {e}")
            await asyncio.sleep(12 if "429" in str(e) else 3)
    return "[Gemini: that bai]"

# ==================================================

# GOI GEMINI OCR  (trich xuat de bai)

# ==================================================

_OCR_PROMPT = (
"Ban la cong cu OCR chuyen nghiep.\n"
"Doc TOAN BO noi dung trong anh, trich xuat thanh plain text.\n"
"Quy tac:\n"
"- Giu nguyen so thu tu cau hoi va dap an A/B/C/D\n"
"- Cong thuc toan: dung ky hieu text (sin(x), x^2, sqrt(x), pi)\n"
"- Tra ve KET QUA trong dung mot markdown code block ```\n"
"- Khong giai thich, khong them gi ngoai code block"
)

async def call_gemini_ocr(image_bytes: bytes) -> str:
    for attempt in range(len(GEMINI_KEYS) * 2):
        key = _next_gemini()
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            r = await asyncio.to_thread(
                model.generate_content,
                [Image.open(io.BytesIO(image_bytes)), _OCR_PROMPT]
            )
            return r.text
        except Exception as e:
            log.warning(f"Gemini OCR [{attempt+1}] {e}")
            await asyncio.sleep(12 if "429" in str(e) else 3)
    return "`\n[OCR that bai]\n`"

# ==================================================

# GOI GROQ  (stream text, sync vision)

# ==================================================

async def call_groq(question: str, image_bytes: bytes = None) -> str:
    import base64
    for attempt in range(len(GROQ_KEYS) * 2):
        key = _next_groq()
        try:
            client = Groq(api_key=key)
            if image_bytes:
                b64 = base64.b64encode(image_bytes).decode()
                msgs = [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": question},
                ]}]
                r = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=msgs, max_tokens=2000,
                )
                return r.choices[0].message.content
            else:
                # stream
                chunks = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": question}],
                    max_tokens=2000, stream=True,
                )
                return "".join(
                    c.choices[0].delta.content or "" for c in chunks
                )
        except Exception as e:
            log.warning(f"Groq [{attempt+1}] {e}")
            await asyncio.sleep(12 if "429" in str(e) else 3)
    return "[Groq: that bai]"

# ==================================================

# GOI MISTRAL  (1 key, stream text, sync vision)

# ==================================================

async def call_mistral(question: str, image_bytes: bytes = None) -> str:
    import base64
    for attempt in range(3):  # retry toi da 3 lan
        try:
            client = Mistral(api_key=MISTRAL_KEY)
            if image_bytes:
                b64 = base64.b64encode(image_bytes).decode()
                msgs = [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": f"data:image/jpeg;base64,{b64}"},
                    {"type": "text", "text": question},
                ]}]
                # pixtral-12b ho tro vision
                r = await asyncio.to_thread(
                    client.chat.complete,
                    model="pixtral-12b-2409",
                    messages=msgs, max_tokens=2000,
                )
                return r.choices[0].message.content
            else:
                # stream voi mistral-small-latest
                stream = await asyncio.to_thread(
                    client.chat.stream,
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": question}],
                    max_tokens=2000,
                )
                return "".join(
                    c.data.choices[0].delta.content or "" for c in stream
                )
        except Exception as e:
            log.warning(f"Mistral [{attempt+1}] {e}")
            await asyncio.sleep(10 if "429" in str(e) else 3)
    return "[Mistral: that bai]"

# ==================================================

# LUU FILE KET QUA

# ==================================================

async def save_result(sid: str, ai: str, question: str, answer: str) -> Path:
    p = RESULTS_DIR / f"{sid}_{ai}.txt"
    body = (
        f"=== KET QUA TU {ai.upper()} ===\n"
        f"Thoi gian : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Cau hoi   : {question}\n"
        f"{'-'*50}\n\n{answer}\n"
    )
    async with aiofiles.open(p, "w", encoding="utf-8") as f:
        await f.write(body)
    return p

# ==================================================

# GUI VAO GROUP

# ==================================================

async def send_to_group(bot, sid: str, question: str,
                        result_files: list, image_bytes: bytes = None):
    header = (
        f"Cau hoi moi | Session {sid}\n\n"
        f"? {question}"
    )
    if image_bytes:
        await bot.send_photo(chat_id=GROUP_ID, photo=image_bytes,
                             caption=header, parse_mode="Markdown")
    else:
        await bot.send_message(chat_id=GROUP_ID, text=header,
                               parse_mode="Markdown")
    await asyncio.sleep(1)

    for fp in result_files:
        async with aiofiles.open(fp, "rb") as f:
            data = await f.read()
        ai_name = fp.stem.split("_", 1)[-1]
        await bot.send_document(
            chat_id=GROUP_ID, document=data,
            filename=fp.name,
            caption=f"{ai_name}",
            parse_mode="Markdown",
        )
        await asyncio.sleep(0.5)

# ==================================================

# XU LY CHINH  (3 AI song song)

# ==================================================

pending_sessions: dict = {}   # sid -> {user_id, question, time}

async def process_question(bot, user_id: int, question: str,
                           image_bytes: bytes = None,
                           status_msg=None) -> str:
    sid      = uuid.uuid4().hex[:8]
    stop_ev  = asyncio.Event()

    if status_msg:
        spin = asyncio.create_task(
            run_spinner(status_msg, "Dang gui den 3 AI...", stop_ev)
        )

    try:
        # 3 AI chay song song
        g_ans, gr_ans, m_ans = await asyncio.gather(
            call_gemini(question, image_bytes),
            call_groq(question, image_bytes),
            call_mistral(question, image_bytes),
            return_exceptions=True,
        )
        def safe(x): return x if isinstance(x, str) else f"[Loi: {x}]"
        g_ans, gr_ans, m_ans = safe(g_ans), safe(gr_ans), safe(m_ans)

        # Luu 3 file song song
        f1, f2, f3 = await asyncio.gather(
            save_result(sid, "Gemini",  question, g_ans),
            save_result(sid, "Groq",    question, gr_ans),
            save_result(sid, "Mistral", question, m_ans),
        )

        await send_to_group(bot, sid, question, [f1, f2, f3], image_bytes)

        pending_sessions[sid] = {
            "user_id": user_id,
            "question": question,
            "time": time.time(),
        }
        return sid

    finally:
        stop_ev.set()
        if status_msg:
            spin.cancel()

# ==================================================

# LUU INPUT CHO /ok

# ==================================================

user_pending: dict = {}   # user_id -> {question, image_bytes}

# ==================================================

# HANDLERS

# ==================================================

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Xin chao!\n"
        "/help  -- Hoi AI (Gemini + Groq + Mistral)\n"
        "/ocr   -- Trich xuat de bai tu anh"
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_pending.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "Che do hoi AI\n\n"
        "1. Gui cau hoi hoac anh bai toan\n"
        "2. Go /ok de xac nhan -> bot gui 3 AI\n"
        "3. Nhan ket qua tong hop tot nhat\n\n"
        "Go /ocr neu chi muon trich xuat text tu anh",
        parse_mode="Markdown",
    )
    return S_INPUT

async def handle_text_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    q   = update.message.text.strip()
    user_pending[uid] = {"question": q, "image_bytes": None}
    await update.message.reply_text(
        f"Da nhan cau hoi:\n_{q[:300]}_\n\n"
        "Go /ok de gui, hoac gui lai de thay doi.",
        parse_mode="Markdown",
    )
    return S_OK

async def handle_photo_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
    caption = update.message.caption or "Hay giai bai toan trong anh nay."
    photo   = update.message.photo[-1]
    file    = await ctx.bot.get_file(photo.file_id)
    raw     = bytes(await file.download_as_bytearray())
    compressed = compress_image(raw)
    user_pending[uid] = {"question": caption, "image_bytes": compressed}
    await update.message.reply_text(
        f"Da nhan anh!\n"
        f"{caption[:200]}\n\n"
        "Go /ok de gui, hoac gui lai de thay doi.",
        parse_mode="Markdown",
    )
    return S_OK

async def cmd_ok(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    data = user_pending.pop(uid, None)
    if not data:
        await update.message.reply_text(
            "Chua co cau hoi! Go /help roi gui cau hoi/anh truoc."
        )
        return S_INPUT

    status = await update.message.reply_text("Dang khoi dong...")
    try:
        sid = await process_question(
            ctx.bot, uid,
            data["question"], data["image_bytes"],
            status,
        )
        await status.edit_text(
            f"Da gui den 3 AI!\n"
            f"Session: {sid}\n\n"
            "Dang cho Bot2 tong hop...",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.error(f"process_question loi: {e}", exc_info=True)
        await status.edit_text(f"Loi: {e}")
    return S_INPUT

# OCR

OCR_WAIT = "OCR_WAIT"

async def cmd_ocr(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Che do OCR\n\nGui anh bai toan de trich xuat toan bo text.",
        parse_mode="Markdown",
    )
    return OCR_WAIT

async def handle_ocr_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    photo  = update.message.photo[-1]
    file   = await ctx.bot.get_file(photo.file_id)
    raw    = bytes(await file.download_as_bytearray())
    img    = compress_image(raw)

    status   = await update.message.reply_text("Dang doc anh...")
    stop_ev  = asyncio.Event()
    spin     = asyncio.create_task(
        run_spinner(status, "Gemini dang doc de bai...", stop_ev)
    )
    try:
        result = await call_gemini_ocr(img)
    finally:
        stop_ev.set()
        spin.cancel()

    if len(result) <= 4000:
        try:
            await status.edit_text(
                f"Ket qua OCR:\n\n{result}",
                parse_mode="Markdown",
            )
        except Exception:
            await status.edit_text(result)
    else:
        await status.edit_text("Ket qua dai, gui duoi dang file:")
        buf = io.BytesIO(result.encode())
        buf.name = "ocr.md"
        await update.message.reply_document(document=buf, filename="ocr.md")

    return OCR_WAIT

# NHAN SUMMARY TU BOT2

async def handle_group_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != GROUP_ID:
        return
    doc     = update.message.document
    caption = update.message.caption or ""
    if not caption.startswith("SUMMARY|"):
        return

    parts = caption.split("|")
    if len(parts) < 2:
        return
    sid     = parts[1].strip()
    session = pending_sessions.pop(sid, None)
    if not session:
        log.warning(f"Session {sid} khong tim thay")
        return

    uid      = session["user_id"]
    question = session["question"]

    try:
        file      = await ctx.bot.get_file(doc.file_id)
        raw_bytes = bytes(await file.download_as_bytearray())
        summary   = raw_bytes.decode("utf-8", errors="replace")

        # Bang tom tat
        lines  = [l for l in summary.splitlines() if l.strip()]
        words  = len(summary.split())
        prev   = "\n".join(lines[:12])
        table  = (
            f"Ket qua tong hop\n"
            f"{'-'*32}\n"
            f"Session   : {sid}\n"
            f"Cau hoi   : {question[:150]}\n"
            f"So tu     : {words}\n"
            f"Thoi gian : {datetime.now():%H:%M:%S}\n"
            f"{'-'*32}\n"
            f"Noi dung chinh:\n"
            f"{prev[:700]}{'...' if len(prev)>700 else ''}"
        )

        # Gui file dinh kem truoc
        await ctx.bot.send_document(
            chat_id=uid,
            document=raw_bytes,
            filename=f"ket_qua_{sid}.txt",
            caption="File ket qua day du",
        )
        # Gui bang tom tat
        await ctx.bot.send_message(
            chat_id=uid,
            text=table,
            parse_mode="Markdown",
        )
        log.info(f"[{sid}] Da gui ket qua -> user {uid}")

    except Exception as e:
        log.error(f"Gui ket qua user loi: {e}", exc_info=True)
        await ctx.bot.send_message(chat_id=uid, text=f"Loi: {e}")

# DON SESSION HET HAN

async def cleanup(ctx: ContextTypes.DEFAULT_TYPE):
    now     = time.time()
    expired = [s for s, d in pending_sessions.items() if now - d["time"] > 1800]
    for s in expired:
        pending_sessions.pop(s, None)
    if expired:
        log.info(f"Xoa session het han: {len(expired)} session")

# ==================================================

# MAIN

# ==================================================

def main():
    app = Application.builder().token(BOT1_TOKEN).build()
    app.job_queue.run_repeating(cleanup, interval=600, first=60)

    # ConversationHandler chinh: /help -> input -> /ok
    conv = ConversationHandler(
        entry_points=[CommandHandler("help", cmd_help)],
        states={
            S_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input),
                MessageHandler(filters.PHOTO, handle_photo_input),
            ],
            S_OK: [
                CommandHandler("ok", cmd_ok),
                # Cho phep thay input truoc khi /ok
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input),
                MessageHandler(filters.PHOTO, handle_photo_input),
            ],
        },
        fallbacks=[
            CommandHandler("help", cmd_help),
            CommandHandler("ok", cmd_ok),
        ],
        per_chat=True,
        allow_reentry=True,
    )

    # ConversationHandler OCR
    ocr_conv = ConversationHandler(
        entry_points=[CommandHandler("ocr", cmd_ocr)],
        states={
            OCR_WAIT: [MessageHandler(filters.PHOTO, handle_ocr_photo)],
        },
        fallbacks=[CommandHandler("ocr", cmd_ocr)],
        per_chat=True,
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(conv)
    app.add_handler(ocr_conv)

    # Nhan SUMMARY tu Bot2
    app.add_handler(MessageHandler(
        filters.Document.ALL & filters.Chat(GROUP_ID),
        handle_group_summary,
    ))

    log.info("Bot1 dang chay...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()