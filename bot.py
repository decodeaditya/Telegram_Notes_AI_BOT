import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import openai
import json
import numpy as np


from telegram import Update, Audio, PhotoSize
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)


from sentence_transformers import SentenceTransformer


# For voice and image
import speech_recognition as sr
from pydub import AudioSegment
import pytesseract
from PIL import Image


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BOT_TOKEN = os.getenv("TELEGRAM_BOT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


model = SentenceTransformer("all-MiniLM-L12-v2")
user_data = {}  # In production: persist to DB
user_chat_history = {}


# -- Helper functions (NLP, storage, multimodal) --


def get_user_name(user):
    return user.first_name or user.username or "friend"


def get_embedding(text):
    return model.encode(text)


def save_note(uid, note):
    user_data.setdefault(uid, []).append(note)


async def ask_llm(messages, temperature=0.5):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Use latest/greatest LLM
            messages=messages,
            temperature=temperature,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


def extract_image_text(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)


def convert_and_transcribe(voice_file):
    ogg_audio = AudioSegment.from_file(voice_file)
    wav_path = voice_file.replace('.oga', '.wav')
    ogg_audio.export(wav_path, format="wav")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        logger.warning(f"Voice recognition error: {e}")
        return None


# -- Conversational memory (short window for context) --
def update_chat_history(uid, role, content, limit=10):
    hist = user_chat_history.setdefault(uid, [])
    hist.append({"role": role, "content": content})
    if len(hist) > limit:
        hist.pop(0)
    return hist


def get_history(uid):
    return user_chat_history.get(uid, [])


# -- Intent Extraction using powerful LLM prompt --
async def classify_intent_and_parse(message, prev_notes):
    prompt = (f"You are a friendly productivity assistant. "
              f"Classify the following message into intents: add_task, add_note, remind, query, delete, or general_chat. "
              f"If it's a new note or task, suggest priority/deadline, extract subtasks if any. "
              f"Respond in JSON with all fields: intent, text, suggest_priority, subtasks (list), time_expression.\\n\\n"
              f"Message: \"{message}\" \\nPrevious notes: {prev_notes[-3:] if prev_notes else 'None'}")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message}
    ]
    raw = await ask_llm(messages, temperature=0.3)
    try:
        return json.loads(raw)
    except Exception:
        logger.warning(f"LLM parsing fallback: {raw}")
        return {"intent": "general_chat", "text": message}


# -- NLP search for notes --
def find_best_notes(uid, query):
    notes = user_data.get(uid, [])
    if not notes:
        return []
    query_emb = get_embedding(query)
    sims = [
        (np.dot(query_emb, n["embedding"]) / (np.linalg.norm(query_emb) * np.linalg.norm(n["embedding"])), n)
        for n in notes
    ]
    sims = sorted(sims, reverse=True, key=lambda x: x[0])
    return [n for s, n in sims[:5] if s > 0.3]  # Lower threshold for flexible NLP


# -- Predictive suggestions --
async def predictive_suggestions(uid):
    notes = user_data.get(uid, [])
    if not notes:
        return ""
    prompt = "Suggest the next task or reminder based on these notes:\\n" + "\\n".join(n['text'] for n in notes[-5:])
    messages = [{"role": "system", "content": prompt}]
    suggestion = await ask_llm(messages, 0.6)
    return suggestion


# -- Handler for text, images, and audio --
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    name = get_user_name(update.effective_user)
    now = datetime.now()
    msg = update.message

    if msg.voice:
        file = await context.bot.get_file(msg.voice.file_id)
        file_path = f"audio_{uid}.oga"
        await file.download_to_drive(file_path)
        transcribed = convert_and_transcribe(file_path)
        if transcribed:
            await msg.reply_text(f"Got your voice note! Transcribed: {transcribed}")
            update_chat_history(uid, "user", transcribed)
            # You can recursively call handle_message with text now
            msg.text = transcribed
        else:
            await msg.reply_text("Sorry, couldn't understand your voice note!")
        return

    if msg.photo:
        file = await context.bot.get_file(msg.photo[-1].file_id)
        file_path = f"photo_{uid}.jpg"
        await file.download_to_drive(file_path)
        extracted = extract_image_text(file_path)
        await msg.reply_text(f"Extracted from image: {extracted.strip()[:300]}")
        update_chat_history(uid, "user", extracted)
        # Optionally treat as note
        msg.text = extracted

    text = msg.text
    update_chat_history(uid, "user", text)
    chat_hist = get_history(uid)
    prev_notes = user_data.get(uid, [])

    # ------------- ADVANCED INTENT ANALYSIS ----------------------------------
    result = await classify_intent_and_parse(text, prev_notes)
    intent = result.get("intent")

    # --------- Handle advanced features via LLM-driven flow ------------------
    if intent in {"add_note", "add_task"}:
        note = {
            "text": result.get("text", text),
            "embedding": get_embedding(result.get("text", text)),
            "timestamp": now.isoformat(),
            "priority": result.get("suggest_priority", "normal"),
            "subtasks": result.get("subtasks", [])
        }
        save_note(uid, note)
        resp = f"Added your {intent.replace('_', ' ')}: “{note['text']}” "
        if note["priority"]:
            resp += f"(Priority: {note['priority']}) "
        if note["subtasks"]:
            resp += f"\\nSubtasks: {', '.join(note['subtasks'])}"
        await msg.reply_text(resp)
    elif intent == "remind":
        # For demo, simply reply; in prod use APScheduler
        reminder_time = result.get("time_expression") or "1 hour"
        note = {
            "text": result.get("text", text),
            "timestamp": now.isoformat(),
            "reminder": reminder_time,
            "embedding": get_embedding(result.get("text", text))
        }
        save_note(uid, note)
        await msg.reply_text(
            f"Reminder set for: \"{note['text']}\" (when: {reminder_time}, smarter time handling coming soon!)"
        )
    elif intent == "query":
        notes = find_best_notes(uid, result.get("text", text))
        if not notes:
            await msg.reply_text("Couldn't find anything matching that! Add something first?")
        else:
            summarized = "\\n".join(f"- {n['text']}" for n in notes)
            # LLM-powered answer (summary, suggestions, etc.)
            summary_msg = [{"role": "system", "content": f"Summarize and answer based on these notes: {summarized}"}]
            summary = await ask_llm(summary_msg)
            await msg.reply_text(f"{summary}\\n(Show more? / Suggest more details?)")
    elif intent == "delete":
        idx = int(''.join([d for d in str(text) if d.isdigit()])) - 1
        try:
            removed = user_data[uid].pop(idx)
            await msg.reply_text(f"Deleted: {removed['text']}")
        except Exception:
            await msg.reply_text("Couldn't delete that, try 'delete 1' or 'delete last'")
    elif intent == "general_chat":
        # Conversational, always LLM-powered, uses context
        messages = get_history(uid)[-8:]  # Recent messages for memory
        messages.insert(0, {"role": "system", "content": (
            "Act as a smart, supportive friend who manages notes & reminders. "
            "Use recent conversations as context. Be empathetic, casual, and funny where appropriate."
        )})
        resp = await ask_llm(messages, 0.8)
        await msg.reply_text(resp or "I'm still here to help you!")

    # ----- Proactive: Suggest next actions occasionally ------
    if len(user_data.get(uid, [])) > 2:
        suggestion = await predictive_suggestions(uid)
        if suggestion:
            await msg.reply_text(f"Suggestion: {suggestion}")


# --- Telegram IVR/Voice/Image etc. handlers are all managed by above ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    name = get_user_name(update.effective_user)
    user_data[uid] = []
    user_chat_history[uid] = []
    greeting = (f"Hey {name} 👋! I'm your AI productivity buddy. "
                "Just type, talk (voice note), or send a photo—I'll organize your tasks, notes, ideas, and reminders, and answer follow-ups like a friend. "
                "Try: 'remind me to schedule doctor next week' or send a voice memo! 🚀")
    await update.message.reply_text(greeting)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(context.error)
    if update and update.message:
        await update.message.reply_text("Oops, something went wrong. I'm still here for you!")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    app.add_error_handler(error_handler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
