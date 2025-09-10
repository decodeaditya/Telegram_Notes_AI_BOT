import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import json
import numpy as np

from telegram import Update, Audio, PhotoSize
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

from sentence_transformers import SentenceTransformer

# For voice and image processing
import speech_recognition as sr
from pydub import AudioSegment
import pytesseract
from PIL import Image

from supabase import create_client, Client

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import openai
openai.api_key = OPENAI_API_KEY

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

model = SentenceTransformer("all-MiniLM-L12-v2")


def get_user_name(user):
    return user.first_name or user.username or "friend"


def get_embedding(text):
    return model.encode(text)


async def ask_llm(messages, temperature=0.5):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
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


def save_note(uid, note):
    logger.info("Save note executed")
    data = {
        "user_id": str(uid),
        "text": note.get("text", ""),
        "embedding": note.get("embedding").tolist() if isinstance(note.get("embedding"), np.ndarray) else note.get("embedding", []),
        "priority": note.get("priority", "normal"),
        "subtasks": note.get("subtasks", []),
        "created_at": note.get("created_at") or datetime.utcnow().isoformat()
    }
    try:
        supabase.table("notes").insert(data).execute()
    except Exception as e:
        logger.error(f"Error saving note to supabase: {e}")


def get_history(uid, limit=10):
    try:
        response = supabase.table("chat_history")\
            .select("*")\
            .eq("user_id", str(uid))\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        data = response.data if response.data else []
        return list(reversed(data))
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return []


def update_chat_history(uid, role, content):
    data = {
        "user_id": str(uid),
        "role": role,
        "content": content,
        "created_at": datetime.utcnow().isoformat()
    }
    try:
        supabase.table("chat_history").insert(data).execute()
    except Exception as e:
        logger.error(f"Error updating chat history: {e}")


def fetch_all_notes(uid):
    try:
        response = supabase.table("notes")\
            .select("text")\
            .eq("user_id", str(uid))\
            .order("created_at", desc=True)\
            .execute()
        data = response.data if response.data else []
        return data
    except Exception as e:
        logger.error(f"Error fetching all notes: {e}")
        return []


async def summarize_notes_and_answer(uid, user_query):
    notes = fetch_all_notes(uid)
    if not notes:
        return "You have no notes yet. You can add some!"

    notes_text = "\n".join(f"- {n['text']}" for n in notes)
    prompt = [
        {"role": "system", "content": "You are a helpful assistant answering questions based on user notes."},
        {"role": "user", "content": f"Here are the user notes:\n{notes_text}\n\nUser query: {user_query}"}
    ]
    answer = await ask_llm(prompt)
    return answer or "I couldn't find an answer based on your notes."


async def classify_intent_and_parse(message, prev_notes):
    prompt = (f"You are a friendly productivity assistant. "
              f"Classify the following message into intents: add_task, add_note, remind, query, delete, or general_chat. "
              f"Also Check if the Message contains brain dump so first classify them based on that and treat them like different intents. "
              f"Respond only with JSON containing keys: intent, text, suggest_priority, subtasks (list), time_expression.\n\n"
              f"Message: \"{message}\" \nPrevious notes: {prev_notes[-3:] if prev_notes else '[]'}")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message}
    ]
    raw = await ask_llm(messages, temperature=0.3)
    raw = raw.replace("```json", "").replace("```", "").strip()
    logger.info(f"LLM raw response: {raw}")
    try:
        return json.loads(raw)
    except Exception:
        logger.warning(f"LLM parsing fallback: {raw}")
        return {"intent": "general_chat", "text": message}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Handling message")
    uid = update.effective_user.id
    now = datetime.utcnow()
    msg = update.message

    # Handle voice
    if msg.voice:
        file = await context.bot.get_file(msg.voice.file_id)
        file_path = f"audio_{uid}.oga"
        await file.download_to_drive(file_path)
        transcribed = convert_and_transcribe(file_path)
        if transcribed:
            await msg.reply_text(f"Got your voice note! Transcribed: {transcribed}")
            update_chat_history(uid, "user", transcribed)
            msg.text = transcribed
        else:
            await msg.reply_text("Sorry, couldn't understand your voice note!")
        return

    # Handle photo
    if msg.photo:
        file = await context.bot.get_file(msg.photo[-1].file_id)
        file_path = f"photo_{uid}.jpg"
        await file.download_to_drive(file_path)
        extracted = extract_image_text(file_path)
        await msg.reply_text(f"Extracted from image: {extracted.strip()[:300]}")
        update_chat_history(uid, "user", extracted)
        msg.text = extracted

    text = msg.text
    update_chat_history(uid, "user", text)

    prev_notes = fetch_all_notes(uid)

    result = await classify_intent_and_parse(text, prev_notes)
    intent = result.get("intent", "general_chat")
    logger.info(f"Classified intent: {intent}")

    if intent in {"add_note", "add_task"}:
        note = {
            "text": result.get("text", text),
            "embedding": get_embedding(result.get("text", text)),
            "created_at": now.isoformat(),
            "priority": result.get("suggest_priority", "normal"),
            "subtasks": result.get("subtasks", [])
        }
        save_note(uid, note)
        resp = f"Added your {intent.replace('_', ' ')}: “{note['text']}” "
        if note["priority"]:
            resp += f"(Priority: {note['priority']}) "
        if note["subtasks"]:
            resp += f"\nSubtasks: {', '.join(note['subtasks'])}"
        await msg.reply_text(resp)

    elif intent == "remind":
        reminder_time = result.get("time_expression") or "1 hour"
        note = {
            "text": result.get("text", text),
            "created_at": now.isoformat(),
            "reminder": reminder_time,
            "embedding": get_embedding(result.get("text", text))
        }
        save_note(uid, note)
        await msg.reply_text(
            f"Reminder set for: \"{note['text']}\" (when: {reminder_time}, smarter time handling coming soon!)"
        )

    elif intent == "query":
        answer = await summarize_notes_and_answer(uid, text)
        await msg.reply_text(answer or "Couldn't find anything matching that! Add something first?")

    elif intent == "delete":
        idx = int(''.join([d for d in str(text) if d.isdigit()])) - 1
        try:
            response = supabase.table("notes").select("*").eq("user_id", str(uid)).execute()
            notes_list = response.data if response.data else []
            if 0 <= idx < len(notes_list):
                note_id = notes_list[idx]["id"]
                supabase.table("notes").delete().eq("id", note_id).execute()
                await msg.reply_text(f"Deleted: {notes_list[idx]['text']}")
            else:
                await msg.reply_text("Index out of range for notes.")
        except Exception as e:
            logger.error(f"Error deleting note: {e}")
            await msg.reply_text("Couldn't delete that, try 'delete 1' or 'delete last'")

    elif intent == "general_chat":
        chat_msgs = get_history(uid, limit=8)
        messages = [{"role": "system", "content": (
            "Act as a smart, supportive friend who manages notes & reminders. "
            "Use recent conversations as context. Be empathetic, casual, and funny as appropriate."
        )}] + chat_msgs
        resp = await ask_llm(messages, temperature=0.8)
        await msg.reply_text(resp or "I'm still here to help you!")

    # Proactive suggestions
    notes_count = len(fetch_all_notes(uid))
    if notes_count > 2:
        suggestion = await predictive_suggestions(uid)
        if suggestion:
            await msg.reply_text(f"Suggestion: {suggestion}")


async def predictive_suggestions(uid):
    try:
        response = supabase.table("notes")\
            .select("text")\
            .eq("user_id", str(uid))\
            .order("created_at", desc=True).limit(5).execute()
        notes = response.data if response.data else []
        if not notes:
            return ""
        prompt = "Suggest the next task or reminder based on these notes:\n" + "\n".join(n['text'] for n in notes)
        messages = [{"role": "system", "content": prompt}]
        return await ask_llm(messages, temperature=0.6)
    except Exception as e:
        logger.error(f"Error fetching notes for predictive suggestion: {e}")
        return ""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    name = get_user_name(update.effective_user)
    await update.message.reply_text(
        f"Hey {name} 👋! I'm your AI productivity buddy. "
        "Just type, talk (voice note), or send a photo—I’ll organize your tasks, notes, ideas, and reminders, and answer follow-ups like a friend."
    )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(context.error)
    if update and update.message:
        await update.message.reply_text("Oops, something went wrong. I'm still here for you!")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    app.add_error_handler(error_handler)
    app.run_polling()


if __name__ == "__main__":
    main()
