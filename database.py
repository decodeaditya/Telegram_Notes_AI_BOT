import os
import logging
import numpy as np
from supabase import create_client, Client

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_note(user_id, content, embedding, category="Uncategorized", reminder_time=None):
    data = {
        "user_id": user_id,
        "content": content,
        "embedding": embedding,
        "category": category,
        "reminder_time": reminder_time,
    }
    response = supabase.table("notes").insert(data).execute()
    if response.error:
        logger.error(f"Error inserting note: {response.error.message}")
        return None
    return response.data[0]


def get_notes(user_id):
    response = supabase.table("notes").select("*").eq("user_id", user_id).execute()
    if response.error:
        logger.error(f"Error fetching notes: {response.error.message}")
        return []
    return response.data


def delete_note(note_id):
    response = supabase.table("notes").delete().eq("id", note_id).execute()
    if response.error:
        logger.error(f"Error deleting note {note_id}: {response.error.message}")
        return False
    return True
