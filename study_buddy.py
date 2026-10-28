
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = "gemini-3.6-flash"

# Keep the database in the project directory, not a
# temporary directory or the current working directory.
DB_PATH = Path(__file__).resolve().parent / "studybuddy.db"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        'Missing API key. Run: set "GEMINI_API_KEY=YOUR_KEY"'
    )

client = genai.Client(api_key=api_key)


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------
# 2. Database connection and schema
# --------------------------------------------------

@contextmanager
def db_connection():
    """Open a database connection and always close it."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_database():
    """Create tables and indexes if they do not exist."""

    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL,
                target_days INTEGER NOT NULL
                    CHECK (target_days BETWEEN 1 AND 30),
                hours_per_day REAL NOT NULL
                    CHECK (hours_per_day > 0
                           AND hours_per_day <= 12),
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN (
                        'active', 'paused', 'completed'
                    )),
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            one_active_goal
            ON goals(status)
            WHERE status = 'active'
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                study_date TEXT NOT NULL,
                minutes_studied INTEGER NOT NULL
                    CHECK (minutes_studied BETWEEN 1 AND 1440),
                topics TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (goal_id)
                    REFERENCES goals(id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            sessions_by_goal_date
            ON study_sessions(goal_id, study_date)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_mastery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                mastery INTEGER NOT NULL
                    CHECK (mastery BETWEEN 1 AND 5),
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (goal_id)
                    REFERENCES goals(id),
                UNIQUE (goal_id, topic)
            )
        """)

    print(f"[Database] Ready: {DB_PATH.name}")


# --------------------------------------------------
# 3. Helper: find the active learning goal
# --------------------------------------------------

def get_active_goal(conn):
    return conn.execute("""
        SELECT *
        FROM goals
        WHERE status = 'active'
        ORDER BY id DESC
        LIMIT 1
    """).fetchone()


# --------------------------------------------------
# 4. TOOL: create and save a learning goal
# --------------------------------------------------

def create_study_schedule(
    goal: str,
    days: int,
    hours_per_day: float
) -> dict:
    """Create and persist a learning goal.

    Args:
        goal: Topic or skill the user wants to learn.
        days: Number of study days, from 1 to 30.
        hours_per_day: Daily study time, above 0 and up to 12.

    Returns:
        Saved goal details and total planned study hours.
    """
    goal = goal.strip()

    if not goal:
        raise ValueError("Goal cannot be empty.")

    if isinstance(days, bool) or not isinstance(days, int):
        raise ValueError("Days must be an integer.")

    if not 1 <= days <= 30:
        raise ValueError("Days must be between 1 and 30.")

    if isinstance(hours_per_day, bool) or not isinstance(
        hours_per_day, (int, float)
    ):
        raise ValueError("Daily hours must be a number.")

    if not 0 < hours_per_day <= 12:
        raise ValueError("Daily hours must be above 0 and at most 12.")

    with db_connection() as conn:
        # Preserve previous goals but pause the previous active one.
        conn.execute("""
            UPDATE goals
            SET status = 'paused'
            WHERE status = 'active'
        """)

        cursor = conn.execute("""
            INSERT INTO goals (
                goal, target_days, hours_per_day,
                status, created_at
            )
            VALUES (?, ?, ?, 'active', ?)
        """, (goal, days, hours_per_day, utc_now()))

        goal_id = cursor.lastrowid

    print(f"[Tool] Saved learning goal #{goal_id}")

    return {
        "goal_id": goal_id,
        "goal": goal,
        "duration_days": days,
        "hours_per_day": hours_per_day,
        "total_planned_hours": days * hours_per_day,
        "status": "active"
    }


# --------------------------------------------------
# 5. TOOL: record a completed study session
# --------------------------------------------------

def record_study_session(
    minutes_studied: int,
    topics: str,
    notes: str = "",
    study_date: str = ""
) -> dict:
    """Persist a study session for the active learning goal.

    Args:
        minutes_studied: Actual study time in minutes.
        topics: Topics studied, separated by commas if needed.
        notes: Optional difficulties, achievements, or observations.
        study_date: Optional date in YYYY-MM-DD format.
            Defaults to the computer's local date.

    Returns:
        The saved session details.
    """
    if isinstance(minutes_studied, bool) or not isinstance(
        minutes_studied, int
    ):
        raise ValueError("Study minutes must be an integer.")

    if not 1 <= minutes_studied <= 1440:
        raise ValueError("Study time must be between 1 and 1440 minutes.")

    topics = topics.strip()
    if not topics:
        raise ValueError("Please specify what you studied.")

    if not study_date:
        study_date = date.today().isoformat()

    try:
        parsed_date = date.fromisoformat(study_date)
        if parsed_date.isoformat() != study_date:
            raise ValueError
    except ValueError:
        raise ValueError("Date must use YYYY-MM-DD format.")

    with db_connection() as conn:
        active_goal = get_active_goal(conn)

        if active_goal is None:
            return {
                "saved": False,
                "message": (
                    "There is no active goal. "
                    "Create a learning goal first."
                )
            }

        cursor = conn.execute("""
            INSERT INTO study_sessions (
                goal_id, study_date, minutes_studied,
                topics, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            active_goal["id"],
            study_date,
            minutes_studied,
            topics,
            notes.strip(),
            utc_now()
        ))

        session_id = cursor.lastrowid

    print(f"[Tool] Saved study session #{session_id}")

    return {
        "saved": True,
        "session_id": session_id,
        "goal": active_goal["goal"],
        "study_date": study_date,
        "minutes_studied": minutes_studied,
        "topics": topics,
        "notes": notes.strip()
    }


# --------------------------------------------------
# 6. TOOL: save or update topic mastery
# --------------------------------------------------

def update_topic_mastery(
    topic: str,
    mastery: int,
    notes: str = ""
) -> dict:
    """Save a topic's current mastery rating.

    Args:
        topic: Topic name, such as Python loops.
        mastery: Rating from 1 (beginner) to 5 (confident).
        notes: Optional explanation of what needs improvement.

    Returns:
        The saved topic mastery.
    """
    topic = topic.strip()

    if not topic:
        raise ValueError("Topic cannot be empty.")

    if isinstance(mastery, bool) or not isinstance(mastery, int):
        raise ValueError("Mastery must be an integer.")

    if not 1 <= mastery <= 5:
        raise ValueError("Mastery must be between 1 and 5.")

    with db_connection() as conn:
        active_goal = get_active_goal(conn)

        if active_goal is None:
            return {
                "saved": False,
                "message": "Create a learning goal first."
            }

        # A repeated topic updates its rating instead of
        # creating duplicate rows.
        conn.execute("""
            INSERT INTO topic_mastery (
                goal_id, topic, mastery, notes, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(goal_id, topic)
            DO UPDATE SET
                mastery = excluded.mastery,
                notes = excluded.notes,
                updated_at = excluded.updated_at
        """, (
            active_goal["id"],
            topic,
            mastery,
            notes.strip(),
            utc_now()
        ))

    print(f"[Tool] Updated mastery: {topic} = {mastery}/5")

    return {
        "saved": True,
        "goal": active_goal["goal"],
        "topic": topic,
        "mastery": mastery,
        "notes": notes.strip()
    }


# --------------------------------------------------
# 7. TOOL: retrieve saved progress
# --------------------------------------------------

def get_progress() -> dict:
    """Retrieve saved goal, study totals, sessions and mastery.

    Returns:
        Persistent progress from the local database.
    """
    with db_connection() as conn:
        goal = get_active_goal(conn)

        if goal is None:
            return {
                "has_active_goal": False,
                "message": (
                    "No active learning goal exists. "
                    "Ask the user to create one."
                )
            }

        totals = conn.execute("""
            SELECT
                COUNT(*) AS session_count,
                COALESCE(SUM(minutes_studied), 0) AS total_minutes,
                COUNT(DISTINCT study_date) AS study_days
            FROM study_sessions
            WHERE goal_id = ?
        """, (goal["id"],)).fetchone()

        sessions = conn.execute("""
            SELECT study_date, minutes_studied, topics, notes
            FROM study_sessions
            WHERE goal_id = ?
            ORDER BY study_date DESC, id DESC
            LIMIT 10
        """, (goal["id"],)).fetchall()

        mastery_rows = conn.execute("""
            SELECT topic, mastery, notes, updated_at
            FROM topic_mastery
            WHERE goal_id = ?
            ORDER BY mastery ASC, topic ASC
        """, (goal["id"],)).fetchall()

    total_minutes = totals["total_minutes"]
    planned_minutes = goal["target_days"] * goal["hours_per_day"] * 60

    return {
        "has_active_goal": True,
        "goal_id": goal["id"],
        "goal": goal["goal"],
        "status": goal["status"],
        "target_days": goal["target_days"],
        "hours_per_day": goal["hours_per_day"],
        "planned_hours": round(planned_minutes / 60, 2),
        "total_study_minutes": total_minutes,
        "total_study_hours": round(total_minutes / 60, 2),
        "study_sessions": totals["session_count"],
        "distinct_study_days": totals["study_days"],
        "remaining_planned_minutes": max(
            0, round(planned_minutes - total_minutes)
        ),
        "recent_sessions": [dict(row) for row in sessions],
        "topic_mastery": [dict(row) for row in mastery_rows]
    }


# --------------------------------------------------
# 8. Initialize database and register agent tools
# --------------------------------------------------

initialize_database()

chat = client.chats.create(
    model=MODEL_NAME,
    config=types.GenerateContentConfig(
        system_instruction="""
You are StudyBuddy, a persistent learning coach.

You have four tools:
- create_study_schedule: create and save a learning goal.
- record_study_session: record actual completed study time.
- update_topic_mastery: save a mastery rating from 1 to 5.
- get_progress: retrieve saved progress from SQLite.

Rules:
1. When the user asks to create a study plan and gives
   the goal, number of days and daily hours, call
   create_study_schedule. Ask for missing information.
2. When the user says they studied or completed a session,
   record it with record_study_session. Ask for the duration
   or topic if essential information is missing.
3. When the user explicitly rates their understanding of a
   topic, use update_topic_mastery.
4. When the user asks what they studied, what they should
   study next, or how they are progressing, call get_progress.
5. Use actual returned database records when describing progress.
   Never invent completed sessions or mastery ratings.
6. A tool returning saved=False means nothing was saved.
   Explain that clearly.
7. Recommend revision for topics with mastery 1 or 2.
   Use the study history to suggest a practical next step.
8. Do not claim to remember a fact unless it is in the
   current conversation or retrieved saved progress.
9. Be encouraging, clear and specific.
10. Creating a new goal pauses the previous active goal.
""",
        tools=[
            create_study_schedule,
            record_study_session,
            update_topic_mastery,
            get_progress
        ]
    )
)


# --------------------------------------------------
# 9. Interactive chat
# --------------------------------------------------

print("\nStudyBuddy Persistent Learning Agent is ready!")
print("Database:", DB_PATH)
print("\nTry:")
print("- Create a 7-day Python plan, 2 hours daily.")
print("- I studied Python variables for 45 minutes.")
print("- My mastery of Python variables is 3 out of 5.")
print("- Show my progress.")
print("- What should I study next?")
print("- Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("StudyBuddy: Your saved progress will be here next time.")
        break

    if not question:
        continue

    try:
        response = chat.send_message(question)

        if response.text:
            print("\nStudyBuddy:", response.text, "\n")
        else:
            print("\nStudyBuddy returned no text. Please try again.\n")

    except Exception as error:
        print("\nSomething went wrong:", error)
        print("Check your API key, model, connection and API quota.\n")
