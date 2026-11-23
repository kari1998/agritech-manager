import os
import re
import secrets
import time
from datetime import date
from pathlib import Path
from typing import Literal, Optional
from typing import Optional
from pydantic import BaseModel
from database import init_db

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel

from database import init_db, hash_password, verify_password, get_connection

# All files are found relative to this file, so the app works no matter
# which folder the server is started from.
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Load .env from the exact same folder as main.py
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# AI features are optional: without a key the rest of the app still works,
# and the AI endpoints return a clear 503 instead of a confusing error.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
if ai_client is None:
    print("WARNING: GEMINI_API_KEY is not set in .env - AI chat and fertilizer calculator are disabled.")

def require_ai_client():
    if ai_client is None:
        raise HTTPException(
            status_code=503,
            detail="AI features are not configured. Add GEMINI_API_KEY to the .env file and restart the server.",
        )
    return ai_client


# Models tried in order. Mixing model families helps when one of them is overloaded.
AI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-4-hydrogen"]
AI_BUSY_CODES = {429, 500, 503, 504}   # overloaded / rate limited / temporary server trouble
AI_KEY_CODES = {401, 403}              # bad or blocked API key: same for every model, so stop
AI_RETRY_ROUNDS = 2                    # go through the model list this many times
AI_RETRY_WAIT_SECONDS = 2              # pause between rounds


def generate_ai_reply(client, contents, config=None) -> str:
    """Asks Gemini for a reply, falling back across models and retrying briefly when
    Google is busy. Raises an HTTPException with a farmer-friendly message if all fail."""
    last_error = None
    all_busy = True
    for round_number in range(AI_RETRY_ROUNDS):
        if round_number > 0:
            time.sleep(AI_RETRY_WAIT_SECONDS)
        for model_name in AI_MODELS:
            try:
                response = client.models.generate_content(model=model_name, contents=contents, config=config)
                if response.text:
                    return response.text
                last_error = f"{model_name} returned an empty reply"
                all_busy = False
            except genai_errors.APIError as e:
                print(f"AI model {model_name} failed: {e.code} {e.status}")   # full detail stays in the server log
                last_error = f"{e.code} {e.status}"
                if e.code in AI_KEY_CODES:
                    raise HTTPException(
                        status_code=502,
                        detail="The AI service rejected the API key. Check GEMINI_API_KEY in the .env file.",
                    )
                if e.code not in AI_BUSY_CODES:
                    all_busy = False
            except Exception as e:   # network problems, timeouts, ...
                print(f"AI model {model_name} failed: {e!r}")
                last_error = type(e).__name__
                all_busy = False

    if all_busy:
        raise HTTPException(
            status_code=503,
            detail="The AI assistant is very busy right now (Google's servers are under high demand). "
                   "Please wait a minute and try again.",
        )
    raise HTTPException(status_code=502, detail=f"The AI assistant could not answer right now ({last_error}). Please try again.")

app = FastAPI(title="AgriTech Manager API")

# No CORS middleware: the pages are served by this same app, so the browser
# never makes cross-origin requests. Leaving CORS off means other websites
# can't call this API with a logged-in farmer's session cookie.

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

init_db()   # creates any missing tables on startup; safe to run every time

SESSION_COOKIE = "agritech_session"
SESSION_MAX_AGE = 7 * 24 * 60 * 60   # 7 days, in seconds
# Set COOKIE_SECURE=true in .env once the site is served over HTTPS,
# so the browser only ever sends the login cookie over an encrypted connection.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# One shared connection helper (defined in database.py) for the whole app
get_db = get_connection


def read_page(filename: str) -> str:
    return (STATIC_DIR / filename).read_text(encoding="utf-8")

def get_current_user_id(request: Request) -> int:
    """Returns the logged-in farmer's user_id from the session cookie, or raises 401."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in.")
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT user_id FROM sessions
            WHERE token = ? AND created_at > datetime('now', '-7 days')
            """,
            (token,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    return row["user_id"]


def require_self(path_user_id: int, current_user_id: int):
    """Blocks access to another farmer's account via /api/.../{user_id} URLs."""
    if path_user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You can only access your own account.")


def require_cycle_owner(cycle_id: int, user_id: int):
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT 1 FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE cc.cycle_id = ? AND p.user_id = ?
        """, (cycle_id, user_id)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=403, detail="That crop cycle does not belong to your account.")


def require_chat_session_owner(cursor, session_id: int, user_id: int):
    """Raises 404 unless the chat session exists and belongs to this farmer."""
    cursor.execute(
        "SELECT 1 FROM chat_sessions WHERE session_id = ? AND user_id = ?",
        (session_id, user_id),
    )
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Chat session not found for this account.")


def require_plot_owner(plot_id: int, user_id: int):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT 1 FROM plots WHERE plot_id = ? AND user_id = ?", (plot_id, user_id)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=403, detail="That plot does not belong to your account.")


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    location: str = "Kenya"
    phone_number: Optional[str] = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class ProfileUpdateRequest(BaseModel):
    name: str
    location: str
    phone_number: Optional[str] = ""

class LogRequest(BaseModel):
    cycle_id: int
    log_type: str
    description: str
    log_date: date

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None   # None = start a new chat session

class PlotRequest(BaseModel):
    plot_name: str
    size_acres: float
    soil_type: Optional[str] = ""

class CropCycleRequest(BaseModel):
    plot_id: int
    crop_name: str
    planting_date: date
    status: Literal["Active", "Harvested", "Completed"] = 'Active'

class FertilizerRequest(BaseModel):
    cycle_id: int

class FinancialRequest(BaseModel):
    cycle_id: int
    transaction_type: str
    category: str
    amount: float
    transaction_date: date
    notes: Optional[str] = ""

class HarvestRequest(BaseModel):
    cycle_id: int
    harvest_date: date
    quantity_kg: float
    storage_location: Optional[str] = "Main Store"
    notes: Optional[str] = ""

class HarvestSaleRequest(BaseModel):
    harvest_id: int
    sold_kg: float
    amount: float
    sale_date: date
    category: str = "Harvest Sale"
    notes: Optional[str] = ""

class TransactionCreateRequest(BaseModel):
    cycle_id: int
    type: str
    amount: float
    category: Optional[str] = ""
    description: Optional[str] = ""
    date: Optional[str] = None
    harvest_id: Optional[int] = None
    quantity_kg: Optional[float] = None

class CreateChatSessionRequest(BaseModel):
    title: Optional[str] = "New Agronomist Chat"

@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/login", response_class=HTMLResponse)
def login_page():
    return read_page("login.html")

@app.get("/register", response_class=HTMLResponse)
def register_page():
    return read_page("register.html")

@app.get("/account", response_class=HTMLResponse)
def account_page():
    return read_page("account.html")

@app.post("/api/auth/register")
def register_farmer(req: RegisterRequest):
    email_clean = req.email.strip().lower()
    name_clean = req.name.strip()
    location_clean = req.location.strip() or "Kenya"
    phone_clean = (req.phone_number or "").strip()

    if not name_clean or not email_clean or len(req.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Name, valid email, and a password of at least 6 characters are required.",
        )
    if not EMAIL_PATTERN.match(email_clean):
        raise HTTPException(status_code=400, detail="Please enter a valid email address, e.g. name@example.com.")

    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = ?", (email_clean,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="An account with this email already exists.")

        cursor.execute(
            """
            INSERT INTO users (name, email, phone_number, location, password_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name_clean, email_clean, phone_clean, location_clean, hash_password(req.password)),
        )
        conn.commit()
        user_id = cursor.lastrowid
    finally:
        conn.close()

    return {
        "status": "success",
        "user": {
            "user_id": user_id,
            "name": name_clean,
            "email": email_clean,
            "phone_number": phone_clean,
            "location": location_clean,
        },
    }

@app.post("/api/auth/login")
def login_farmer(req: LoginRequest, response: Response):
    email_clean = req.email.strip().lower()
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id, name, email, phone_number, location, password_hash
            FROM users
            WHERE LOWER(email) = ?
            """,
            (email_clean,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = secrets.token_urlsafe(32)
    conn = get_db()
    try:
        # Housekeeping: drop expired sessions so the table doesn't grow forever
        conn.execute("DELETE FROM sessions WHERE created_at <= datetime('now', '-7 days')")
        conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, row["user_id"]))
        conn.commit()
    finally:
        conn.close()

    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,      # page scripts can't read or change it
        samesite="lax",     # browser won't send it on form posts from other sites
        max_age=SESSION_MAX_AGE,
        secure=COOKIE_SECURE,   # HTTPS-only once COOKIE_SECURE=true is set in .env
    )

    return {
        "status": "success",
        "user": {
            "user_id": row["user_id"],
            "name": row["name"],
            "email": row["email"],
            "phone_number": row["phone_number"] or "",
            "location": row["location"] or "Kenya",
        },
    }

@app.post("/api/auth/logout")
def logout_farmer(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        conn = get_db()
        try:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        finally:
            conn.close()
    response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="lax", secure=COOKIE_SECURE)
    return {"status": "success"}

@app.put("/api/users/{user_id}")
def update_profile(user_id: int, req: ProfileUpdateRequest, current_user_id: int = Depends(get_current_user_id)):
    require_self(user_id, current_user_id)
    name_clean = req.name.strip()
    location_clean = req.location.strip() or "Kenya"
    phone_clean = (req.phone_number or "").strip()

    if not name_clean:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Farmer not found.")

        cursor.execute(
            """
            UPDATE users
            SET name = ?, location = ?, phone_number = ?
            WHERE user_id = ?
            """,
            (name_clean, location_clean, phone_clean, user_id),
        )
        conn.commit()

        cursor.execute(
            "SELECT user_id, name, email, phone_number, location FROM users WHERE user_id = ?",
            (user_id,),
        )
        updated = dict(cursor.fetchone())
    finally:
        conn.close()

    return {"status": "success", "user": updated}

@app.get("/financial", response_class=HTMLResponse)
def financials_page():
    return read_page("financials.html")

@app.get("/harvests", response_class=HTMLResponse)
def harvests_page():
    return read_page("harvests.html")

@app.get("/api/dashboard/{user_id}")
def get_dashboard(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    require_self(user_id, current_user_id)
    conn = get_db()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COALESCE(SUM(fl.amount), 0.0)
            FROM financial_ledger fl
            JOIN crop_cycles cc ON fl.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ? AND fl.transaction_type = 'Income'
        """, (user_id,))
        total_revenue = cursor.fetchone()[0] or 0.0

        cursor.execute("""
            SELECT COALESCE(SUM(fl.amount), 0.0)
            FROM financial_ledger fl
            JOIN crop_cycles cc ON fl.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ? AND fl.transaction_type = 'Expense'
        """, (user_id,))
        total_expenses = cursor.fetchone()[0] or 0.0

        net_profit = total_revenue - total_expenses

        # 5 most recent transactions for the dashboard summary
        # (the Financials page uses /api/financials/{user_id} for the full ledger)
        cursor.execute("""
            SELECT fl.*, cc.crop_name, p.plot_name
            FROM financial_ledger fl
            JOIN crop_cycles cc ON fl.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ?
            ORDER BY fl.transaction_date DESC, fl.transaction_id DESC LIMIT 5
        """, (user_id,))
        transactions = [dict(row) for row in cursor.fetchall()]

        # User profile (never send the password hash to the browser)
        cursor.execute(
            "SELECT user_id, name, email, phone_number, location, created_at FROM users WHERE user_id = ?",
            (user_id,),
        )
        user_row = cursor.fetchone()
        user = dict(user_row) if user_row else {"name": "Farmer", "location": "Kenya"}

        cursor.execute("SELECT * FROM plots WHERE user_id = ? ORDER BY plot_id", (user_id,))
        plots = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT cc.*, p.plot_name, p.soil_type, p.size_acres
            FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ?
            ORDER BY cc.cycle_id
        """, (user_id,))
        crop_cycles = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT fl.*, cc.crop_name
            FROM farm_logs fl
            JOIN crop_cycles cc ON fl.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ?
            ORDER BY fl.log_date DESC, fl.log_id DESC
        """, (user_id,))
        logs = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT
                cc.cycle_id,
                cc.crop_name,
                cc.status AS cycle_status,
                p.plot_id,
                p.plot_name,
                COALESCE(SUM(CASE WHEN LOWER(fl.transaction_type) IN ('income', 'revenue') THEN fl.amount ELSE 0 END), 0) AS total_revenue,
                COALESCE(SUM(CASE WHEN LOWER(fl.transaction_type) = 'expense' THEN fl.amount ELSE 0 END), 0) AS total_expenses,
                COALESCE(SUM(CASE WHEN LOWER(fl.transaction_type) IN ('income', 'revenue') THEN fl.amount ELSE 0 END), 0) -
                COALESCE(SUM(CASE WHEN LOWER(fl.transaction_type) = 'expense' THEN fl.amount ELSE 0 END), 0) AS net_profit
            FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            LEFT JOIN financial_ledger fl ON fl.cycle_id = cc.cycle_id
            WHERE p.user_id = ?
            GROUP BY cc.cycle_id, cc.crop_name, cc.status, p.plot_id, p.plot_name
            ORDER BY p.plot_name ASC, cc.crop_name ASC
            """,
            (user_id,),
        )
        crop_profitability = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT hi.*, cc.crop_name, p.plot_name
            FROM harvest_inventory hi
            JOIN crop_cycles cc ON hi.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ?
            ORDER BY hi.harvest_date DESC, hi.harvest_id DESC
        """, (user_id,))
        harvests = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    total_acreage = sum(p["size_acres"] or 0 for p in plots)
    active_crops = len([c for c in crop_cycles if (c["status"] or "").lower() == "active"])

    return {
        "user": user,
        "plots": plots,
        "crop_cycles": crop_cycles,
        "logs": logs,
        "harvests": harvests,
        "metrics": {
            "total_acreage": round(total_acreage, 2),
            "plots_count": len(plots),
            "active_crops_count": active_crops,
            "total_logs_count": len(logs)
        },
        "financials": {
            "revenue": total_revenue,
            "expenses": total_expenses,
            "net_profit": net_profit,
            "transactions": transactions,
            "crop_profitability": crop_profitability
        }
    }

@app.post("/api/logs")
def create_log(log: LogRequest, current_user_id: int = Depends(get_current_user_id)):
    if not log.description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")
    require_cycle_owner(log.cycle_id, current_user_id)
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO farm_logs (cycle_id, log_type, description, log_date)
            VALUES (?, ?, ?, ?)
        """, (log.cycle_id, log.log_type, log.description.strip(), log.log_date.isoformat()))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success", "message": "Farm log added successfully"}

@app.get("/api/chat/history")
def get_chat_history(current_user_id: int = Depends(get_current_user_id)):
    """Returns the logged-in farmer's last 50 chat messages, oldest first."""
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT role, message, timestamp FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC LIMIT 50
        """, (current_user_id,)).fetchall()
    finally:
        conn.close()
    return {"messages": [dict(r) for r in reversed(rows)]}

@app.post("/api/chat")
def chat_with_agronomist(chat: ChatRequest, current_user_id: int = Depends(get_current_user_id)):
    client = require_ai_client()
    user_id = current_user_id
    message = chat.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name, location FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Farmer not found")

        cursor.execute("""
            SELECT p.plot_name, p.size_acres, p.soil_type, cc.crop_name, cc.planting_date, cc.status
            FROM plots p
            LEFT JOIN crop_cycles cc ON p.plot_id = cc.plot_id
            WHERE p.user_id = ?
        """, (user_id,))
        context_rows = cursor.fetchall()

        # Previous conversation in this chat session only. The new message is saved
        # further down, together with the AI's reply, and only if the AI call succeeds.
        past_messages = []
        if chat.session_id is not None:
            require_chat_session_owner(cursor, chat.session_id, user_id)
            cursor.execute("""
                SELECT role, content FROM chat_messages
                WHERE session_id = ?
                ORDER BY message_id DESC LIMIT 9
            """, (chat.session_id,))
            past_messages = cursor.fetchall()
    finally:
        conn.close()

    past_messages.reverse()   # oldest first, the order the model expects

    contents = []
    for row in past_messages:
        role = "user" if row["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=row["content"])]
        ))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    context_text = f"Farmer: {user['name']} located in {user['location']}\nPlots & Crops:\n"
    for row in context_rows:
        context_text += f"- Plot: {row['plot_name']} ({row['size_acres']} acres, {row['soil_type']} soil) | Crop: {row['crop_name'] or 'None'} (Planted: {row['planting_date'] or 'N/A'}, Status: {row['status'] or 'N/A'})\n"

    system_instruction = (
        "You are an expert Kenyan agricultural agronomist chatbot. "
        "Provide practical, region-aware farming advice based strictly on the farmer's database context below. "
        "Keep answers concise, professional, and actionable.\n\n"
        f"DATABASE CONTEXT:\n{context_text}"
    )

    ai_reply = generate_ai_reply(
        client,
        contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )

    conn = get_db()
    try:
        cursor = conn.cursor()
        session_id = chat.session_id
        if session_id is None:
            # First message of a new chat: name the session after the question
            title = message if len(message) <= 60 else message[:57] + "..."
            cursor.execute(
                "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
                (user_id, title),
            )
            session_id = cursor.lastrowid

        cursor.executemany(
            "INSERT INTO chat_messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            [(session_id, user_id, "user", message), (session_id, user_id, "model", ai_reply)],
        )
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = DATETIME('now') WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
    finally:
        conn.close()

    return {"response": ai_reply, "session_id": session_id}

@app.get("/api/plots")
def get_plots(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    require_self(user_id, current_user_id)
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM plots WHERE user_id = ? ORDER BY plot_id", (user_id,)).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]

@app.post("/api/plots")
def create_plot(plot: PlotRequest, current_user_id: int = Depends(get_current_user_id)):
    name_clean = plot.plot_name.strip()
    if not name_clean:
        raise HTTPException(status_code=400, detail="Plot name is required.")
    if plot.size_acres <= 0:
        raise HTTPException(status_code=400, detail="Plot size must be greater than zero.")

    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO plots (user_id, plot_name, size_acres, soil_type)
            VALUES (?, ?, ?, ?)
        """, (current_user_id, name_clean, plot.size_acres, (plot.soil_type or "").strip()))
        conn.commit()
        plot_id = cursor.lastrowid
    finally:
        conn.close()
    return {"status": "success", "message": "Plot added successfully", "plot_id": plot_id}

@app.post("/api/crop-cycles")
def create_crop_cycle(cycle: CropCycleRequest, current_user_id: int = Depends(get_current_user_id)):
    if not cycle.crop_name.strip():
        raise HTTPException(status_code=400, detail="Crop name is required.")
    require_plot_owner(cycle.plot_id, current_user_id)
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO crop_cycles (plot_id, crop_name, planting_date, status)
            VALUES (?, ?, ?, ?)
        """, (cycle.plot_id, cycle.crop_name.strip(), cycle.planting_date.isoformat(), cycle.status))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success", "message": "Crop cycle assigned successfully"}

@app.post("/api/calculate-fertilizer")
def calculate_fertilizer(req: FertilizerRequest, current_user_id: int = Depends(get_current_user_id)):
    client = require_ai_client()
    require_cycle_owner(req.cycle_id, current_user_id)
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT cc.crop_name, p.size_acres, p.plot_name, p.soil_type
            FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE cc.cycle_id = ?
        """, (req.cycle_id,)).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Crop cycle not found.")

    crop_name, size_acres, plot_name, soil_type = row
    prompt = (
         f"Calculate the exact fertilizer requirements for growing {crop_name} on a {size_acres}-acre plot "
         f"named '{plot_name}' with {soil_type} soil in Kenya. "
         "Provide: 1) Recommended planting fertilizer (e.g., DAP) in kg or bags, "
         "2) Top-dressing fertilizer (e.g., CAN/Urea) in kg or bags, "
         "3) A brief, step-by-step application schedule. Keep it practical and concise."
    )

    return {"recommendation": generate_ai_reply(client, prompt)}

@app.post("/api/financials")
def create_financial_record(txn: FinancialRequest, current_user_id: int = Depends(get_current_user_id)):
    if txn.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero.")
    if txn.transaction_type not in ("Income", "Expense"):
        raise HTTPException(status_code=400, detail="Type must be 'Income' or 'Expense'.")
    if not txn.category.strip():
        raise HTTPException(status_code=400, detail="Category is required.")

    require_cycle_owner(txn.cycle_id, current_user_id)

    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO financial_ledger (cycle_id, transaction_type, category, amount, transaction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (txn.cycle_id, txn.transaction_type, txn.category, txn.amount, txn.transaction_date.isoformat(), txn.notes))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success", "message": "Financial transaction recorded successfully"}

@app.get("/api/financials/{user_id}")
def get_financial_ledger(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    require_self(user_id, current_user_id)
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fl.*, cc.crop_name, p.plot_name
            FROM financial_ledger fl
            JOIN crop_cycles cc ON fl.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE p.user_id = ?
            ORDER BY fl.transaction_date DESC, fl.transaction_id DESC
        """, (user_id,))
        transactions = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return {"transactions": transactions}

@app.delete("/api/transactions/{transaction_id}")
def delete_transaction(transaction_id: int, current_user_id: int = Depends(get_current_user_id)):
    conn = get_db()
    try:
        cursor = conn.cursor()
        # Only allow deleting a transaction that belongs to the logged-in farmer
        cursor.execute("""
            SELECT fl.transaction_id
            FROM financial_ledger fl
            JOIN crop_cycles cc ON fl.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE fl.transaction_id = ? AND p.user_id = ?
        """, (transaction_id, current_user_id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Transaction not found for this account.")

        cursor.execute("DELETE FROM financial_ledger WHERE transaction_id = ?", (transaction_id,))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success"}

@app.post("/api/harvests")
def create_harvest(harvest: HarvestRequest, current_user_id: int = Depends(get_current_user_id)):
    require_cycle_owner(harvest.cycle_id, current_user_id)
    if harvest.quantity_kg <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than zero.")

    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO harvest_inventory (cycle_id, harvest_date, quantity_kg, remaining_kg, storage_location, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            harvest.cycle_id,
            harvest.harvest_date.isoformat(),
            harvest.quantity_kg,
            harvest.quantity_kg,
            harvest.storage_location or "Main Store",
            harvest.notes or ""
        ))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success", "message": "Harvest recorded in inventory."}

@app.post("/api/harvests/sell")
def sell_harvest_stock(sale: HarvestSaleRequest, current_user_id: int = Depends(get_current_user_id)):
    if sale.sold_kg <= 0 or sale.amount <=0:
        raise HTTPException(status_code=400, detail="Sold weight and amount must be greater than zero.")

    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hi.cycle_id
            FROM harvest_inventory hi
            JOIN crop_cycles cc ON hi.cycle_id = cc.cycle_id
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE hi.harvest_id = ? AND p.user_id = ?
        """, (sale.harvest_id, current_user_id))
        batch = cursor.fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="Harvest batch not found.")

        # Check and subtract in one statement, so two sales can't both pass
        cursor.execute("""
            UPDATE harvest_inventory
            SET remaining_kg = ROUND(remaining_kg - ?, 2)
            WHERE harvest_id = ? AND remaining_kg >= ?
        """, (sale.sold_kg, sale.harvest_id, sale.sold_kg))

        if cursor.rowcount == 0:
            cursor.execute("SELECT remaining_kg FROM harvest_inventory WHERE harvest_id = ?", (sale.harvest_id,))
            left = cursor.fetchone()["remaining_kg"]
            raise HTTPException(status_code=400, detail=f"Only {left} kg remaining in stock for this batch.")

        cursor.execute("""
            INSERT INTO financial_ledger (cycle_id, transaction_type, category, amount, transaction_date, notes)
            VALUES (?, 'Income', ?, ?, ?, ?)
        """, (
            batch["cycle_id"],
            sale.category,
            sale.amount,
            sale.sale_date.isoformat(),
            sale.notes or f"Sold {sale.sold_kg} kg from Harvest #{sale.harvest_id}"
        ))

        cursor.execute("SELECT remaining_kg FROM harvest_inventory WHERE harvest_id = ?", (sale.harvest_id,))
        new_remaining = cursor.fetchone()["remaining_kg"]
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "success",
        "message": f"Recorded sale of {sale.sold_kg} kg and logged KSh {sale.amount:,.2f} income.",
        "remaining_kg": new_remaining
    }

@app.patch("/api/cycles/{cycle_id}/complete")
def complete_crop_cycle(cycle_id: int, user_id: int = Depends(get_current_user_id)):
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cc.cycle_id
            FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE cc.cycle_id = ? AND p.user_id = ?
            """,
            (cycle_id, user_id),
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Crop cycle not found for this account.")

        cursor.execute(
            "UPDATE crop_cycles SET status = 'Completed' WHERE cycle_id = ?",
            (cycle_id,),
        )
        conn.commit()
    finally:
        conn.close()
    return {"status": "success", "cycle_id": cycle_id}

@app.delete("/api/cycles/{cycle_id}")
def delete_crop_cycle(cycle_id: int, user_id: int = Depends(get_current_user_id)):
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cc.cycle_id
            FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE cc.cycle_id = ? AND p.user_id = ?
            """,
            (cycle_id, user_id),
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Crop cycle not found for this account.")

        cursor.execute("SELECT COUNT(*) FROM farm_logs WHERE cycle_id = ?", (cycle_id,))
        logs_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM harvest_inventory WHERE cycle_id = ?", (cycle_id,))
        harvests_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM financial_ledger WHERE cycle_id = ?", (cycle_id,))
        tx_count = cursor.fetchone()[0]

        if logs_count or harvests_count or tx_count:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a crop cycle that already has logs, harvests, or financial entries. Mark it as Completed instead.",
            )

        cursor.execute("DELETE FROM crop_cycles WHERE cycle_id = ?", (cycle_id,))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success"}

@app.delete("/api/plots/{plot_id}")
def delete_plot(plot_id: int, user_id: int = Depends(get_current_user_id)):
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT plot_id FROM plots WHERE plot_id = ? AND user_id = ?",
            (plot_id, user_id),
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Plot not found for this account.")

        cursor.execute("SELECT COUNT(*) FROM crop_cycles WHERE plot_id = ?", (plot_id,))
        if cursor.fetchone()[0] > 0:
            raise HTTPException(
                status_code=400,
                detail="Delete or clear this plot's crop cycles first before deleting the plot.",
            )

        cursor.execute("DELETE FROM plots WHERE plot_id = ?", (plot_id,))
        conn.commit()
    finally:
        conn.close()
    return {"status": "success"}

@app.post("/api/transactions")
def create_transaction(req: TransactionCreateRequest, current_user_id: int = Depends(get_current_user_id)):
    tx_type = (req.type or "").strip().lower()
    if tx_type not in ("income", "expense"):
        raise HTTPException(status_code=400, detail="Transaction type must be 'income' or 'expense'.")
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero.")

    conn = get_db()
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT cc.cycle_id
            FROM crop_cycles cc
            JOIN plots p ON cc.plot_id = p.plot_id
            WHERE cc.cycle_id= ? AND p.user_id = ?
            """,
            (req.cycle_id, current_user_id),
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Crop cycle not found for this account.")

        if tx_type == "income" and req.harvest_id is not None:
            if req.quantity_kg is None or req.quantity_kg <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Please enter the kilograms sold when linking income to harvest stock.",
                )

            cursor.execute(
                """
                SELECT hi.harvest_id, hi.quantity_kg
                FROM harvest_inventory hi
                JOIN crop_cycles cc ON hi.cycle_id = cc.cycle_id
                JOIN plots p ON cc.plot_id = p.plot_id
                WHERE hi.harvest_id = ? AND p.user_id = ?
                """,
                (req.harvest_id, current_user_id),
            )
            harvest_row = cursor.fetchone()
            if not harvest_row:
                raise HTTPException(status_code=404, detail="Harvest stock batch not found for this account.")

            current_kg = float(harvest_row["quantity_kg"] or 0)
            if req.quantity_kg > current_kg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient harvest stock: only {current_kg:g} kg available.",
                )

            cursor.execute(
                """
                UPDATE harvest_inventory
                SET quantity_kg = quantity_kg - ?
                WHERE harvest_id = ?
                """,
                (req.quantity_kg, req.harvest_id),
            )

            cursor.execute(
                """
                INSERT INTO finacial_ledger (cycle_id, type, amount, category, decription, tx_date)
                VALUES (?, ?, ?, ?, ?, COALESCE(?, DATE('now')))
                """,
                (
                    req.cycle_id,
                    tx_type,
                    req.amount,
                    (req.category or ("Harvest Sale" if req.harvest_id else "")).strip(),
                    (req.description or "").strip(),
                    req.date,
                ),
            )
            conn.commit()
            tx_id = cursor.lastrowid
    finally:
        conn.close()

    return{"status": "success", "transaction_id": tx_id}

@app.get("/api/chat/sessions")
def list_chat_sessions(current_user_id: int = Depends(get_current_user_id)):
    """The logged-in farmer's chat sessions, most recently used first."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                cs.session_id,
                cs.title,
                cs.created_at,
                cs.updated_at,
                COUNT(cm.message_id) AS message_count
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cm.session_id = cs.session_id
            WHERE cs.user_id = ?
            GROUP BY cs.session_id, cs.title, cs.created_at, cs.updated_at
            ORDER BY cs.updated_at DESC, cs.session_id DESC
            """,
            (current_user_id,),
        )
        sessions = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return {"sessions": sessions}

@app.post("/api/chat/sessions")
def create_chat_session(req: CreateChatSessionRequest, current_user_id: int = Depends(get_current_user_id)):
    title = (req.title or "").strip() or "New Agronomist Chat"
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_sessions (user_id, title)
            VALUES (?, ?)
            """,
            (current_user_id, title),
        )
        conn.commit()
        session_id = cursor.lastrowid
        cursor.execute(
            "SELECT session_id, title, created_at, updated_at FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        created = dict(cursor.fetchone())
    finally:
        conn.close()
    return {"session": created}

@app.get("/api/chat/sessions/{session_id}/messages")
def get_session_messages(session_id: int, current_user_id: int = Depends(get_current_user_id)):
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, title FROM chat_sessions WHERE session_id = ? AND user_id = ?",
            (session_id, current_user_id),
        )
        session_row = cursor.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Chat session not found for this account.")

        cursor.execute(
            """
            SELECT message_id, role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY message_id ASC
            """,
            (session_id,),
        )
        messages = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return {"session": dict(session_row), "messages": messages}