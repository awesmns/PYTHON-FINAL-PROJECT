"""User accounts for PowerWise: SQLite + salted PBKDF2 password hashes + login lock-out.

Passwords are never stored. For each user we keep a random salt and the
PBKDF2-HMAC-SHA256 hash (200,000 iterations).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "powerwise.db")
ITERATIONS = 200_000
MAX_ATTEMPTS = 5
LOCK_SECONDS = 300
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")


class AuthError(Exception):
    """Message is safe to show to the user."""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


@contextmanager
def _db(db_path: str | None = None):
    """Open the database; ALWAYS commit (even when an AuthError is raised inside, so the
    failed-attempt counter is saved) and close it."""
    con = _connect(db_path)
    try:
        yield con
    finally:
        con.commit()
        con.close()


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def init_db(db_path: str | None = None, seed_demo: bool = True) -> None:
    with _db(db_path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, full_name TEXT NOT NULL, salt BLOB NOT NULL, pw_hash BLOB NOT NULL,
            created REAL NOT NULL, failed INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0)""")
        n = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if seed_demo and n == 0:
        register("demo", "Demo User", "demo123", db_path)


def validate_password(password: str) -> None:
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters.")
    if password.isdigit() or password.isalpha():
        raise AuthError("Password must contain both letters and numbers.")


def register(username: str, full_name: str, password: str, db_path: str | None = None) -> None:
    username, full_name = username.strip(), full_name.strip()
    if not USERNAME_RE.match(username):
        raise AuthError("Username must be 3-20 characters: letters, numbers or underscore.")
    if not full_name:
        raise AuthError("Please enter your name.")
    validate_password(password)
    salt = secrets.token_bytes(16)
    try:
        with _db(db_path) as con:
            con.execute("INSERT INTO users (username, full_name, salt, pw_hash, created) VALUES (?,?,?,?,?)",
                        (username.lower(), full_name, salt, _hash(password, salt), time.time()))
    except sqlite3.IntegrityError:
        raise AuthError("That username is already taken.") from None


def login(username: str, password: str, db_path: str | None = None) -> dict:
    """Return {'username', 'full_name'} on success, otherwise raise AuthError."""
    username = username.strip().lower()
    generic = AuthError("Incorrect username or password.")
    with _db(db_path) as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            _hash(password, b"0" * 16)                     # same work as a real check (hides which names exist)
            raise generic
        if row["locked_until"] > time.time():
            mins = int((row["locked_until"] - time.time()) // 60) + 1
            raise AuthError(f"Too many failed attempts. Try again in about {mins} minute(s).")
        if hmac.compare_digest(_hash(password, row["salt"]), row["pw_hash"]):
            con.execute("UPDATE users SET failed=0, locked_until=0 WHERE username=?", (username,))
            return {"username": row["username"], "full_name": row["full_name"]}
        failed = row["failed"] + 1
        lock = time.time() + LOCK_SECONDS if failed >= MAX_ATTEMPTS else 0
        con.execute("UPDATE users SET failed=?, locked_until=? WHERE username=?",
                    (0 if lock else failed, lock, username))
        if lock:
            raise AuthError(f"Too many failed attempts. Account locked for {LOCK_SECONDS // 60} minutes.")
        raise generic


def change_password(username: str, old: str, new: str, db_path: str | None = None) -> None:
    login(username, old, db_path)                           # re-checks the old password
    validate_password(new)
    salt = secrets.token_bytes(16)
    with _db(db_path) as con:
        con.execute("UPDATE users SET salt=?, pw_hash=? WHERE username=?", (salt, _hash(new, salt), username.lower()))
