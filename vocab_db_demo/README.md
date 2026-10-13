# Vocabulary database transmission test

A minimal runnable prototype for **HTML → FastAPI → PostgreSQL → FastAPI → HTML**.

## 1. Create a local database

Create the database in PostgreSQL:

```sql
CREATE DATABASE vocabulary_db;
```

Then create the development user. The application creates the `users` and `word_pairs` tables on startup:

```sql
\c vocabulary_db

INSERT INTO users (username, email, password_hash)
VALUES ('demo', 'demo@example.test', 'DEVELOPMENT_ONLY_NOT_A_REAL_PASSWORD_HASH');
```

The placeholder hash is only to satisfy the schema for this local data-transmission test. Do not use it for login. A real registration endpoint must hash passwords with Argon2id (or another suitable password-hashing scheme).

If your PostgreSQL username/password differs, set `DATABASE_URL` before starting the server. Example:

```bash
export DATABASE_URL='postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/vocabulary_db'
export TEST_USER_ID=1
```

On Windows PowerShell:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/vocabulary_db"
$env:TEST_USER_ID = "1"
```

## 2. Install and run

Use Python 3.10 or newer:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000**. Do not open `index.html` directly; serving it through FastAPI avoids browser-origin issues.

## 3. Test the transmission

1. Fill in the title, German and English text, and optionally a note.
2. Click **Send to database**.
3. The status should show the new database ID.
4. The page reloads the records from PostgreSQL and renders the response.
5. Refresh the browser and click **Load saved records** to verify persistence.
6. Visit **http://127.0.0.1:8000/docs** to inspect and call the API directly.

For a container node, leave both German and English fields empty. For a translation pair, provide both.

The `game_pairs` field accepts JSON, e.g.:

```json
[
  {"de": "German original text", "en": "English original translation", "order": 1},
  {"de": "Another German text", "en": "Its English translation", "order": 2}
]
```

## API routes

- `GET /api/health` — test database connectivity
- `POST /api/word-pairs` — validate and save one node
- `GET /api/word-pairs` — read saved nodes back from PostgreSQL

## Important limitations

- This is a local development prototype, not a production-ready authenticated app.
- For simplicity, the current user is configured with `TEST_USER_ID`. Replace this with the authenticated user ID from a verified session/token before deploying.
- `create_all()` is convenient for a prototype. Use Alembic migrations for a production schema.
- Parent ownership is checked. Longer-cycle prevention should be implemented in a dedicated hierarchy-move endpoint before adding move/edit operations.
- The prototype doesn't include registration/login, update/delete endpoints, or Phaser integration yet.
