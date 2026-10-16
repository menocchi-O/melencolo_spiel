import json
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint,
    Identity, Index, Integer, String, Text, UniqueConstraint, create_engine,
    func, select
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]
# Development-only shortcut. In a real app, derive this from the authenticated user.
TEST_USER_ID = int(os.environ["TEST_USER_ID"])

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class WordPair(Base):
    __tablename__ = "word_pairs"
    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uq_word_pairs_id_user"),
        ForeignKeyConstraint(
            ["parent_id", "user_id"], ["word_pairs.id", "word_pairs.user_id"],
            ondelete="CASCADE", name="fk_word_pairs_parent"
        ),
        CheckConstraint(
            "(de IS NULL AND en IS NULL) OR (de IS NOT NULL AND en IS NOT NULL)",
            name="chk_word_pairs_translation"
        ),
        CheckConstraint("parent_id IS NULL OR parent_id <> id", name="chk_not_own_parent"),
        Index("idx_word_pairs_user", "user_id"),
        Index("idx_word_pairs_parent", "user_id", "parent_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    de: Mapped[str | None] = mapped_column(Text, nullable=True)
    en: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    game_pairs: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class WordPairCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    de: str | None = None
    en: str | None = None
    note: str | None = None
    parent_id: int | None = Field(default=None, gt=0)
    game_pairs: list[dict[str, Any]] | None = None
    sort_order: int = 0

    @model_validator(mode="after")
    def translations_together(self):
        if (self.de is None) != (self.en is None):
            raise ValueError("Provide both de and en, or leave both null for a container.")
        return self


class WordPairOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    parent_id: int | None
    title: str
    de: str | None
    en: str | None
    note: str | None
    game_pairs: list[dict[str, Any]] | None
    sort_order: int
    created_at: datetime
    updated_at: datetime


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Creates tables for this self-contained prototype. Use Alembic migrations in production.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Vocabulary DB transmission demo", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(select(1))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database connection failed") from exc


@app.post("/api/word-pairs", response_model=WordPairOut, status_code=201)
def create_word_pair(payload: WordPairCreate, db: Session = Depends(get_db)):
    # This is a local prototype: make sure the development user exists.
    # Do not expose TEST_USER_ID as a production identity mechanism.
    user_exists = db.scalar(select(User.id).where(User.id == TEST_USER_ID))
    if user_exists is None:
        raise HTTPException(
            status_code=400,
            detail=f"Development user id {TEST_USER_ID} does not exist. See README setup.",
        )

    if payload.parent_id is not None:
        parent = db.scalar(
            select(WordPair).where(
                WordPair.id == payload.parent_id,
                WordPair.user_id == TEST_USER_ID,
            )
        )
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent node not found for this user.")

    item = WordPair(
        user_id=TEST_USER_ID,
        parent_id=payload.parent_id,
        title=payload.title,
        de=payload.de,
        en=payload.en,
        note=payload.note,
        game_pairs=payload.game_pairs,
        sort_order=payload.sort_order,
    )
    db.add(item)
    try:
        db.commit()
        db.refresh(item)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database rejected the record.") from exc
    return item


@app.get("/api/word-pairs", response_model=list[WordPairOut])
def list_word_pairs(db: Session = Depends(get_db)):
    records = db.scalars(
        select(WordPair)
        .where(WordPair.user_id == TEST_USER_ID)
        .order_by(WordPair.parent_id.nullsfirst(), WordPair.sort_order, WordPair.id)
    ).all()
    return list(records)
