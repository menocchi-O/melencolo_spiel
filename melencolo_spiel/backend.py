# C version for translate_align
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import torch
import itertools
import random
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

selected_pairs = []
tokenizer = None
model = None
device = None

MODEL_NAME = "aneuraz/awesome-align-with-co"
#MODEL_NAME = "google/madlad400-3b-mt"
app = FastAPI(title="Word Alignment API")
# --------- MODEL LOADING ---------- #
@app.on_event("startup")
def load_model():
    global tokenizer, model, translator, device
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    # MODEL_NAME, device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = pipeline(
     "translation_de_to_en",
     model="Helsinki-NLP/opus-mt-de-en",
     device=0 if torch.cuda.is_available() else -1
    )
    model.to(device)
    model.eval()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
@app.get("/")
def index():
       return FileResponse("static/jammer.html")

def build_game(selected_pairs):
    english_words = [p["en"] for p in selected_pairs]

    game = []

    for pair in selected_pairs:

        correct = pair["en"]

        # All possible distractors except the correct one
        distractors = [w for w in english_words if w != correct]

        # Pick two random distractors
        wrong = random.sample(distractors, k=min(2, len(distractors)))

        # Build the options and shuffle them
        options = wrong + [correct]
        random.shuffle(options)

        game.append({
            "de": pair["de"],
            "en": options,
            "correct": correct
        })

    return game

# 🔥 CORS — MUST be here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # OK for dev
    allow_credentials=True,
    allow_methods=["*"],        # <-- THIS enables OPTIONS
    allow_headers=["*"],
)

ALIGN_LAYER = 8
THRESHOLD = 0.05

class AlignRequest(BaseModel):
    text: str


class WordPair(BaseModel):
    de: str
    en: str
    confidence: float = 0.0
    source_indices: List[int] = Field(default_factory=list)
    target_indices: List[int] = Field(default_factory=list)


class AlignResponse(BaseModel):
    translation: str
    pairs: List[WordPair]

def translate_sentence(src: str) -> str:
    return translator(src)[0]["translation_text"]

def tokenize_words(text: str):
    return text.strip().split()

def get_alignment_embeddings(words, tokenizer, model, device):

    ...

def calculate_word_alignment(src_data, tgt_data):

    ...

def build_word_pairs(src_words, tgt_words, alignment):

    ...

def align_words(src, tgt):

    return pairs

def align_sentence(src: str):
    # ---------------------------------------------------------
    # 1. Translate
    # ---------------------------------------------------------
    tgt = translate_sentence(src)
    # ---------------------------------------------------------
    # 2. Tokenize each word separately
    # ---------------------------------------------------------
    src_words = tokenize_words(src)
    tgt_words = tokenize_words(tgt)
    # ---------------------------------------------------------
    # 3. Flatten
    # ---------------------------------------------------------
    src_data = get_alignment_embeddings(src_words,...)
    tgt_data = get_alignment_embeddings(tgt_words,...)
    # ---------------------------------------------------------
    # 4. alignment
    # ---------------------------------------------------------
    alignment = calculate_word_alignment(src_data, tgt_data)

    return tgt, build_word_pairs(src_words, tgt_words, alignment)

### DASHBOARD ###
@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest):
    translation, pairs = align_sentence(req.text)
    return {
        "translation": translation,
        "pairs": pairs
    }

@app.post("/save_pairs")
async def save_pairs(data: dict):
    global selected_pairs
    selected_pairs = data["pairs"]
    return {"status": "ok"}

@app.get("/game_words")
async def game_words():
    return build_game(selected_pairs)
