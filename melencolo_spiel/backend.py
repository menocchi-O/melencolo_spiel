# B version for translate_align
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


def align_sentence(src: str):
    # ---------------------------------------------------------
    # 1. Translate
    # ---------------------------------------------------------
    tgt = translator(src)[0]["translation_text"]

    print("GERMAN: ", src)
    print("ENGLISH:", tgt)

    src_words = src.strip().split()
    tgt_words = tgt.strip().split()

    if not src_words or not tgt_words:
        return tgt, []

    # ---------------------------------------------------------
    # 2. Tokenize each word separately
    # ---------------------------------------------------------
    token_src = [tokenizer.tokenize(w) for w in src_words]
    token_tgt = [tokenizer.tokenize(w) for w in tgt_words]

    wid_src = [
        tokenizer.convert_tokens_to_ids(tokens)
        for tokens in token_src
    ]

    wid_tgt = [
        tokenizer.convert_tokens_to_ids(tokens)
        for tokens in token_tgt
    ]

    # ---------------------------------------------------------
    # 3. Flatten
    # ---------------------------------------------------------
    flat_src = list(itertools.chain(*wid_src))
    flat_tgt = list(itertools.chain(*wid_tgt))

    ids_src = tokenizer.prepare_for_model(
        flat_src,
        return_tensors="pt",
        truncation=True
    )["input_ids"].to(device)

    ids_tgt = tokenizer.prepare_for_model(
        flat_tgt,
        return_tensors="pt",
        truncation=True
    )["input_ids"].to(device)

    # ---------------------------------------------------------
    # 4. Map subwords -> words
    # ---------------------------------------------------------
    sub2word_src = []

    for word_idx, tokens in enumerate(token_src):
        sub2word_src.extend([word_idx] * len(tokens))

    sub2word_tgt = []

    for word_idx, tokens in enumerate(token_tgt):
        sub2word_tgt.extend([word_idx] * len(tokens))

    # ---------------------------------------------------------
    # 5. Contextual embeddings
    # ---------------------------------------------------------
    with torch.no_grad():

        src_output = model(
            ids_src.unsqueeze(0),
            output_hidden_states=True
        )

        tgt_output = model(
            ids_tgt.unsqueeze(0),
            output_hidden_states=True
        )

        h_src = src_output.hidden_states[ALIGN_LAYER][0, 1:-1]
        h_tgt = tgt_output.hidden_states[ALIGN_LAYER][0, 1:-1]

        # Normalize => cosine similarity
        h_src = torch.nn.functional.normalize(
            h_src, p=2, dim=-1
        )

        h_tgt = torch.nn.functional.normalize(
            h_tgt, p=2, dim=-1
        )

        scores = torch.matmul(h_src, h_tgt.T)

    # ---------------------------------------------------------
    # 6. Aggregate subword similarities into word similarities
    # ---------------------------------------------------------
    word_scores = {}

    for src_sub_idx, src_word_idx in enumerate(sub2word_src):

        for tgt_sub_idx, tgt_word_idx in enumerate(sub2word_tgt):

            key = (src_word_idx, tgt_word_idx)

            score = scores[
                src_sub_idx,
                tgt_sub_idx
            ].item()

            if key not in word_scores:
                word_scores[key] = []

            word_scores[key].append(score)

    # Average subword scores
    for key in word_scores:
        word_scores[key] = (
            sum(word_scores[key])
            / len(word_scores[key])
        )

    # ---------------------------------------------------------
    # 7. Find best candidates in BOTH directions
    # ---------------------------------------------------------
    best_src_to_tgt = {}
    best_tgt_to_src = {}

    for src_idx in range(len(src_words)):

        candidates = []

        for tgt_idx in range(len(tgt_words)):

            score = word_scores.get(
                (src_idx, tgt_idx),
                -1.0
            )

            candidates.append((score, tgt_idx))

        candidates.sort(reverse=True)

        if candidates:
            best_src_to_tgt[src_idx] = candidates[0]

    for tgt_idx in range(len(tgt_words)):

        candidates = []

        for src_idx in range(len(src_words)):

            score = word_scores.get(
                (src_idx, tgt_idx),
                -1.0
            )

            candidates.append((score, src_idx))

        candidates.sort(reverse=True)

        if candidates:
            best_tgt_to_src[tgt_idx] = candidates[0]

    # ---------------------------------------------------------
    # 8. Mutual alignment
    #
    # A pair is considered strong when:
    #
    #   German -> English chooses English
    #   AND
    #   English -> German chooses German
    #
    # This eliminates many accidental associations.
    # ---------------------------------------------------------

    MIN_SCORE = 0.25
    RELATIVE_SCORE = 0.80

    alignments = []

    for src_idx, src_word in enumerate(src_words):

        if src_idx not in best_src_to_tgt:
            continue

        best_score, best_tgt_idx = best_src_to_tgt[src_idx]

        if best_score < MIN_SCORE:
            continue

        # Check mutuality
        reverse = best_tgt_to_src.get(best_tgt_idx)

        if reverse is None:
            continue

        reverse_score, reverse_src_idx = reverse

        if reverse_src_idx != src_idx:
            continue

        # -----------------------------------------------------
        # Find additional English words, but only if they are
        # very close to the best candidate.
        # -----------------------------------------------------

        candidates = []

        for tgt_idx in range(len(tgt_words)):

            score = word_scores.get(
                (src_idx, tgt_idx),
                -1.0
            )

            if score < MIN_SCORE:
                continue

            if score >= best_score * RELATIVE_SCORE:
                candidates.append(
                    (tgt_idx, score)
                )

        # Always include the best candidate
        if best_tgt_idx not in [x[0] for x in candidates]:
            candidates.append(
                (best_tgt_idx, best_score)
            )

        # Sort by English position
        candidates.sort(key=lambda x: x[0])

        selected_target_indices = [
            idx for idx, score in candidates
        ]

        english_phrase = " ".join(
            tgt_words[i]
            for i in selected_target_indices
        )

        alignments.append({
            "de": src_word,
            "en": english_phrase,
            "confidence": round(best_score, 4),
            "source_indices": [src_idx],
            "target_indices": selected_target_indices,
        })

    return tgt, alignments    

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
