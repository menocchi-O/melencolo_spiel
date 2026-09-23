# C version for translate_align
import token
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
import spacy

selected_pairs = []
tokenizer = None
model = None
device = None
nlp_de = spacy.load("de_core_news_sm")

CHUNK_THRESHOLD = 80
ALIGN_LAYER = 8
THRESHOLD = 1e-3

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
def split_into_sentences(text):
    doc = nlp_de(text)
    return [sent.text.strip() for sent in doc.sents]
def chunk_text(text):
    words = text.split()

    # Short text: don't touch it
    if len(words) <= CHUNK_THRESHOLD:
        return [text.strip()]

    # Long text: sentence segmentation
    doc = nlp_de(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())

        # Sentence itself is small enough
        if current_size + sentence_size <= CHUNK_THRESHOLD:
            current.append(sentence)
            current_size += sentence_size

        else:
            if current:
                chunks.append(" ".join(current))

            current = [sentence]
            current_size = sentence_size

    if current:
        chunks.append(" ".join(current))

    return chunks

# 🔥 CORS — MUST be here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # OK for dev
    allow_credentials=True,
    allow_methods=["*"],        # <-- THIS enables OPTIONS
    allow_headers=["*"],
)


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

def split_words(txt: str):
    return txt.strip().split()

def tokenize_words(words, tokenizer):
    return [tokenizer.tokenize(w) for w in words]

def get_alignment_embeddings(tokens, tokenizer, device):
    # tokenizer.convert_tokens_to_ids(...) turns: ["Das", "Haus", "ist"] into: [1234, 5678, 9012] || It's important not to interpret 1234 as some semantic representation of "Das". It's just an ID
    wid = [tokenizer.convert_tokens_to_ids(x) for x in tokens]
    # prepare_for_model() is preparing those IDs to be fed into the Transformer.Depending on the model/tokenizer, this can involve things such as adding special tokens.
    return tokenizer.prepare_for_model(                                                      
        list(itertools.chain(*wid)),
        return_tensors="pt",                 # it means: "Give me the result as a PyTorch tensor."
        truncation=True
    )["input_ids"].to(device)

def sub2word(tokens):
    ### RE-ASSOCIATES THE SET OF SUBWORDS TO EACH WORD
    sub2word = []
    for i, w in enumerate(tokens):
        sub2word += [i] * len(w)
    return sub2word

def extract_layer_info(ids):
    ### THE ACTUAL VECTOR REPRESENTATION ###
    with torch.no_grad():
        return model(ids.unsqueeze(0), output_hidden_states=True)[2][ALIGN_LAYER][0, 1:-1]

def calculate_word_alignment(src_vec, tgt_vec, src_sub2w, tgt_sub2w):
    scores = torch.matmul(src_vec, tgt_vec.T)
    s2t = torch.softmax(scores, dim=-1)
    t2s = torch.softmax(scores, dim=-2)
    mask = (s2t > THRESHOLD) & (t2s > THRESHOLD)
    aligned = set()
    for i, j in torch.nonzero(mask):
               aligned.add((src_sub2w[i], tgt_sub2w[j]))
    return aligned

def build_word_pairs(src_words, tgt_words, alignment):
    return [
        {"de": src_words[i], "en": tgt_words[j]}
        for i, j in sorted(alignment)
    ]

def align_sentence(src: str):
    # ---------------------------------------------------------
    # 0. Translate
    # ---------------------------------------------------------
    tgt = translate_sentence(src)
    # ---------------------------------------------------------
    # 1. Split
    # ---------------------------------------------------------
    src_words = split_words(src)
    tgt_words = split_words(tgt)
    # ---------------------------------------------------------
    # 2. Tokenize each word separately
    # ---------------------------------------------------------
    src_tokens = tokenize_words(src_words, tokenizer)
    tgt_tokens = tokenize_words(tgt_words, tokenizer)
    # ---------------------------------------------------------
    # 3. Flatten
    # ---------------------------------------------------------
    src_ids = get_alignment_embeddings(src_tokens, tokenizer, device)
    tgt_ids = get_alignment_embeddings(tgt_tokens, tokenizer, device)
    # ---------------------------------------------------------
    # 4. Actual Vector Representation
    # ---------------------------------------------------------
    vec_src = extract_layer_info(src_ids)
    vec_tgt = extract_layer_info(tgt_ids)
    # ---------------------------------------------------------
    # 5. Re-Alignment Subwords -> Word
    # ---------------------------------------------------------
    sub2word_src = sub2word(src_tokens)
    sub2word_tgt = sub2word(tgt_tokens)
    # ---------------------------------------------------------
    # 6. alignment
    # ---------------------------------------------------------
    alignment = calculate_word_alignment(vec_src, vec_tgt, sub2word_src, sub2word_tgt)

    return tgt, build_word_pairs(src_words, tgt_words, alignment)

def align_text(src: str):

    chunks = chunk_text(src)

    all_pairs = []
    translations = []

    for chunk in chunks:
        print("************************* SENTENCE ***********************************")
        print(chunk)
        print("**********************************************************************")
        translation, pairs = align_sentence(chunk)

        translations.append(translation)
        all_pairs.extend(pairs)

    return " ".join(translations), all_pairs

### DASHBOARD ###
@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest):
    translation, pairs = align_text(req.text)
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

# Newer tasks to accomplish:
#     - text fragmentation into smaller chunks before translating/processing;
#     - detect trennbare verben
#     - sentence/construct level alignment