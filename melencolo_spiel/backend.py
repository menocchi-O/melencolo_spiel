from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import itertools
from transformers import AutoModel, AutoTokenizer, pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import FileResponse

app = FastAPI(title="Word Alignment API")

@app.get("/")
def index():
    return FileResponse("static/frontend.html")

# 🔥 CORS — MUST be here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # OK for dev
    allow_credentials=True,
    allow_methods=["*"],        # <-- THIS enables OPTIONS
    allow_headers=["*"],
)

# --------- MODEL LOADING (ONCE) ----------

MODEL_NAME = "aneuraz/awesome-align-with-co"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

translator = pipeline(
     "translation_de_to_en",
     model="Helsinki-NLP/opus-mt-de-en",
     device=0 if torch.cuda.is_available() else -1
 )

ALIGN_LAYER = 8
THRESHOLD = 1e-3

class AlignRequest(BaseModel):
    text: str


class WordPair(BaseModel):
    de: str
    en: str


class AlignResponse(BaseModel):
    pairs: List[WordPair]

def align_sentence(src: str):
    tgt = translator(src)[0]["translation_text"]

    src_words = src.strip().split()
    tgt_words = tgt.strip().split()

    token_src = [tokenizer.tokenize(w) for w in src_words]
    token_tgt = [tokenizer.tokenize(w) for w in tgt_words]

    wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
    wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

    ids_src = tokenizer.prepare_for_model(
        list(itertools.chain(*wid_src)),
        return_tensors="pt",
        truncation=True
    )["input_ids"].to(device)

    ids_tgt = tokenizer.prepare_for_model(
        list(itertools.chain(*wid_tgt)),
        return_tensors="pt",
        truncation=True
    )["input_ids"].to(device)

    sub2word_src = []
    for i, w in enumerate(token_src):
        sub2word_src += [i] * len(w)

    sub2word_tgt = []
    for i, w in enumerate(token_tgt):
        sub2word_tgt += [i] * len(w)

    with torch.no_grad():
        h_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][ALIGN_LAYER][0, 1:-1]
        h_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][ALIGN_LAYER][0, 1:-1]

        scores = torch.matmul(h_src, h_tgt.T)
        s2t = torch.softmax(scores, dim=-1)
        t2s = torch.softmax(scores, dim=-2)
        mask = (s2t > THRESHOLD) & (t2s > THRESHOLD)

    aligned = set()
    for i, j in torch.nonzero(mask):
        aligned.add((sub2word_src[i], sub2word_tgt[j]))

    return [
        {"de": src_words[i], "en": tgt_words[j]}
        for i, j in sorted(aligned)
    ]
@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest):
    pairs = align_sentence(req.text)
    return {"pairs": pairs}
