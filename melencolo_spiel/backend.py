# THIS WILL BE THE TRUE TEST ENVIRONMENT

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import itertools
import random
import spacy

from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# =========================================================
# GLOBALS
# =========================================================

selected_pairs = []

tokenizer = None
model = None
translator = None
device = None
nlp_de = None



MODEL_NAME = "aneuraz/awesome-align-with-co"

ALIGN_LAYER = 8

# Awesome-Align's demo uses a very low threshold.
ALIGN_THRESHOLD = 1e-3


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="Word Alignment API")


# =========================================================
# MODEL LOADING
# =========================================================

# -----------------------------------------------------
# German spaCy model
# -----------------------------------------------------

print("Loading German spaCy model...")

nlp_de = spacy.load("de_core_news_sm")

print("spaCy German model loaded.")
print("Models loaded.\n")

@app.on_event("startup")
def load_model():

    global tokenizer
    global model
    global translator
    global device

    print("\nLoading alignment model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Alignment device:", device)

    translator = pipeline(
        "translation_de_to_en",
        model="Helsinki-NLP/opus-mt-de-en",
        device=0 if torch.cuda.is_available() else -1
    )

    model.to(device)
    model.eval()

    print("Models loaded.\n")


# =========================================================
# STATIC FRONTEND
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="static", html=True),
    name="static"
)


@app.get("/")
def index():
    return FileResponse("static/jammer.html")


# =========================================================
# GAME
# =========================================================

def build_game(selected_pairs):

    english_words = [
        p["en"]
        for p in selected_pairs
    ]

    game = []

    for pair in selected_pairs:

        correct = pair["en"]

        distractors = [
            w
            for w in english_words
            if w != correct
        ]

        wrong = random.sample(
            distractors,
            k=min(2, len(distractors))
        )

        options = wrong + [correct]

        random.shuffle(options)

        game.append({
            "de": pair["de"],
            "en": options,
            "correct": correct
        })

    return game


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATA MODELS
# =========================================================

class AlignRequest(BaseModel):
    text: str


class WordPair(BaseModel):

    de: str
    en: str

    confidence: float = 0.0

    source_indices: List[int] = Field(
        default_factory=list
    )

    target_indices: List[int] = Field(
        default_factory=list
    )


class AlignResponse(BaseModel):

    translation: str

    pairs: List[WordPair]

# =========================================================
# INSPECTION API MODELS
# =========================================================

class InspectRequest(BaseModel):
    text: str


class InspectResponse(BaseModel):
    text: str
    words: List[Dict[str, Any]]
    dependency_candidates: List[Dict[str, Any]]


# =========================================================
# 1. TRANSLATION
# =========================================================

def translate_sentence(src: str) -> str:

    result = translator(src)

    return result[0]["translation_text"]


# =========================================================
# 2. WORD TOKENIZATION
# =========================================================

def tokenize_words(text: str):

    return text.strip().split()


# =========================================================
# 3. ALIGNMENT EMBEDDINGS
# =========================================================
#
# This function:
#
#   - tokenizes every word into subwords
#   - converts subwords into token IDs
#   - builds the model input
#   - obtains the hidden states
#   - returns the subword embeddings
#   - keeps a mapping from subwords -> original words
#
# It does NOT perform alignment.
#
# =========================================================

def get_alignment_embeddings(
    words,
    tokenizer,
    model,
    device
):

    # -----------------------------------------------------
    # 1. Tokenize every word separately
    # -----------------------------------------------------

    tokenized_words = [
        tokenizer.tokenize(word)
        for word in words
    ]

    # -----------------------------------------------------
    # 2. Convert subword tokens -> token IDs
    # -----------------------------------------------------

    word_token_ids = [
        tokenizer.convert_tokens_to_ids(tokens)
        for tokens in tokenized_words
    ]

    # -----------------------------------------------------
    # 3. Flatten
    #
    # Example:
    #
    #   ["zur"]       -> [123]
    #   ["Ruh"]       -> [456]
    #
    # becomes:
    #
    #   [123, 456]
    # -----------------------------------------------------

    flat_token_ids = list(
        itertools.chain.from_iterable(
            word_token_ids
        )
    )

    if not flat_token_ids:

        return {
            "embeddings": None,
            "sub2word": [],
            "tokenized_words": tokenized_words
        }

    # -----------------------------------------------------
    # 4. Add special tokens
    #
    # prepare_for_model() creates something like:
    #
    #   [CLS] token1 token2 ... tokenN [SEP]
    # -----------------------------------------------------

    input_ids = tokenizer.prepare_for_model(
        flat_token_ids,
        return_tensors="pt",
        truncation=True
    )["input_ids"]

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Ensure shape is:
    #
    #   [batch_size, sequence_length]
    #
    # BERT expects a batch dimension.
    # -----------------------------------------------------

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(device)

    # -----------------------------------------------------
    # 5. Map subwords -> original word
    # -----------------------------------------------------

    sub2word = []

    for word_idx, tokens in enumerate(
        tokenized_words
    ):

        sub2word.extend(
            [word_idx] * len(tokens)
        )

    # -----------------------------------------------------
    # 6. Run Awesome-Align model
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
    input_ids,
    output_hidden_states=True

)

    # -----------------------------------------------------
    # 7. Extract hidden states
    #
    # output.hidden_states[layer]
    #
    # has shape:
    #
    #   [batch, sequence, hidden_size]
    #
    # We remove:
    #
    #   [CLS]
    #   [SEP]
    # -----------------------------------------------------

    embeddings = output.hidden_states[
        ALIGN_LAYER
    ][0, 1:-1]

    # -----------------------------------------------------
    # Safety check
    #
    # The number of embeddings should correspond to the
    # number of flattened subword tokens.
    # -----------------------------------------------------

    if len(sub2word) != embeddings.shape[0]:

        raise RuntimeError(
            "Subword mapping and embedding count differ: "
            f"{len(sub2word)} mappings vs "
            f"{embeddings.shape[0]} embeddings."
        )

    return {
        "embeddings": embeddings,
        "sub2word": sub2word,
        "tokenized_words": tokenized_words
    }

# =========================================================
# WORD-LEVEL DEBUG STRUCTURES
# =========================================================

@dataclass
class WordEmbedding:
    index: int
    text: str
    embedding: torch.Tensor


# =========================================================
# WORD-LEVEL CONTEXTUAL EMBEDDINGS
# =========================================================
#
# This stage is completely independent from spaCy.
#
# Example:
#
#   Ich will nicht schlafen gehen
#
#   0 Ich
#   1 will
#   2 nicht
#   3 schlafen
#   4 gehen
#
# Each original word receives ONE contextual embedding.
#
# =========================================================

def get_word_level_embeddings(
    words,
    tokenizer,
    model,
    device
):

    tokenized_words = [
        tokenizer.tokenize(word)
        for word in words
    ]

    word_token_ids = [
        tokenizer.convert_tokens_to_ids(tokens)
        for tokens in tokenized_words
    ]

    flat_token_ids = []

    subword_to_word = []

    for word_idx, token_ids in enumerate(
        word_token_ids
    ):

        flat_token_ids.extend(token_ids)

        subword_to_word.extend(
            [word_idx] * len(token_ids)
        )

    if not flat_token_ids:
        return []

    input_ids = tokenizer.prepare_for_model(
        flat_token_ids,
        return_tensors="pt",
        truncation=True
    )["input_ids"]

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(device)

    with torch.no_grad():

        output = model(
            input_ids,
            output_hidden_states=True
        )

    subword_embeddings = output.hidden_states[
        ALIGN_LAYER
    ][0, 1:-1]

    if len(subword_to_word) != subword_embeddings.shape[0]:

        raise RuntimeError(
            "Subword mapping and embedding count differ: "
            f"{len(subword_to_word)} mappings vs "
            f"{subword_embeddings.shape[0]} embeddings."
        )

    word_embeddings = []

    for word_idx, word in enumerate(words):

        subword_indices = [
            i
            for i, mapped_word_idx
            in enumerate(subword_to_word)
            if mapped_word_idx == word_idx
        ]

        if not subword_indices:

            raise RuntimeError(
                f"No embedding found for word "
                f"{word_idx}: {word}"
            )

        # -------------------------------------------------
        # One embedding for the original word.
        #
        # If a word consists of multiple subwords,
        # average their contextual embeddings.
        # -------------------------------------------------

        embedding = subword_embeddings[
            subword_indices
        ].mean(dim=0)

        word_embeddings.append(
            WordEmbedding(
                index=word_idx,
                text=word,
                embedding=embedding
            )
        )

    return word_embeddings


# =========================================================
# PRINT WORD-LEVEL EMBEDDINGS
# =========================================================

def print_word_level_embeddings(
    word_embeddings,
    show_vector=False
):

    print()
    print("=" * 90)
    print("1. WORD-LEVEL CONTEXTUAL EMBEDDINGS")
    print("=" * 90)

    for word in word_embeddings:

        embedding = word.embedding

        print(
            f"[{word.index}] "
            f"{word.text:<20} "
            f"shape={tuple(embedding.shape)}"
        )

        if show_vector:

            print(
                embedding
                .detach()
                .cpu()
                .tolist()
            )

    print("=" * 90)


# =========================================================
# SPACY DEPENDENCY CANDIDATES
# =========================================================
#
# spaCy creates hypothetical sentence elements from:
#
#     HEAD
#       +
#     CHILDREN
#       +
#     GRANDCHILDREN
#
# The candidate contains references to the ORIGINAL
# word indices.
#
# =========================================================

def generate_dependency_candidates(spacy_tokens):
    """
    Generate hypothetical sentence elements from:

        HEAD
        ├── CHILD
        │   └── GRANDCHILD
        └── CHILD

    The returned indices refer to the canonical spaCy-token list,
    which is also the index used by the original word embeddings.
    """

    candidates = []
    seen_candidates = set()

    # Map actual spaCy token -> canonical index
    token_to_index = {
        token: i
        for i, token in enumerate(spacy_tokens)
    }

    for head in spacy_tokens:

        if head.is_punct or head.is_space:
            continue

        indices = {
            token_to_index[head]
        }

        # -----------------------------------------------------
        # CHILDREN
        # -----------------------------------------------------

        for child in head.children:

            if child.is_punct or child.is_space:
                continue

            if child not in token_to_index:
                continue

            indices.add(
                token_to_index[child]
            )

            # -------------------------------------------------
            # GRANDCHILDREN
            # -------------------------------------------------

            for grandchild in child.children:

                if grandchild.is_punct or grandchild.is_space:
                    continue

                if grandchild not in token_to_index:
                    continue

                indices.add(
                    token_to_index[grandchild]
                )

        # Need at least two words
        if len(indices) < 2:
            continue

        # Original sentence order
        indices = sorted(indices)

        key = tuple(indices)

        if key in seen_candidates:
            continue

        seen_candidates.add(key)

        tokens = [
            spacy_tokens[i]
            for i in indices
        ]

        candidates.append({
            "text": " ".join(
                token.text
                for token in tokens
            ),

            "token_indices": indices,

            "tokens": [
                token.text
                for token in tokens
            ],

            "dependencies": [
                {
                    "index": i,
                    "text": token.text,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "head_index": (
                        token_to_index.get(
                            token.head,
                            None
                        )
                    ),
                }
                for i, token in zip(indices, tokens)
            ],
        })

    return candidates

# =========================================================
# ATTACH ORIGINAL WORD EMBEDDINGS
# =========================================================
#
# IMPORTANT:
#
# This does NOT create a new embedding.
#
# It simply associates every hypothetical candidate
# with the embeddings that were calculated in Stage 1.
#
# =========================================================

def attach_original_embeddings(
    candidates,
    word_embeddings,
):
    """
    Attach references to the ORIGINAL word-level embeddings.

    No embedding is recalculated here.
    """

    result = []

    for candidate in candidates:

        original_embeddings = []

        for index in candidate["token_indices"]:

            if index < 0 or index >= len(word_embeddings):
                raise RuntimeError(
                    f"Embedding index out of range: {index}"
                )

            embedding = word_embeddings[index]

            original_embeddings.append({
                "index": embedding.index,
                "text": embedding.text,
                "shape": list(
                    embedding.embedding.shape
                ),
            })

        candidate_copy = {
            **candidate,
            "original_embeddings":
                original_embeddings,
        }

        result.append(candidate_copy)

    return result

# =========================================================
# PRINT SPACY HYPOTHETICAL SENTENCE ELEMENTS
# =========================================================

def print_dependency_candidates(
    candidates
):

    print()
    print("=" * 90)
    print("2. SPACY HYPOTHETICAL SENTENCE ELEMENTS")
    print("=" * 90)

    if not candidates:

        print(
            "No dependency candidates generated."
        )

        print("=" * 90)

        return

    for number, candidate in enumerate(
        candidates,
        start=1
    ):

        print()
        print(
            f"CANDIDATE {number}"
        )

        print(
            f"    phrase: "
            f"{candidate['text']}"
        )

        print(
            f"    word indices: "
            f"{candidate['token_indices']}"
        )

        print(
            f"    words: "
            f"{candidate['tokens']}"
        )

        print(
            "    dependency structure:"
        )

        for dependency in candidate[
            "dependencies"
        ]:

            print(
                f"        "
                f"[{dependency['index']}] "
                f"{dependency['text']:<20} "
                f"POS={dependency['pos']:<8} "
                f"DEP={dependency['dep']:<12} "
                f"HEAD={dependency['head']}"
            )

        print(
            "    original word embeddings:"
        )

        for embedding in candidate[
            "original_embeddings"
        ]:

            print(
                f"        "
                f"[{embedding['index']}] "
                f"{embedding['text']:<20} "
                f"shape={tuple(embedding['shape'])}"
            )

    print()
    print("=" * 90)


# =========================================================
# COMPLETE WORD + DEPENDENCY INSPECTION
# =========================================================

def inspect_text(text):

    print("\n" + "#" * 90)
    print("INSPECTING TEXT")
    print("#" * 90)
    print(text)
    print("#" * 90)

    # =========================================================
    # 1. SPAcy parses the original text
    # =========================================================

    doc = nlp_de(text)

    print_spacy_tree(doc)

    # =========================================================
    # 2. spaCy is the canonical word tokenizer
    # =========================================================

    spacy_tokens = [
        token
        for token in doc
        if not token.is_space
    ]

    words = [
        token.text
        for token in spacy_tokens
    ]

    print("\nCANONICAL WORDS")
    print("-" * 90)

    for i, token in enumerate(spacy_tokens):
        print(
            f"[{i}] {token.text}"
        )

    # =========================================================
    # 3. Calculate ORIGINAL word-level contextual embeddings
    #
    # These are calculated exactly once.
    # =========================================================

    word_embeddings = get_word_level_embeddings(
        words,
        tokenizer,
        model,
        device,
    )

    print_word_level_embeddings(
        word_embeddings
    )

    # =========================================================
    # 4. Generate hypothetical sentence elements
    #
    # spaCy ONLY selects original word indices here.
    # No embeddings are recalculated.
    # =========================================================

    candidates = generate_dependency_candidates(
        spacy_tokens
    )

    # =========================================================
    # 5. Attach ORIGINAL embeddings
    # =========================================================

    candidates = attach_original_embeddings(
        candidates,
        word_embeddings,
    )

    # =========================================================
    # 6. Print candidates
    # =========================================================

    print_dependency_candidates(
        candidates
    )

    # =========================================================
    # 7. JSON-safe response
    # =========================================================

    return {
        "text": text,

        "words": [
            {
                "index": i,
                "text": token.text,
                "embedding_shape": list(
                    word_embeddings[i].embedding.shape
                ),
            }
            for i, token in enumerate(spacy_tokens)
        ],

        "dependency_candidates": candidates,
    }

# =========================================================
# 4. AWESOME-ALIGN WORD ALIGNMENT
# =========================================================
#
# This implements the core alignment calculation used by
# Awesome-Align's model example:
#
#   source embeddings
#          x
#   target embeddings
#          |
#          v
#      dot product
#          |
#     +----+----+
#     |         |
#   softmax   softmax
#   src->tgt  tgt->src
#     |         |
#     +----+----+
#          |
#     intersection
#
# Then subword links are converted into word links.
#
# =========================================================

def calculate_raw_scores(src_data, tgt_data):
    """
    Calculate the raw source-to-target similarity matrix.

    Rows    = source subwords
    Columns = target subwords

    No thresholding, softmax, or intersection is applied here.
    """

    src_embeddings = src_data["embeddings"]
    tgt_embeddings = tgt_data["embeddings"]

    if src_embeddings is None or tgt_embeddings is None:
        return None

    return torch.matmul(
        src_embeddings,
        tgt_embeddings.transpose(-1, -2)
    )

def extract_word_candidates(
    src_words,
    tgt_words,
    src_data,
    tgt_data,
    raw_scores,
    top_k=3
):
    """
    Convert the subword-level raw scores into word-level candidates.

    For every German source word, return the strongest English
    candidate words.

    The score for a word pair is the average of all subword-pair
    scores belonging to that word pair.
    """

    src_sub2word = src_data["sub2word"]
    tgt_sub2word = tgt_data["sub2word"]

    word_scores = {}

    for src_sub_idx, src_word_idx in enumerate(src_sub2word):

        for tgt_sub_idx, tgt_word_idx in enumerate(tgt_sub2word):

            score = raw_scores[
                src_sub_idx,
                tgt_sub_idx
            ].item()

            key = (src_word_idx, tgt_word_idx)

            word_scores.setdefault(key, []).append(score)

    candidates = []

    for src_idx, src_word in enumerate(src_words):

        source_candidates = []

        for tgt_idx, tgt_word in enumerate(tgt_words):

            key = (src_idx, tgt_idx)

            if key not in word_scores:
                continue

            scores = word_scores[key]

            average_score = sum(scores) / len(scores)

            source_candidates.append({
                "src_index": src_idx,
                "tgt_index": tgt_idx,
                "src_word": src_word,
                "tgt_word": tgt_word,
                "score": average_score
            })

        source_candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        candidates.extend(
            source_candidates[:top_k]
        )

    return candidates

def print_raw_candidate_scores(src_words, tgt_words, src_data, tgt_data):
    """
    Print the raw source -> target similarity scores.

    This does NOT apply the alignment threshold or mutual-intersection rule.
    It shows what the model itself considers similar.
    """

    src_embeddings = src_data["embeddings"]
    tgt_embeddings = tgt_data["embeddings"]

    # Raw similarity matrix:
    # rows    = source subwords
    # columns = target subwords
    similarity = torch.matmul(
        src_embeddings,
        tgt_embeddings.transpose(-1, -2)
    )

    src_sub2word = src_data["sub2word"]
    tgt_sub2word = tgt_data["sub2word"]

    # Collect all subword scores belonging to the same word pair.
    word_scores = {}

    for src_sub_idx, src_word_idx in enumerate(src_sub2word):
        for tgt_sub_idx, tgt_word_idx in enumerate(tgt_sub2word):

            key = (src_word_idx, tgt_word_idx)

            score = similarity[
                src_sub_idx,
                tgt_sub_idx
            ].item()

            word_scores.setdefault(key, []).append(score)

    print()
    print("=" * 75)
    print("RAW CANDIDATE SCORES")
    print("=" * 75)

    for src_idx, src_word in enumerate(src_words):

        candidates = []

        for tgt_idx, tgt_word in enumerate(tgt_words):

            key = (src_idx, tgt_idx)

            if key not in word_scores:
                continue

            # Average subword scores for this word pair.
            score = sum(word_scores[key]) / len(word_scores[key])

            candidates.append(
                (tgt_word, score)
            )

        # Strongest candidates first
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print()
        print(f"{src_word}")

        for tgt_word, score in candidates:
            print(
                f"    -> {tgt_word:<20} {score:>10.4f}"
            )

    print()
    print("=" * 75)

def calculate_strict_alignment(src_data, tgt_data):
    """
    Reproduce the original Awesome-Align-style
    bidirectional softmax + intersection alignment.

    This is deliberately kept separate from the raw scores.
    """

    src_embeddings = src_data["embeddings"]
    tgt_embeddings = tgt_data["embeddings"]

    if src_embeddings is None or tgt_embeddings is None:
        return []

    similarity = torch.matmul(
        src_embeddings,
        tgt_embeddings.transpose(-1, -2)
    )

    # Source -> target
    softmax_src_to_tgt = torch.softmax(
        similarity,
        dim=-1
    )

    # Target -> source
    softmax_tgt_to_src = torch.softmax(
        similarity,
        dim=-2
    )

    # Mutual alignment
    intersection = (
        (softmax_src_to_tgt > ALIGN_THRESHOLD)
        *
        (softmax_tgt_to_src > ALIGN_THRESHOLD)
    )

    src_sub2word = src_data["sub2word"]
    tgt_sub2word = tgt_data["sub2word"]

    links = []

    for src_sub_idx, tgt_sub_idx in intersection.nonzero(
        as_tuple=False
    ):

        src_word_idx = src_sub2word[src_sub_idx.item()]
        tgt_word_idx = tgt_sub2word[tgt_sub_idx.item()]

        score = similarity[
            src_sub_idx,
            tgt_sub_idx
        ].item()

        links.append({
            "src_index": src_word_idx,
            "tgt_index": tgt_word_idx,
            "score": score
        })

    return links

def print_word_candidates(
    src_words,
    tgt_words,
    candidates,
    top_k=3
):
    print()
    print("=" * 75)
    print("RAW WORD CANDIDATES")
    print("=" * 75)

    for src_idx, src_word in enumerate(src_words):

        source_candidates = [
            candidate
            for candidate in candidates
            if candidate["src_index"] == src_idx
        ]

        source_candidates = source_candidates[:top_k]

        print()
        print(f"{src_word}")

        for candidate in source_candidates:
            print(
                f"    -> "
                f"{candidate['tgt_word']:<20} "
                f"{candidate['score']:>10.4f}"
            )

    print()
    print("=" * 75)

def print_strict_alignment(
    src_words,
    tgt_words,
    links
):
    print()
    print("=" * 75)
    print("STRICT ALIGNMENT")
    print("=" * 75)

    print(
        f"{'SRC':<25}"
        f"{'TGT':<30}"
        f"{'SCORE':>10}"
    )

    print("-" * 75)

    for link in links:

        src_word = src_words[
            link["src_index"]
        ]

        tgt_word = tgt_words[
            link["tgt_index"]
        ]

        print(
            f"{src_word:<25}"
            f"{tgt_word:<30}"
            f"{link['score']:>10.4f}"
        )

    print("=" * 75)

# =========================================================
# 5. BUILD WORD PAIRS
# =========================================================
#
# For now this function does NOT try to create phrases.
#
# One alignment link becomes one WordPair.
#
# Later this is where we will implement:
#
#     zur Ruh -> to rest
#
# =========================================================

def build_word_pairs(src_words, tgt_words, alignment):

    pairs = []

    for link in alignment:

        src_idx = link["src_index"]
        tgt_idx = link["tgt_index"]

        pairs.append({
            "de": src_words[src_idx],
            "en": tgt_words[tgt_idx],
            "confidence": link["score"],
            "source_indices": [src_idx],
            "target_indices": [tgt_idx]
        })

    return pairs

# =========================================================
# 6. ALIGN WORDS
# =========================================================
#
# This function is the alignment stage.
#
# Translation happens outside it.
#
# =========================================================

def align_words(src, tgt):

    src_words = tokenize_words(src)
    tgt_words = tokenize_words(tgt)

    if not src_words or not tgt_words:

        return []

    # -----------------------------------------------------
    # Get contextual embeddings for both sentences
    # -----------------------------------------------------

    src_data = get_alignment_embeddings(
        src_words,
        tokenizer,
        model,
        device
    )

    tgt_data = get_alignment_embeddings(
        tgt_words,
        tokenizer,
        model,
        device
    )

    # -----------------------------------------------------
    # Calculate Awesome-Align links
    # -----------------------------------------------------

    alignment = calculate_word_alignment(
        src_data,
        tgt_data
    )

    return build_word_pairs(
        src_words,
        tgt_words,
        alignment
    )


# =========================================================
# 7. PRINT RAW ALIGNMENT
# =========================================================

def print_alignment(
    src,
    tgt,
    alignment
):

    src_words = tokenize_words(src)
    tgt_words = tokenize_words(tgt)

    print()
    print("=" * 75)
    print("GERMAN:")
    print(src)
    print()
    print("ENGLISH:")
    print(tgt)
    print("=" * 75)

    print(
        f"{'SRC':<25}"
        f"{'TGT':<25}"
        f"{'SCORE':>10}"
    )

    print("-" * 75)

    for link in alignment:

        src_idx = link["source_indices"][0]
        tgt_idx = link["target_indices"][0]

        src_word = src_words[src_idx]
        tgt_word = tgt_words[tgt_idx]

        score = link["confidence"]

        print(
            f"{src_word:<25}"
            f"{tgt_word:<25}"
            f"{score:>10.4f}"
        )

    print("=" * 75)
    print()


# =========================================================
# 8. COMPLETE PIPELINE
# =========================================================

def align_sentence(src: str):

    # ---------------------------------------------------------
    # 1. Translate
    # ---------------------------------------------------------

    tgt = translate_sentence(src)

    src_words = tokenize_words(src)
    tgt_words = tokenize_words(tgt)

    # ---------------------------------------------------------
    # 2. Get contextual embeddings
    # ---------------------------------------------------------

    src_data = get_alignment_embeddings(
        src_words,
        tokenizer,
        model,
        device
    )

    tgt_data = get_alignment_embeddings(
        tgt_words,
        tokenizer,
        model,
        device
    )

    # ---------------------------------------------------------
    # 3. Calculate RAW similarity matrix
    # ---------------------------------------------------------

    raw_scores = calculate_raw_scores(
        src_data,
        tgt_data
    )

    # ---------------------------------------------------------
    # 4. Convert raw subword scores into word candidates
    # ---------------------------------------------------------

    candidates = extract_word_candidates(
        src_words,
        tgt_words,
        src_data,
        tgt_data,
        raw_scores,
        top_k=3
    )

    # ---------------------------------------------------------
    # 5. Calculate strict Awesome-Align links
    # ---------------------------------------------------------

    strict_links = calculate_strict_alignment(
        src_data,
        tgt_data
    )

    # ---------------------------------------------------------
    # 6. Diagnostic output
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("GERMAN:")
    print(src)

    print()
    print("ENGLISH:")
    print(tgt)

    print("=" * 75)

    print_word_candidates(
        src_words,
        tgt_words,
        candidates,
        top_k=3
    )

    print_strict_alignment(
        src_words,
        tgt_words,
        strict_links
    )

    # ---------------------------------------------------------
    # For now, return the strict links.
    # Phrase grouping comes later.
    # ---------------------------------------------------------

    pairs = build_word_pairs(
        src_words,
        tgt_words,
        strict_links
    )

    return tgt, pairs
# =========================================================
# API
# =========================================================

@app.post(
    "/align",
    response_model=AlignResponse
)
def align(req: AlignRequest):

    translation, pairs = align_sentence(
        req.text
    )

    return {
        "translation": translation,
        "pairs": pairs
    }

# =========================================================
# DEBUG / INSPECTION API
# =========================================================

@app.post(
    "/inspect",
    response_model=InspectResponse
)
def inspect(req: InspectRequest):

    print()
    print()
    print("#" * 90)
    print("INSPECTING TEXT")
    print("#" * 90)
    print(req.text)
    print("#" * 90)

    result = inspect_text(
        req.text
    )

    return result

# =========================================================
# SAVE GAME PAIRS
# =========================================================

@app.post("/save_pairs")
async def save_pairs(data: dict):

    global selected_pairs

    selected_pairs = data["pairs"]

    return {
        "status": "ok"
    }


# =========================================================
# GAME WORDS
# =========================================================

@app.get("/game_words")
async def game_words():

    return build_game(
        selected_pairs
    )

@app.get("/inspect")
def inspect_page():

    return FileResponse(
        "static/inspect.html"
    )