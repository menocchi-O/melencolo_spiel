# =========================================================
# WORD-LEVEL + SPACY HYPOTHETICAL CANDIDATES
# =========================================================

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import spacy


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class WordEmbedding:
    """
    One original German word and its contextual embedding.

    IMPORTANT:
    This embedding is calculated ONCE.

    spaCy candidates reference this object by word index.
    They do not recalculate or modify it.
    """

    index: int
    text: str
    embedding: torch.Tensor


@dataclass
class DependencyCandidate:
    """
    A hypothetical phrase generated from the spaCy
    dependency tree.

    token_indices refer back to the original German
    word-level embeddings.
    """

    text: str
    token_indices: List[int]
    tokens: List[str]


# =========================================================
# 1. WORD TOKENIZATION
# =========================================================

def tokenize_words(text: str) -> List[str]:

    return text.strip().split()


# =========================================================
# 2. WORD-LEVEL EMBEDDINGS
# =========================================================
#
# This is the ONLY place where we calculate the original
# word-level contextual embeddings.
#
# Example:
#
#   Ich will nicht schlafen gehen
#
# becomes:
#
#   0 Ich       -> embedding
#   1 will      -> embedding
#   2 nicht     -> embedding
#   3 schlafen  -> embedding
#   4 gehen     -> embedding
#
# These embeddings become the immutable basis for all
# subsequent hypothetical candidates.
#
# =========================================================

def get_word_level_embeddings(
    words,
    tokenizer,
    model,
    device,
    align_layer=8
):
    """
    Calculate contextual embeddings for every original word.

    Returns:
        List[WordEmbedding]

    IMPORTANT:
        One word can consist of multiple subword tokens.

        The embedding for the word is currently obtained
        by averaging its subword embeddings.
    """

    # -----------------------------------------------------
    # Tokenize each original word independently
    # -----------------------------------------------------

    tokenized_words = [
        tokenizer.tokenize(word)
        for word in words
    ]

    # -----------------------------------------------------
    # Convert subwords -> token IDs
    # -----------------------------------------------------

    word_token_ids = [
        tokenizer.convert_tokens_to_ids(tokens)
        for tokens in tokenized_words
    ]

    # -----------------------------------------------------
    # Flatten
    # -----------------------------------------------------

    flat_token_ids = []

    subword_to_word = []

    for word_idx, token_ids in enumerate(word_token_ids):

        flat_token_ids.extend(token_ids)

        subword_to_word.extend(
            [word_idx] * len(token_ids)
        )

    if not flat_token_ids:
        return []

    # -----------------------------------------------------
    # Prepare model input
    # -----------------------------------------------------

    input_ids = tokenizer.prepare_for_model(
        flat_token_ids,
        return_tensors="pt",
        truncation=True
    )["input_ids"]

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(device)

    # -----------------------------------------------------
    # Run model
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            input_ids,
            output_hidden_states=True
        )

    # -----------------------------------------------------
    # Extract requested layer
    #
    # Remove:
    #
    #   [CLS]
    #   [SEP]
    # -----------------------------------------------------

    subword_embeddings = output.hidden_states[
        align_layer
    ][0, 1:-1]

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if len(subword_to_word) != subword_embeddings.shape[0]:

        raise RuntimeError(
            "Subword mapping and embedding count differ: "
            f"{len(subword_to_word)} mappings vs "
            f"{subword_embeddings.shape[0]} embeddings."
        )

    # -----------------------------------------------------
    # Convert subword embeddings -> word embeddings
    #
    # Every original word gets ONE embedding.
    # -----------------------------------------------------

    word_embeddings = []

    for word_idx, word in enumerate(words):

        indices = [
            i
            for i, mapped_word_idx
            in enumerate(subword_to_word)
            if mapped_word_idx == word_idx
        ]

        if not indices:
            raise RuntimeError(
                f"No subword embedding found for word "
                f"{word_idx}: {word}"
            )

        embedding = subword_embeddings[
            indices
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
# 3. PRINT WORD-LEVEL EMBEDDINGS
# =========================================================
#
# This function ONLY prints the original word-level
# embeddings.
#
# It knows nothing about spaCy.
# It knows nothing about dependency candidates.
#
# =========================================================

def print_word_level_embeddings(
    word_embeddings,
    show_vector=False
):

    print()
    print("=" * 90)
    print("WORD-LEVEL CONTEXTUAL EMBEDDINGS")
    print("=" * 90)

    for word in word_embeddings:

        vector = word.embedding

        print()
        print(
            f"[{word.index}] "
            f"{word.text:<20}"
            f"shape={tuple(vector.shape)}"
        )

        if show_vector:

            print(
                vector.detach()
                .cpu()
                .tolist()
            )

    print()
    print("=" * 90)


# =========================================================
# 4. BUILD WORD LOOKUP
# =========================================================

def build_word_lookup(word_embeddings):

    return {
        word.index: word
        for word in word_embeddings
    }


# =========================================================
# 5. SPACY DEPENDENCY CANDIDATES
# =========================================================
#
# spaCy operates independently of the embeddings.
#
# It generates hypothetical German phrases from:
#
#       HEAD
#        |
#      CHILD
#        |
#   GRANDCHILD
#
# The resulting token indices are then connected back to
# the ORIGINAL word-level embeddings.
#
# =========================================================

def generate_dependency_candidates(
    doc,
    word_embeddings
):

    word_lookup = build_word_lookup(
        word_embeddings
    )

    candidates = []

    for head in doc:

        # -------------------------------------------------
        # Ignore ROOT
        # -------------------------------------------------

        if head.dep_ == "ROOT":
            continue

        # -------------------------------------------------
        # Ignore punctuation
        # -------------------------------------------------

        if head.pos_ == "PUNCT":
            continue

        children = list(head.children)

        if not children:
            continue

        # -------------------------------------------------
        # Start with HEAD
        # -------------------------------------------------

        tokens = [head]

        # -------------------------------------------------
        # Add CHILDREN
        # -------------------------------------------------

        tokens.extend(children)

        # -------------------------------------------------
        # Add GRANDCHILDREN
        # -------------------------------------------------

        for child in children:

            tokens.extend(
                list(child.children)
            )

        # -------------------------------------------------
        # Remove punctuation
        # -------------------------------------------------

        tokens = [
            token
            for token in tokens
            if token.pos_ != "PUNCT"
        ]

        # -------------------------------------------------
        # Remove duplicate token indices
        # -------------------------------------------------

        unique_tokens = {
            token.i: token
            for token in tokens
        }

        tokens = list(
            unique_tokens.values()
        )

        # -------------------------------------------------
        # At most one token per POS
        # -------------------------------------------------

        pos_seen = set()
        filtered_tokens = []

        for token in tokens:

            if token.pos_ in pos_seen:
                continue

            pos_seen.add(token.pos_)
            filtered_tokens.append(token)

        tokens = filtered_tokens

        # -------------------------------------------------
        # Need at least two words
        # -------------------------------------------------

        if len(tokens) < 2:
            continue

        # -------------------------------------------------
        # Restore original sentence order
        # -------------------------------------------------

        tokens.sort(
            key=lambda token: token.i
        )

        # -------------------------------------------------
        # spaCy token indices correspond to the original
        # sentence word positions ONLY if the tokenization
        # matches our word tokenizer.
        #
        # We explicitly verify this.
        # -------------------------------------------------

        token_indices = [
            token.i
            for token in tokens
        ]

        # -------------------------------------------------
        # Verify that every token has an original embedding
        # -------------------------------------------------

        missing = [
            index
            for index in token_indices
            if index not in word_lookup
        ]

        if missing:

            raise RuntimeError(
                "spaCy token indices do not match "
                "word-level embedding indices. "
                f"Missing indices: {missing}"
            )

        # -------------------------------------------------
        # Candidate text
        # -------------------------------------------------

        text = " ".join(
            token.text
            for token in tokens
        )

        # -------------------------------------------------
        # Candidate
        # -------------------------------------------------

        candidates.append(
            DependencyCandidate(
                text=text,
                token_indices=token_indices,
                tokens=[
                    token.text
                    for token in tokens
                ]
            )
        )

    # -----------------------------------------------------
    # Remove duplicate candidate strings
    # -----------------------------------------------------

    unique_candidates = {}

    for candidate in candidates:

        unique_candidates[
            candidate.text
        ] = candidate

    return list(
        unique_candidates.values()
    )


# =========================================================
# 6. ATTACH ORIGINAL WORD EMBEDDINGS
# =========================================================
#
# This does NOT calculate a new embedding.
#
# It simply associates the candidate's words with the
# embeddings calculated in Stage 1.
#
# =========================================================

def attach_word_embeddings(
    candidates,
    word_embeddings
):

    word_lookup = build_word_lookup(
        word_embeddings
    )

    results = []

    for candidate in candidates:

        candidate_embeddings = [
            word_lookup[index].embedding
            for index in candidate.token_indices
        ]

        results.append({
            "text": candidate.text,
            "token_indices": candidate.token_indices,
            "tokens": candidate.tokens,

            # These are the ORIGINAL word embeddings.
            "word_embeddings": candidate_embeddings
        })

    return results


# =========================================================
# 7. PRINT SPACY HYPOTHETICAL SENTENCE ELEMENTS
# =========================================================
#
# IMPORTANT:
#
# This output is deliberately separate from the word-level
# embedding output.
#
# =========================================================

def print_dependency_candidates(
    candidates
):

    print()
    print("=" * 90)
    print("SPACY HYPOTHETICAL SENTENCE ELEMENTS")
    print("=" * 90)

    for i, candidate in enumerate(
        candidates
    ):

        print()

        print(
            f"CANDIDATE {i + 1}"
        )

        print(
            f"    text: "
            f"{candidate['text']}"
        )

        print(
            f"    token indices: "
            f"{candidate['token_indices']}"
        )

        print(
            f"    tokens: "
            f"{candidate['tokens']}"
        )

        print(
            "    original word embeddings:"
        )

        for token_index, embedding in zip(
            candidate["token_indices"],
            candidate["word_embeddings"]
        ):

            print(
                f"        [{token_index}] "
                f"shape={tuple(embedding.shape)}"
            )

    print()
    print("=" * 90)


# =========================================================
# 8. COMPLETE DEBUG PIPELINE
# =========================================================
#
# This function deliberately stops AFTER the dependency
# candidate stage.
#
# NO English alignment.
# NO phrase scoring.
# NO phrase re-embedding.
#
# That comes later.
#
# =========================================================

def inspect_word_and_dependency_structure(
    text,
    tokenizer,
    model,
    device,
    nlp_de,
    align_layer=8
):

    # =====================================================
    # STAGE 1
    # WORD-LEVEL CONTEXTUAL EMBEDDINGS
    # =====================================================

    words = tokenize_words(text)

    word_embeddings = get_word_level_embeddings(
        words,
        tokenizer,
        model,
        device,
        align_layer=align_layer
    )

    print_word_level_embeddings(
        word_embeddings,
        show_vector=False
    )

    # =====================================================
    # STAGE 2
    # SPACY DEPENDENCY TREE
    # =====================================================

    doc = nlp_de(text)

    # =====================================================
    # STAGE 3
    # GENERATE HYPOTHETICAL PHRASES
    # =====================================================

    candidates = generate_dependency_candidates(
        doc,
        word_embeddings
    )

    # =====================================================
    # STAGE 4
    # CONNECT CANDIDATES TO ORIGINAL EMBEDDINGS
    # =====================================================

    candidates = attach_word_embeddings(
        candidates,
        word_embeddings
    )

    # =====================================================
    # STAGE 5
    # PRINT CANDIDATES
    # =====================================================

    print_dependency_candidates(
        candidates
    )

    return {
        "words": words,
        "word_embeddings": word_embeddings,
        "candidates": candidates
    }