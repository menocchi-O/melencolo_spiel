import inspect
import spacy


# ---------------------------------------------------------
# Load language models
# ---------------------------------------------------------

nlp_de = spacy.load("de_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------
def print_dependency_relations(doc, language):

    print()
    print("=" * 90)
    print(f"{language} DEPENDENCY RELATIONS")
    print("=" * 90)

    print(
        f"{'TOKEN':<15}"
        f"{'HEAD':<15}"
        f"{'DEP':<15}"
        f"{'TOKEN INDEX':<15}"
        f"{'HEAD INDEX':<15}"
    )

    print("-" * 90)

    for token in doc:

        if token.head != token:

            print(
                f"{token.text:<15}"
                f"{token.head.text:<15}"
                f"{token.dep_:<15}"
                f"{token.i:<15}"
                f"{token.head.i:<15}"
            )
def inspect_dependency_spans(doc):
    print()
    print("=" * 90)
    print("POSSIBLE DEPENDENCY SPANS")
    print("=" * 90)

    for token in doc:
        children = list(token.children)

        if not children:
            continue

        # Token + direct children
        indices = [token.i] + [child.i for child in children]
        indices.sort()

        words = [doc[i].text for i in indices]
        span = " ".join(words)

        print()
        print(f"HEAD: {token.text}")
        print(f"DEP : {token.dep_}")
        print(f"SPAN: {span}")
        print(
            "TOKENS:",
            ", ".join(
                f"{child.text} ({child.dep_})"
                for child in children
            )
        )
def inspect_dependency_chains(doc):
    print()
    print("=" * 90)
    print("POSSIBLE DEPENDENCY CHAINS")
    print("=" * 90)

    for token in doc:
        for child in token.children:

            # Is the child itself a head of something?
            grandchildren = list(child.children)

            if grandchildren:
                indices = [token.i, child.i]

                for grandchild in grandchildren:
                    indices.append(grandchild.i)

                indices = sorted(set(indices))

                words = [doc[i].text for i in indices]

                print()
                print(
                    f"{token.text} "
                    f"({token.dep_})"
                    f" -> "
                    f"{child.text} "
                    f"({child.dep_})"
                )

                for grandchild in grandchildren:
                    print(
                        f"    -> {grandchild.text} "
                        f"({grandchild.dep_})"
                    )

                print("SPAN:", " ".join(words))
def inspect_pos_and_dependencies(doc):
    print()
    print("=" * 110)
    print("POS + DEPENDENCY + MORPHOLOGY")
    print("=" * 110)

    print(
        f"{'TOKEN':<15}"
        f"{'POS':<10}"
        f"{'DEP':<10}"
        f"{'HEAD':<15}"
        f"MORPH"
    )

    print("-" * 110)

    for token in doc:
        print(
            f"{token.text:<15}"
            f"{token.pos_:<10}"
            f"{token.dep_:<10}"
            f"{token.head.text:<15}"
            f"{token.morph}"
        )
def generate_dependency_candidates(doc):
    candidates = []

    for head in doc:

        # We don't want ROOTs
        if head.dep_ == "ROOT":
            continue

        # Ignore punctuation
        if head.pos_ == "PUNCT":
            continue

        children = list(head.children)

        if not children:
            continue

        # Start with the HEAD itself
        tokens = [head]

        # Add children
        tokens.extend(children)

        # Add grandchildren
        for child in children:
            tokens.extend(list(child.children))

        # Remove punctuation
        tokens = [
            token
            for token in tokens
            if token.pos_ != "PUNCT"
        ]

        # Remove duplicates
        unique_tokens = {
            token.i: token
            for token in tokens
        }

        tokens = list(unique_tokens.values())

        # ---------------------------------------------------------
        # At most one token of each POS
        # ---------------------------------------------------------
        pos_seen = set()
        filtered_tokens = []

        for token in tokens:
            if token.pos_ in pos_seen:
                continue

            pos_seen.add(token.pos_)
            filtered_tokens.append(token)

        tokens = filtered_tokens

        # Need at least two words for a compound
        if len(tokens) < 2:
            continue

        # Restore original sentence order
        tokens.sort(key=lambda token: token.i)

        # Create the text
        text = " ".join(token.text for token in tokens)

        candidates.append({
            "text": text,
            "token_indices": [token.i for token in tokens],
            "tokens": tokens
        })

    # Remove duplicate candidates
    unique_candidates = {}

    for candidate in candidates:
        unique_candidates[candidate["text"]] = candidate

    return list(unique_candidates.values())
# ---------------------------------------------------------
# Test sentences
# ---------------------------------------------------------

german = """Geh ich vor der Nacht zur Ruh.
Deck ich mich mit Schwermut zu.
Die helle Welt will mir nicht glücken.
Muss mich mit Finsternis verzücken"""
english = """I'll go to sleep before night.
I'll cover myself with melancholy.
The bright world will not bring me joy.
I must exalt with darkness.
"""


# ---------------------------------------------------------
# Parse sentences
# ---------------------------------------------------------

doc_de = nlp_de(german)
doc_en = nlp_en(english)


# ---------------------------------------------------------
# Print dependency information
# ---------------------------------------------------------

print()
print("=" * 90)
print("GERMAN DEPENDENCY TREE")
print("=" * 90)

print(
    f"{'TOKEN':<15}"
    f"{'LEMMA':<15}"
    f"{'POS':<10}"
    f"{'DEP':<15}"
    f"{'HEAD':<15}"
)

print("-" * 90)

for token in doc_de:
    print(
        f"{token.text:<15}"
        f"{token.lemma_:<15}"
        f"{token.pos_:<10}"
        f"{token.dep_:<15}"
        f"{token.head.text:<15}"
    )


print()
print("=" * 90)
print("ENGLISH DEPENDENCY TREE")
print("=" * 90)

print(
    f"{'TOKEN':<15}"
    f"{'LEMMA':<15}"
    f"{'POS':<10}"
    f"{'DEP':<15}"
    f"{'HEAD':<15}"
)

print("-" * 90)

for token in doc_en:
    print(
        f"{token.text:<15}"
        f"{token.lemma_:<15}"
        f"{token.pos_:<10}"
        f"{token.dep_:<15}"
        f"{token.head.text:<15}"
    )


print()
print("=" * 90)
print()
# print_dependency_relations(
#     doc_de,
#     "GERMAN"
# )

# print_dependency_relations(
#     doc_en,
#     "ENGLISH"
# )

# inspect_dependency_spans(
#     doc_de
#     )

# inspect_dependency_chains(
#     doc_de
#     )

# inspect_pos_and_dependencies(
#     doc_de
#     )

candidates = generate_dependency_candidates(doc_de)

print()
print("=" * 90)
print("DEPENDENCY CANDIDATES")
print("=" * 90)

for candidate in candidates:
    print(
        f"{candidate['text']:<30}"
        f"{candidate['token_indices']}"
    )