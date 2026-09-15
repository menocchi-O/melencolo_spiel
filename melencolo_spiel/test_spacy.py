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
print_dependency_relations(
    doc_de,
    "GERMAN"
)

print_dependency_relations(
    doc_en,
    "ENGLISH"
)

inspect_dependency_spans(
    doc_de
    )
