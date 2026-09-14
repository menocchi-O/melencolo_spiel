import spacy


# ---------------------------------------------------------
# Load language models
# ---------------------------------------------------------

nlp_de = spacy.load("de_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")


# ---------------------------------------------------------
# Test sentences
# ---------------------------------------------------------

german = """Geh ich vor der Nacht zur Ruh.
Deck ich mich mit Schwermut zu.
Die helle Welt will mir nicht glücken,
Muss mich mit Finsternis verzücken.
Es ist die totenschwangere Nacht
Die uns verzückt, zu Sündern macht.
Gebote, die wir übergehen.
Kann im Dunkeln niemand sehen.
Die Nacht ist wunderschön.
Ich will nicht schlafen gehen."""
english = """Before the night, I go to rest.
I cover myself with melancholy.
The bright world does not bring me joy,
I must delight myself in darkness.

It is the night, heavy with death,
That enchants us, makes sinners of us.
Commandments that we transgress—
In the darkness, no one can see.

The night is beautiful.
I do not want to go to sleep.
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