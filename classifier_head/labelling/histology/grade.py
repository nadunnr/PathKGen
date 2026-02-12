GRADE_PATTERNS = {
    "grade 1": [
        r"\bgrade\s*i\b",
        r"\bg1\b",
        r"\bwell differentiated\b",
        r"\bwell-differentiated\b"
    ],

    "grade 2": [
        r"\bgrade\s*ii\b",
        r"\bg2\b",
        r"\bmoderately differentiated\b",
        r"\bmoderate differentiation\b",
        r"\bmoderate differentiated\b"
    ],

    "grade 3": [
        r"\bgrade\s*iii\b",
        r"\bg3\b",
        r"\bpoorly differentiated\b",
        r"\bpoor differentiation\b"
    ],

    "grade 4": [
        r"\bgrade\s*iv\b",
        r"\bg4\b",
        r"\bundifferentiated\b",
        r"\banaplastic\b"
    ]
}

GRADE_PRIORITY = [
    "grade 4",
    "grade 3",
    "grade 2",
    "grade 1"
]

GRADE_TO_CUI = {
    "grade 1": "C0205632",  # Well differentiated
    "grade 2": "C0205633",  # Moderately differentiated
    "grade 3": "C0205634",  # Poorly differentiated
    "grade 4": "C0334045"   # Undifferentiated / Anaplastic
}