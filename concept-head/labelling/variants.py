VARIANT_PATTERNS = {

    # Renal Cell Carcinoma
    "renal cell carcinoma": {

        "clear cell": [
            r"clear cell carcinoma",
            r"clear cell renal",
            r"conventional \(clear cell\)",
            r"clear cytoplasm"
        ],

        "papillary": [
            r"papillary renal",
            r"papillary renal cell carcinoma",
            r"papillary type",
            r"papillary growth pattern"
        ],

        "chromophobe": [
            r"chromophobe"
        ],

        "sarcomatoid": [
            r"sarcomatoid"
        ]
    },

    # Squamous Cell Carcinoma
    "squamous cell carcinoma": {

        "keratinizing": [
            r"keratinizing",
            r"organized keratinization",
            r"keratin pearls?",
            r"well[- ]differentiated squamous"
        ],

        "non-keratinizing": [
            r"non[- ]keratinizing",
            r"lacks keratinization",
            r"minimal keratinization"
        ],

        "spindle cell": [
            r"spindle cell carcinoma",
            r"sarcomatoid"
        ],

        "basaloid": [
            r"basaloid"
        ]
    },

    # Adenocarcinoma (General)
    "adenocarcinoma": {

        "mucinous": [
            r"mucinous"
        ],

        "signet ring": [
            r"signet ring"
        ],

        "serous": [
            r"serous carcinoma",
            r"serous type",
            r"high[- ]grade serous",
            r"serous features"
        ],

        "endometrioid": [
            r"endometrioid",
            r"endometrioid type"
        ],

        "ductal": [
            r"ductal carcinoma",
            r"invasive ductal carcinoma",
            r"ductal carcinoma in situ",
            r"\bDCIS\b"
        ],

        "acinar": [
            r"acinar pattern",
            r"acinar adenocarcinoma",
            r"gland[- ]forming"
        ]
    },

    # Urothelial Carcinoma
    "urothelial carcinoma": {

        "sarcomatoid": [
            r"sarcomatoid differentiation",
            r"sarcomatoid features"
        ],

        "squamous differentiation": [
            r"squamous differentiation"
        ]
    },

    # Germ Cell Tumors
    "germ cell tumor": {

        "seminoma": [
            r"seminoma"
        ],

        "nonseminoma": [
            r"non[- ]seminoma",
            r"embryonal carcinoma",
            r"yolk sac tumor",
            r"teratoma"
        ]
    }
}

VARIANT_TO_CUI = {

    # =========================
    # Renal Cell Carcinoma
    # =========================
    "clear cell": "C0206698",          # Clear Cell Renal Cell Carcinoma
    "papillary": "C0334251",            # Papillary Renal Cell Carcinoma
    "chromophobe": "C0206699",          # Chromophobe Renal Cell Carcinoma
    "sarcomatoid": "C0278488",          # Sarcomatoid Carcinoma (used across organs)

    # =========================
    # Squamous Cell Carcinoma
    # =========================
    "keratinizing": "C0205678",          # Keratinizing Squamous Cell Carcinoma
    "non-keratinizing": "C1335545",      # Non-keratinizing Squamous Cell Carcinoma
    "spindle cell": "C0278488",          # Spindle Cell / Sarcomatoid Carcinoma
    "basaloid": "C0334285",              # Basaloid Squamous Cell Carcinoma

    # =========================
    # Adenocarcinoma Variants
    # =========================
    "mucinous": "C0334042",              # Mucinous Adenocarcinoma
    "signet ring": "C0037057",           # Signet Ring Cell Carcinoma
    "serous": "C0205612",                # Serous Carcinoma

    "endometrioid": "C0205614",          # Endometrioid Adenocarcinoma
    "ductal": "C0019693",                # Ductal Carcinoma (NOS, breast-centric)
    "acinar": "C0206695",                # Acinar Adenocarcinoma

    # =========================
    # Urothelial Carcinoma
    # =========================
    "squamous differentiation": "C0334286",  # Squamous Differentiation

    # =========================
    # Germ Cell Tumors
    # =========================
    "seminoma": "C0036494",               # Seminoma
    "nonseminoma": "C0205853"             # Nonseminomatous Germ Cell Tumor
}

