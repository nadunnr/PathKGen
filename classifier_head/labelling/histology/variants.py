VARIANT_PATTERNS = {

    "renal cell carcinoma": {
        "clear cell": [
            r"clear cell carcinoma",
            r"clear cell renal"
        ],
        "papillary": [
            r"papillary renal",
            r"papillary carcinoma"
        ],
        "chromophobe": [
            r"chromophobe"
        ]
    },

    "squamous cell carcinoma": {
        "non-keratinizing": [
            r"non[- ]keratinizing",
            r"lacks keratinization"
        ],
        "keratinizing": [
            r"keratinizing",
            r"keratin pearl"
        ],
        "spindle cell": [
            r"spindle cell carcinoma",
            r"sarcomatoid"
        ],
        "basaloid": [
            r"basaloid"
        ]
    },

    "adenocarcinoma": {
        "mucinous": [
            r"mucinous"
        ],
        "signet ring": [
            r"signet ring"
        ],
        "serous": [
            r"serous carcinoma"
        ]
    },

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
    "clear cell": "C0206698",
    "papillary": "C0334251",
    "chromophobe": "C0206699",

    "non-keratinizing": "C1335545",
    "keratinizing": "C0205678",
    "spindle cell": "C0278488",
    "basaloid": "C0334285",

    "mucinous": "C0334042",
    "signet ring": "C0037057",
    "serous": "C0205612",

    "seminoma": "C0036494",
    "nonseminoma": "C0205853"
}
