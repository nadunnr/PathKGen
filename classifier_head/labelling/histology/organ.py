# Canonical primary site dictionary
SITE_PATTERNS = {
    "breast": [
        r"breast"
    ],

    "lung": [
        r"lung",
        r"bronchus"
    ],

    "brain": [
        r"brain"
    ],

    "kidney": [
        r"kidney",
        r"renal"
    ],

    "soft tissue": [
        r"connective",
        r"subcutaneous",
        r"soft tissue",
        r"retroperitoneum",
        r"peritoneum"
    ],

    "uterus": [
        r"corpus uteri",
        r"uterus",
        r"endometrial",
        r"cervix uteri"
    ],

    "thyroid": [
        r"thyroid"
    ],

    "bladder": [
        r"bladder"
    ],

    "prostate": [
        r"prostate"
    ],

    "liver": [
        r"liver",
        r"intrahepatic bile",
        r"biliary tract"
    ],

    "adrenal gland": [
        r"adrenal"
    ],

    "colon": [
        r"colon",
        r"rectum",
        r"rectosigmoid"
    ],

    "stomach": [
        r"stomach"
    ],

    "lymph node": [
        r"lymph node"
    ],

    "larynx": [
        r"larynx"
    ],

    "thymus": [
        r"thymus"
    ],

    "skin": [
        r"skin"
    ],

    "pleura": [
        r"pleura",
        r"mediastinum"
    ],

    "esophagus": [
        r"esophagus"
    ],

    "oral cavity": [
        r"tongue",
        r"mouth",
        r"floor of mouth",
        r"gum",
        r"salivary gland",
        r"lip"
    ],

    "eye": [
        r"eye",
        r"adnexa"
    ],

    "ovary": [
        r"ovary"
    ],

    "small intestine": [
        r"small intestine"
    ],

    "oropharynx": [
        r"oropharynx",
        r"tonsil",
        r"base of tongue",
        r"hypopharynx"
    ],

    "bone": [
        r"bone",
        r"joint",
        r"articular cartilage"
    ],

    "testis": [
        r"testis",
        r"testicular"
    ]
}

SITE_PRIORITY = [
    "breast",
    "lung",
    "brain",
    "kidney",
    "liver",
    "bladder",
    "prostate",
    "thyroid",
    "uterus",
    "colon",
    "stomach",
    "ovary",
    "testis",
    "adrenal gland",
    "lymph node",
    "soft tissue",
    "bone",
    "skin",
    "pleura",
    "esophagus",
    "larynx",
    "thymus",
    "oral cavity",
    "oropharynx",
    "eye",
    "small intestine"
]

SITE_TO_CUI = {
    "breast": "C0006142",
    "lung": "C0024109",
    "brain": "C0006104",
    "kidney": "C0022646",
    "soft tissue": "C0037363",
    "uterus": "C0042149",
    "thyroid": "C0040134",
    "bladder": "C0005684",
    "prostate": "C0033572",
    "liver": "C0023884",
    "adrenal gland": "C0001623",
    "colon": "C0009378",
    "stomach": "C0038350",
    "lymph node": "C0024204",
    "larynx": "C0023144",
    "thymus": "C0040102",
    "skin": "C0037284",
    "pleura": "C0032227",
    "esophagus": "C0014854",
    "oral cavity": "C0226896",
    "eye": "C0015392",
    "ovary": "C0029925",
    "small intestine": "C0037199",
    "oropharynx": "C0028978",
    "bone": "C0005940",
    "testis": "C0029925"
}


