# Canonical histology dictionary
HISTOLOGY_PATTERNS = {
  "squamous cell carcinoma": [
      r"squamous cell carcinoma",
      r"squamous carcinoma",
      r"spindle cell carcinoma",
      r"sarcomatoid carcinoma"
  ],
  "adenocarcinoma": [
      r"adenocarcinoma",
      r"endometrial carcinoma",
      r"endometrioid carcinoma",
      r"uterine corpus endometrial carcinoma",
      r"uterine endometrial carcinoma"
  ],
  "breast carcinoma": [
      r"breast carcinoma",
      r"invasive breast carcinoma",
      r"breast invasive carcinoma"
  ],
  "renal cell carcinoma": [
      r"renal cell carcinoma",
      r"clear cell carcinoma",
      r"papillary renal"
  ],
  "urothelial carcinoma": [
      r"urothelial carcinoma",
      r"transitional cell carcinoma"
  ],
  "hepatocellular carcinoma": [
      r"hepatocellular carcinoma"
  ],
  "melanoma": [
      r"melanoma"
  ],
  "glioma": [
      r"glioma"
  ],
  "sarcoma": [
      r"sarcoma"
  ],
  "lymphoma": [
      r"lymphoma"
  ],
  "thyroid carcinoma": [
      r"thyroid carcinoma"
  ],
  "mesothelioma": [
      r"mesothelioma"
  ],
  "thymoma": [
      r"thymoma"
  ],
  "pheochromocytoma": [
      r"pheochromocytoma",
      r"paraganglioma"
  ],
  "germ cell tumor": [
      r"germ cell tumor",
      r"germ cell tumour",
      r"seminoma"
  ],
  "plasma cell neoplasm": [
      r"plasma cell neoplasm",
      r"plasma cell dyscrasia",
      r"multiple myeloma",
      r"plasmacytoma"
  ]
}

# Priority
HISTOLOGY_PRIORITY = [
  "breast carcinoma",
  "squamous cell carcinoma",
  "renal cell carcinoma",
  "urothelial carcinoma",
  "hepatocellular carcinoma",
  "germ cell tumor",
  "adenocarcinoma",
  "melanoma",
  "glioma",
  "sarcoma",
  "plasma cell neoplasm",
  "lymphoma",
  "thyroid carcinoma",
  "mesothelioma",
  "thymoma",
  "pheochromocytoma"
]

# UMLS CUI mapping
HISTOLOGY_TO_CUI = {
  "squamous cell carcinoma": "C0007137",
  "adenocarcinoma": "C0001418",
  "breast carcinoma": "C0678222",
  "renal cell carcinoma": "C0007134",
  "urothelial carcinoma": "C2145472",
  "hepatocellular carcinoma": "C2239176",
  "germ cell tumor": "C0205851",
  "melanoma": "C0025202",
  "glioma": "C0017638",
  "sarcoma": "C1261473",
  "lymphoma": "C0024299",
  "plasma cell neoplasm": "C1959632",
  "thyroid carcinoma": "C0549473",
  "mesothelioma": "C0025500",
  "thymoma": "C0040100",
  "pheochromocytoma": "C0031511"
}