"""Canonical label taxonomies.

This module is the SINGLE SOURCE OF TRUTH for every label mapping used in the
paper. The original exploratory notebooks contained three divergent copies of
``CATEGORY_MAP`` (they disagreed on ``Insect Bite`` and on whether the
``traumatic_other`` / ``pigmentary`` groups existed). The version below is the
one that produced the reported results in Table 2, Table 3 and Figure 2:
``Insect Bite`` is grouped as *inflammatory* (it is a reactive dermatosis, not
an infection), and seven categories are defined.

Do not redefine these dictionaries anywhere else.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source domain: HAM10000 + ISIC 2019 -> shared 8-class neoplastic taxonomy
# ---------------------------------------------------------------------------

HAM_TO_8CLASS = {
    "mel": "MEL",
    "nv": "NV",
    "bcc": "BCC",
    "akiec": "AK",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
}

ISIC_TO_8CLASS = {
    "MEL": "MEL",
    "NV": "NV",
    "BCC": "BCC",
    "AK": "AK",
    "BKL": "BKL",
    "DF": "DF",
    "VASC": "VASC",
    "SCC": "SCC",
    "UNK": None,  # dropped; note that UNK is all-zero in the 2019 training GT
}

NEOPLASTIC_CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

# ---------------------------------------------------------------------------
# DDI Fitzpatrick group codes
# ---------------------------------------------------------------------------

TONE_MAP = {12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"}

# ---------------------------------------------------------------------------
# SCIN: fine-grained condition -> clinical category
# ---------------------------------------------------------------------------

SCIN_CATEGORY_MAP = {
    # --- INFLAMMATORY / ECZEMATOUS / REACTIVE ---
    "Eczema": "inflammatory",
    "Allergic Contact Dermatitis": "inflammatory",
    "Urticaria": "inflammatory",
    "Psoriasis": "inflammatory",
    "Drug Rash": "inflammatory",
    "Acne": "inflammatory",
    "CD - Contact dermatitis": "inflammatory",
    "Acute dermatitis, NOS": "inflammatory",
    "Pityriasis rosea": "inflammatory",
    "Keratosis pilaris": "inflammatory",
    "Irritant Contact Dermatitis": "inflammatory",
    "Lichen Simplex Chronicus": "inflammatory",
    "Lichen planus/lichenoid eruption": "inflammatory",
    "Stasis Dermatitis": "inflammatory",
    "Rosacea": "inflammatory",
    "Prurigo nodularis": "inflammatory",
    "Hypersensitivity": "inflammatory",
    "Photodermatitis": "inflammatory",
    "Acute and chronic dermatitis": "inflammatory",
    "Intertrigo": "inflammatory",
    "Perioral Dermatitis": "inflammatory",
    "Seborrheic Dermatitis": "inflammatory",
    "Chronic dermatitis, NOS": "inflammatory",
    "Miliaria": "inflammatory",
    "Granuloma annulare": "inflammatory",
    "Insect Bite": "inflammatory",  # reactive, not an infection
    "Erythema multiforme": "inflammatory",
    "Cutaneous lupus": "inflammatory",
    "Pityriasis lichenoides": "inflammatory",
    "Lichen nitidus": "inflammatory",
    "Cutaneous sarcoidosis": "inflammatory",
    "Hidradenitis": "inflammatory",
    "Lichenified eczematous dermatitis": "inflammatory",
    "Lichen spinulosus": "inflammatory",
    # --- INFECTIOUS ---
    "Folliculitis": "infectious",
    "Tinea": "infectious",
    "Impetigo": "infectious",
    "Herpes Zoster": "infectious",
    "Herpes Simplex": "infectious",
    "Tinea Versicolor": "infectious",
    "Viral Exanthem": "infectious",
    "Verruca vulgaris": "infectious",
    "Scabies": "infectious",
    "Cellulitis": "infectious",
    "Molluscum Contagiosum": "infectious",
    "Ecthyma": "infectious",
    "Onychomycosis": "infectious",
    "Infected eczema": "infectious",
    # --- VASCULAR / PURPURIC ---
    "Pigmented purpuric eruption": "vascular_purpuric",
    "Leukocytoclastic Vasculitis": "vascular_purpuric",
    "Purpura": "vascular_purpuric",
    "O/E - ecchymoses present": "vascular_purpuric",
    "Livedo reticularis": "vascular_purpuric",
    "Traumatic petechiae": "vascular_purpuric",
    # --- NEOPLASTIC / PIGMENTED LESIONS ---
    "Actinic Keratosis": "neoplastic",
    "SCC/SCCIS": "neoplastic",
    "Hemangioma": "neoplastic",
    "Dermatofibroma": "neoplastic",
    "SK/ISK": "neoplastic",
    "Cutaneous T Cell Lymphoma": "neoplastic",
    "Melanocytic Nevus": "neoplastic",
    "Basal Cell Carcinoma": "neoplastic",
    # --- TRAUMATIC / OTHER ---
    "Abrasion, scrape, or scab": "traumatic_other",
    "Scar Condition": "traumatic_other",
    "Abscess": "traumatic_other",
    "Inflicted skin lesions": "traumatic_other",
    "Xerosis": "traumatic_other",
    "Superficial wound of body region": "traumatic_other",
    # --- PIGMENTARY ---
    "Post-Inflammatory hyperpigmentation": "pigmentary",
    "Erythema ab igne": "pigmentary",
}

# Keyword fallback for the long tail (SCIN has 211 distinct labels).
_FALLBACK_RULES = [
    ("inflammatory", ["dermatitis", "eczema", "psoriasis", "lichen", "urticaria",
                      "prurig", "rosacea", "acne", "bite"]),
    ("infectious", ["tinea", "herpes", "fungal", "candida", "infection", "viral",
                    "warts", "verruca", "molluscum", "scabies", "cellulitis",
                    "impetigo", "syphilis", "larva", "mycobacter"]),
    ("neoplastic", ["carcinoma", "melanoma", "nevus", "keratosis", "lymphoma",
                    "sarcoma", "neoplasm", "metastasis", "xanthoma", "fibroma"]),
    ("vascular_purpuric", ["purpura", "vasculitis", "petechiae", "ecchymos",
                           "livedo", "hematoma", "varicose", "stasis ulcer"]),
    ("pigmentary", ["pigment", "melasma", "vitiligo", "hypomelanosis",
                    "hyperpigment", "hypopigment"]),
]

SCIN_CATEGORIES = [
    "inflammatory", "infectious", "vascular_purpuric",
    "neoplastic", "traumatic_other", "pigmentary", "other",
]


def map_scin_category(label: str) -> str:
    """Map a fine-grained SCIN condition label to a clinical category.

    Exact dictionary lookup first, then an ordered keyword fallback, then
    ``"other"``. Order matters: the inflammatory rule is checked before the
    infectious one, so e.g. "Contact dermatitis, NOS" stays inflammatory.
    """
    if label in SCIN_CATEGORY_MAP:
        return SCIN_CATEGORY_MAP[label]
    low = str(label).lower()
    for category, keywords in _FALLBACK_RULES:
        if any(k in low for k in keywords):
            return category
    return "other"
