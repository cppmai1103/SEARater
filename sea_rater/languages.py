"""
Central list of the pilot's 8 SEA languages and their FineWeb2
(HuggingFaceFW/fineweb-2) config names, shared by every script that streams
FineWeb2, builds language-keyed output paths, or loops per-language.

Every script used to hardcode its own copy of this list; once the pilot
expanded from 4 to 8 languages that duplication became risky (one script
forgetting an entry would silently skip a language instead of erroring), so
it's centralized here instead. Dependency-free (no torch) so it can be
imported from any script, GPU or not.
"""

LANGUAGE_HF_CONFIGS = {
    "vi": "vie_Latn",  # Vietnamese
    "id": "ind_Latn",  # Indonesian
    "th": "tha_Thai",  # Thai
    "km": "khm_Khmr",  # Khmer
    "ms": "zsm_Latn",  # Malay
    "tl": "fil_Latn",  # Filipino (Tagalog)
    "my": "mya_Mymr",  # Burmese
    "lo": "lao_Laoo",  # Lao
}

LANGUAGES = list(LANGUAGE_HF_CONFIGS)
