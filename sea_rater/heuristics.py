"""
pipeline.md Section 5 - cheap prefilter features.

Pure-Python, dependency-free (no torch, no external NLP libs) so these can
be computed at candidate-corpus build time (scripts/build_candidate_corpus.py),
without a GPU or the rater. Feeds both the "Cheap prefilter features" list
in Section 5 and Baseline 2 ("Clean-only heuristic") in Section 7 --
`language_score` (already computed from FineWeb2's own langid metadata)
stands in for `langid_conf`.

These are intentionally simple pilot-scale heuristics, not a real NLP
pipeline: Thai, Khmer, Burmese, and Lao have no whitespace word
segmentation, so `stopword_ratio` for those languages approximates "word
count" with plain substring occurrence counts rather than a real
tokenizer.
"""

# Unicode codepoint ranges covering each pilot language's expected script,
# used by target_script_ratio to flag script-mismatched/garbled documents.
SCRIPT_RANGES = {
    "vi": [(0x0041, 0x024F), (0x1E00, 0x1EFF)],  # Latin + Latin Extended Additional (Vietnamese diacritics)
    "id": [(0x0041, 0x024F)],  # Latin
    "th": [(0x0E00, 0x0E7F)],  # Thai
    "km": [(0x1780, 0x17FF)],  # Khmer
    "ms": [(0x0041, 0x024F)],  # Latin (Malay)
    "tl": [(0x0041, 0x024F)],  # Latin (Filipino/Tagalog)
    "my": [(0x1000, 0x109F)],  # Myanmar
    "lo": [(0x0E80, 0x0EFF)],  # Lao
}

# Small hand-picked lists of common function words per language, used only
# for the cheap stopword_ratio heuristic below -- not exhaustive.
STOPWORDS = {
    "vi": {
        "và", "của", "là", "có", "trong", "được", "cho", "một", "này", "với",
        "đã", "các", "để", "không", "những", "khi", "đến", "như", "về", "từ",
        "người", "cũng", "sẽ", "đó", "ra", "nên", "còn", "vì", "nếu", "rằng",
    },
    "id": {
        "yang", "dan", "di", "itu", "dengan", "untuk", "tidak", "ini", "dari",
        "dalam", "akan", "adalah", "pada", "juga", "ke", "karena", "ada",
        "atau", "saya", "kita", "mereka", "bisa", "sudah", "saat", "oleh",
        "para", "namun", "seperti", "jika", "harus",
    },
    "th": {
        "และ", "ที่", "ใน", "การ", "เป็น", "ของ", "มี", "ได้", "ไม่", "ให้",
        "จะ", "ก็", "แต่", "หรือ", "กับ", "นี้", "ว่า", "จาก", "เพื่อ", "ต้อง",
    },
    "km": {
        "និង", "នៃ", "គឺ", "នេះ", "ជា", "បាន", "ដែល", "នៅ", "ក្នុង", "ចំពោះ",
        "ដើម្បី", "ពី", "ទៅ", "មាន", "ត្រូវ", "ដោយ", "ឬ", "ប៉ុន្តែ", "ដូច", "ថា",
    },
    "ms": {
        "yang", "dan", "di", "itu", "dengan", "untuk", "tidak", "ini", "dari",
        "dalam", "akan", "adalah", "pada", "juga", "ke", "kerana", "ada",
        "atau", "saya", "kita", "mereka", "boleh", "sudah", "oleh", "para",
        "namun", "seperti", "jika", "mesti",
    },
    "tl": {
        "ang", "ng", "sa", "na", "at", "ay", "mga", "ito", "may", "hindi",
        "kung", "para", "dahil", "siya", "ako", "kami", "tayo", "sila",
        "din", "rin", "pa", "po", "wala", "naman", "lang",
    },
    "my": {
        "နှင့်", "သည်", "များ", "တွင်", "ဖြစ်", "ရှိ", "မှ", "အတွက်",
        "လည်း", "ကို", "နဲ့", "ပါ",
    },
    "lo": {
        "ແລະ", "ແມ່ນ", "ຂອງ", "ໃນ", "ມີ", "ໄດ້", "ບໍ່", "ໃຫ້",
        "ກັບ", "ນີ້", "ວ່າ", "ຈາກ", "ຈະ", "ກໍ",
    },
}

WHITESPACE_SEGMENTED_LANGUAGES = {"vi", "id", "ms", "tl"}


def repetition_score(text):
    """1.0 = maximally repetitive, 0.0 = no detected repetition. Takes the
    worse of duplicate-line ratio (catches repeated boilerplate lines) and
    duplicate word-trigram ratio (catches repetitive prose within a
    single block of text)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_dup_ratio = 1 - len(set(lines)) / len(lines) if lines else 0.0

    words = text.split()
    if len(words) < 6:
        return line_dup_ratio
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    trigram_dup_ratio = 1 - len(set(trigrams)) / len(trigrams)

    return max(line_dup_ratio, trigram_dup_ratio)


def target_script_ratio(text, lang):
    """Fraction of alphabetic characters falling in `lang`'s expected
    Unicode script range(s) -- low values flag script-mismatched or
    garbled/mis-encoded documents."""
    ranges = SCRIPT_RANGES.get(lang, [])
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    in_script = sum(1 for ch in letters if any(lo <= ord(ch) <= hi for lo, hi in ranges))
    return in_script / len(letters)


def stopword_ratio(text, lang):
    """Approximate function-word density. Vietnamese/Indonesian/Malay/
    Filipino are whitespace-segmented, so this counts real word tokens;
    Thai/Khmer/Burmese/Lao are not, so it falls back to counting stopword
    substring occurrences per 20 characters as a rough proxy for "stopwords
    per word"."""
    stopwords = STOPWORDS.get(lang)
    if not stopwords:
        return 0.0

    if lang in WHITESPACE_SEGMENTED_LANGUAGES:
        words = text.lower().split()
        if not words:
            return 0.0
        hits = sum(1 for w in words if w.strip(".,!?;:\"'()") in stopwords)
        return hits / len(words)

    if not text:
        return 0.0
    hits = sum(text.count(sw) for sw in stopwords)
    approx_word_count = max(1, len(text) / 20)
    return min(1.0, hits / approx_word_count)


def symbol_ratio(text):
    """Fraction of characters that are punctuation/symbols (neither
    alphanumeric nor whitespace) -- high values flag boilerplate/spam."""
    if not text:
        return 0.0
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbols / len(text)


def numeric_ratio(text):
    """Fraction of characters that are digits."""
    if not text:
        return 0.0
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / len(text)


def is_latin_script(text):
    """1 if at least half of the document's alphabetic characters are
    Latin script, else 0 -- a cheap sanity flag, e.g. to catch non-Latin-
    script documents that are actually mostly untranslated Latin boilerplate."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0
    latin = sum(1 for ch in letters if ord(ch) < 0x0250 or 0x1E00 <= ord(ch) <= 0x1EFF)
    return int(latin / len(letters) >= 0.5)


def compute_cheap_features(text, lang):
    """All of Section 5's cheap prefilter features for one document,
    except langid confidence (`language_score`), which comes from
    FineWeb2's own metadata and is computed upstream in
    build_candidate_corpus.py's stream_docs, not here."""
    return {
        "repetition_score": repetition_score(text),
        "target_script_ratio": target_script_ratio(text, lang),
        "stopword_ratio": stopword_ratio(text, lang),
        "symbol_ratio": symbol_ratio(text),
        "numeric_ratio": numeric_ratio(text),
        "is_latin": is_latin_script(text),
    }
