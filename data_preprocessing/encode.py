import argparse
import unicodedata

import pandas as pd
from icu import Transliterator
from myanmartools import ZawgyiDetector

ZAWGYI_THRESHOLD = 0.5

detector = ZawgyiDetector()
converter = Transliterator.createInstance("Zawgyi-my")


def normalize_text(text, threshold=ZAWGYI_THRESHOLD):
    probability = detector.get_zawgyi_probability(text)
    is_zawgyi = probability >= threshold
    if is_zawgyi:
        text = converter.transliterate(text)
    text = unicodedata.normalize("NFC", text)
    return text, is_zawgyi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="batch3_random.csv",
        help="Input CSV with a 'text' column",
    )
    parser.add_argument(
        "--output", default="batch3_unicode.csv",
        help="Where to write the CSV with converted text",
    )
    parser.add_argument("--threshold", type=float, default=ZAWGYI_THRESHOLD)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    results = df["text"].apply(lambda t: normalize_text(t, args.threshold))
    df["text"] = results.apply(lambda r: r[0])
    is_zawgyi = results.apply(lambda r: r[1])

    df.to_csv(args.output, index=False)

    total = len(df)
    n_zawgyi = int(is_zawgyi.sum())
    print(f"Total samples: {total}")
    print(f"Zawgyi samples: {n_zawgyi} ({n_zawgyi / total:.1%})")
    print(f"Already Unicode: {total - n_zawgyi} ({(total - n_zawgyi) / total:.1%})")
    print(f"Converted output written to: {args.output}")


if __name__ == "__main__":
    main()
