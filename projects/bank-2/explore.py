#!/usr/bin/env python3
"""شناخت داده — بدون هیچ وابستگی. اول این را اجرا کن.

    python explore.py
"""

import collections
import csv
import io
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", 'tayaee-bank-customer-churn-prediction-full.csv')
TARGET = 'Churn'
DELIMITER = ','


def load():
    with io.open(DATA, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter=DELIMITER))


def main():
    rows = load()
    if not rows:
        print("فایل خالی است.")
        return
    columns = list(rows[0].keys())
    print(f"{len(rows)} سطر · {len(columns)} ستون\n")

    for name in columns:
        values = [(r.get(name) or "").strip() for r in rows]
        filled = [v for v in values if v]
        missing = 100 * (1 - len(filled) / len(values))
        distinct = collections.Counter(filled)
        print(f"{name}")
        print(f"   خالی {missing:.1f}%  ·  یکتا {len(distinct)}")
        for value, count in distinct.most_common(3):
            print(f"   {count:>6}  {value[:50]}")
        print()

    if TARGET and TARGET in columns:
        print("=" * 50)
        print(f"توزیع ستون هدف: {TARGET}")
        counts = collections.Counter((r.get(TARGET) or "").strip() for r in rows)
        total = sum(counts.values()) or 1
        for value, count in counts.most_common(10):
            print(f"   {count:>6}  ({100 * count / total:5.1f}%)  {value[:40]}")
        if len(counts) == 2:
            share = max(counts.values()) / total
            print(f"\n   کلاس غالب {100 * share:.1f}% است.")
            print("   یعنی مدلی که همیشه همین را بگوید، همین دقت را می‌گیرد.")
            print("   پس accuracy معیار بی‌معنایی است — سراغ ROC-AUC برو.")
    elif TARGET:
        print(f"هشدار: ستون هدف {TARGET!r} در فایل نیست.")
    else:
        print("ستون هدفی تعیین نشده. SPEC.md را بخوان.")


if __name__ == "__main__":
    main()
