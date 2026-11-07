"""Read-only audit of S5 candidate group sizes from prepared metadata."""
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW, PROCESSED
df = pd.read_parquet(PROCESSED / "metadata.parquet")
parents = {child: parent for parent, child in (line.split() for line in (RAW / "parent_child_rel.txt").read_text(encoding="utf-8").splitlines())}


def has_ancestor(style, ancestor):
    while True:
        if style == ancestor:
            return True
        if style not in parents:
            return False
        style = parents[style]


def tagged(raw, ancestor):
    return any(has_ancestor(style, ancestor) for style in json.loads(raw))


def tally(mask):
    part = df[mask]
    return {"images": len(part), "churches": part.church_id.nunique()}


year = df.construction_year
gothic = df.level1_style.eq("Gothic")
gothic_revival = df.all_styles.map(lambda raw: tagged(raw, "Q186363"))
romanesque = df.level1_style.eq("Romanesque")
romanesque_revival = df.all_styles.map(lambda raw: tagged(raw, "Q744373"))

gothic_conflict = gothic_revival & year.le(1550)
roman_conflict = romanesque_revival & year.le(1250)

result = {
    "scope": {"all_metadata_images": len(df), "wc14_images": int(df.wc14_style_id.notna().sum())},
    "gothic": {
        "year_constrained_original_nonrevival_le1550": tally(gothic & ~gothic_revival & year.le(1550)),
        "s5_revival_label_or_ge1800_excluding_conflict": tally((gothic_revival | (gothic & year.ge(1800))) & ~gothic_conflict),
        "strict_year_only_revival_ge1800": tally(gothic & year.ge(1800)),
        "label_only_original_nonrevival": tally(gothic & ~gothic_revival),
        "label_only_revival": tally(gothic_revival),
        "revival_label_with_year_le1550_conflict": tally(gothic_conflict),
        "nonrevival_with_year_ge1800": tally(gothic & ~gothic_revival & year.ge(1800)),
        "revival_label_missing_year": tally(gothic_revival & year.isna()),
    },
    "romanesque": {
        "year_constrained_original_nonrevival_le1250": tally(romanesque & ~romanesque_revival & year.le(1250)),
        "revival_label_excluding_conflict": tally(romanesque_revival & ~roman_conflict),
        "revival_label_all_years": tally(romanesque_revival),
        "label_only_original_nonrevival": tally(romanesque & ~romanesque_revival),
        "revival_label_with_year_le1250_conflict": tally(roman_conflict),
        "revival_label_missing_year": tally(romanesque_revival & year.isna()),
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2))
