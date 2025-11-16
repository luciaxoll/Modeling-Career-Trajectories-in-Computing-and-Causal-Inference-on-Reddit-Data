import os
import json
import pandas as pd

# -------------------------------------------------
# 1. INPUTS — change these to your real paths
# -------------------------------------------------
CANDS_PATH = r"C:\Users\lucia\Documents\thesis\output\candidates_v2.csv"
TL_PATH    = r"C:\Users\lucia\Documents\thesis\output\user_timelines.csv"
OUT_PATH   = r"C:\Users\lucia\Documents\thesis\output\user_activity.json"

# -------------------------------------------------
# 2. LOAD FILES
# -------------------------------------------------
cands = pd.read_csv(CANDS_PATH)
timelines = pd.read_csv(TL_PATH)

# -------------------------------------------------
# 3. TEXT HELPER: prefer body, then text
# -------------------------------------------------
def pick_text(row):
    body = str(row.get("body")) if "body" in row and pd.notna(row["body"]) else ""
    text = str(row.get("text")) if "text" in row and pd.notna(row["text"]) else ""
    body = body.strip()
    text = text.strip()
    return body if body else text

cands["__text__"] = cands.apply(pick_text, axis=1)

# make sure we have created_utc column
if "created_utc" not in cands.columns:
    cands["created_utc"] = ""

# -------------------------------------------------
# 4. FIND AUTHORS WE CARE ABOUT
#    (only those that have at least 1 labeled event)
# -------------------------------------------------
authors = (
    timelines["author"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

# -------------------------------------------------
# 5. FILTER BIG FILE TO ONLY THOSE AUTHORS
# -------------------------------------------------
cands_sub = cands[cands["author"].astype(str).isin(authors)].copy()

# parse date for sorting
cands_sub["__created_dt"] = pd.to_datetime(
    cands_sub["created_utc"], errors="coerce", utc=True
)

# -------------------------------------------------
# 6. BUILD JSON: {author: [activities...]}
# -------------------------------------------------
user_activities = {}

for author in authors:
    user_rows = cands_sub[cands_sub["author"].astype(str) == author].copy()
    # sort by time
    user_rows = user_rows.sort_values(by="__created_dt")
    events = []
    for _, r in user_rows.iterrows():
        events.append({
            "created_utc": str(r.get("created_utc", "")),
            "text": r.get("__text__", ""),
            # optional: you can add more fields here
            # "subreddit": r.get("subreddit", ""),
            # "id": r.get("id", ""),
        })
    user_activities[author] = events

# -------------------------------------------------
# 7. WRITE JSON
# -------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(user_activities, f, ensure_ascii=False, indent=2)

print(f"wrote {OUT_PATH} with {len(user_activities)} users")
