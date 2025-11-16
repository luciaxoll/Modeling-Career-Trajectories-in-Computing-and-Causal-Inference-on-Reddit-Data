# label_and_build_timelines.py
# 1) Label up to ALL rows in candidates_v2.csv using an LLM + your seed examples
# 2) Then build per-user timelines sorted by created_utc
#
# Output 1 (row-level, merged across runs):  output/candidates_v2_labeled_llm_clean.csv
# Output 2 (timeline from merged labels):    output/user_timelines.csv

import os
import json
import random
import time
import re
import hashlib
from typing import List, Dict, Any, Optional

import pandas as pd
from openai import OpenAI

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_PATH   = "output/candidates_v2.csv"    # ~200k rows
SEED_PATH    = "output/label_500.csv"        # has event_type + text/body
ROW_OUTPUT   = "output/candidates_v2_labeled_llm_clean.csv"   # NEW clean merged file
TL_OUTPUT    = "output/user_timelines.csv"

MAX_ROWS             = 10000
FEWSHOTS_PER_LABEL   = 4           # 3–6 good for cost
TEMPERATURE          = 0.0
MODEL                = "gpt-4o-mini"
SEED                 = 42
CHECKPOINT_EVERY     = 500       # write partial CSV every N newly labeled rows

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------
RUBRIC = (
    "You are a precise annotator for short career-related comments.\n"
    "Your job is to assign exactly ONE label from {Graduation, Interview, Got an Offer} "
    "or 'None' if unsure.\n\n"
    "Use meaning, not keywords:\n"
    "- Graduation: The AUTHOR of the comment states they graduated, finished their degree/program, "
    "  had convocation, completed requirements, or are officially graduating.\n"
    "- Interview: The AUTHOR of the comment reports having (or scheduling/attending) an interview "
    "  (phone/onsite/technical/HR), or is discussing performance on one they had. A confirmed/scheduled "
    "  interview in the near future (e.g. 'onsite Friday', 'interview tomorrow/next week') is also Interview.\n"
    "- Got an Offer: The AUTHOR of the comment states they received an offer (intern/full-time/return offer), "
    "  or explicitly says they ‘got an offer’ or ‘received an offer’. Accepting/declining still counts.\n\n"
    "Disambiguation:\n"
    "- If the comment is advice, hypothetical, second-hand, or only asking about interviews/offers, return 'None'.\n"
    "- Only label if the event clearly happened TO THE AUTHOR of the comment, not to the person they are replying to.\n\n"
    "Temporal rule (strict): Only label if you are confident the event happened to the author within about 31 days "
    "before or after the comment date, OR if it is a clearly scheduled/confirmed event within the next 31 days.\n"
    "If timing is vague, return 'None'. If unsure, return 'None'.\n"
    "Return the decision via the provided JSON tool schema only."
)

RECENCY_INSTRUCTIONS = (
    "Decide if the event described in the text is temporally close to the comment date (±31 days).\n"
    "- Past mentions within 31 days → true\n"
    "- Confirmed/scheduled events within the next 31 days → true\n"
    "- Distant past (>31 days) or vague/unscheduled future → false\n"
    "Be conservative — if uncertain, answer false.\n"
    "Consider phrases like: today, yesterday, tomorrow, next week, upcoming, this week/month, "
    "last week/month, N days/weeks ago, explicit month/day.\n"
    "Return JSON only."
)

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
client = OpenAI(api_key=api_key)

def tool_schema(name: str, properties: Dict[str, Any], required: List[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Return a strict JSON object.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
        }
    }

LABEL_TOOL = tool_schema(
    "LabelDecision",
    {
        "label":  {"type": "string", "enum": ["Graduation", "Interview", "Got an Offer", "None"]},
        "reason": {"type": "string"}
    },
    ["label", "reason"]
)

RECENCY_TOOL = tool_schema(
    "RecencyCheck",
    {
        "within_one_month": {"type": "boolean"},
        "reason":           {"type": "string"}
    },
    ["within_one_month", "reason"]
)

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------
def pick_text_from_row(row: pd.Series) -> str:
    """Prefer 'body', then 'text'."""
    b = str(row.get("body")) if "body" in row else ""
    t = str(row.get("text")) if "text" in row else ""
    b = b.strip() if b and b.strip() else ""
    t = t.strip() if t and t.strip() else ""
    return b if b else t

def normalize_seed_label(x: str) -> str:
    if not isinstance(x, str):
        return "None"
    v = x.strip().lower()
    if v in {"graduation","graduate","graduated","convocation"}:
        return "Graduation"
    if v in {"interview","interviewed","phone screen","onsite","on-site","technical interview","hr interview"}:
        return "Interview"
    if v in {"offer","got an offer","received an offer","return offer","full-time offer","intern offer"}:
        return "Got an Offer"
    if x in {"Graduation","Interview","Got an Offer"}:
        return x
    return "None"

def build_fewshots_from_seed(seed_path: str, per_label: int, rng_seed: int = 42) -> List[Dict[str, str]]:
    random.seed(rng_seed)
    df = pd.read_csv(seed_path)
    if "event_type" not in df.columns:
        raise ValueError(f"{seed_path} must contain 'event_type'.")
    df["__text__"]  = df.apply(pick_text_from_row, axis=1)
    df["__label__"] = df["event_type"].map(normalize_seed_label)
    df = df[df["__label__"].isin(["Graduation","Interview","Got an Offer"])]
    df = df[df["__text__"].astype(str).str.strip().ne("")]

    shots: List[Dict[str, str]] = []
    for lbl in ["Graduation","Interview","Got an Offer"]:
        block = df[df["__label__"] == lbl]
        if len(block) == 0:
            continue
        if len(block) > per_label:
            block = block.sample(per_label, random_state=rng_seed)
        for _, row in block.iterrows():
            txt = str(row["__text__"]).strip()
            if len(txt) > 280:
                txt = txt[:277] + "…"
            shots.append({"role":"user", "content": f"TEXT:\n{txt}"})
            shots.append({"role":"assistant", "content": json.dumps({
                "label": lbl,
                "reason": "Clear, explicit instance from the seed set."
            })})
    return shots

def backoff_try(fn, tries=4, base=1.6):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base**i + random.random() * 0.25)

# ----------- Canonical content key (single scheme for all runs) -----------
_WS_RE = re.compile(r"\s+")
def _norm_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = x.strip()
    return _WS_RE.sub(" ", x)

def _to_iso_for_key(v: Any) -> str:
    """
    Normalize created_utc for keying only:
    - If epoch-like digits -> parse as seconds, convert to '%Y-%m-%d %H:%M:%S%z'
    - Else try generic parse to UTC ISO
    - On failure, return original string trimmed (stable fallback)
    NOTE: We DO NOT overwrite the created_utc column; this is for keying only.
    """
    s = "" if v is None else str(v).strip()
    if not s:
        return ""
    # epoch seconds?
    if s.isdigit():
        try:
            dt = pd.to_datetime(float(s), unit="s", utc=True)
            return dt.strftime('%Y-%m-%d %H:%M:%S%z')
        except Exception:
            pass
    # generic parse
    try:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            return s  # fallback to raw (stable)
        return dt.strftime('%Y-%m-%d %H:%M:%S%z')
    except Exception:
        return s

def make_content_key(author: Any, created_utc: Any, body: Any) -> str:
    sig = f"{author or ''}|{_to_iso_for_key(created_utc)}|{_norm_text(body)}"
    return hashlib.md5(sig.encode("utf-8")).hexdigest()

def compute_row_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure body exists and compute canonical row_key from content (key uses ISO-normalized created_utc)."""
    df = df.copy()
    if "body" not in df.columns or df["body"].isna().any():
        df["body"] = df.apply(pick_text_from_row, axis=1)
    df["row_key"] = df.apply(lambda r: make_content_key(r.get("author"), r.get("created_utc"), r.get("body")), axis=1)
    return df

def repair_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute row_key canonically (ISO-normalized created_utc) and drop duplicates by row_key,
    preferring rows with a real label over 'None'. Does NOT rewrite created_utc values.
    """
    if df is None or df.empty:
        return df
    df = compute_row_key_cols(df)
    if "event_label" not in df.columns:
        df["event_label"] = "None"
    df["event_label"] = df["event_label"].fillna("None").astype(str)
    df["__has_label__"] = df["event_label"].ne("None")
    df = (df
          .sort_values("__has_label__")   # False first
          .drop(columns="__has_label__")
          .drop_duplicates("row_key", keep="last"))
    return df

# --------------------------------------------------
# LLM CALLS
# --------------------------------------------------
def call_label_tool(text: str, fewshots: List[Dict[str, str]]) -> Dict[str, Any]:
    messages = [{"role":"system","content": RUBRIC}]
    messages.extend(fewshots)
    messages.append({
        "role":"user",
        "content": f"TEXT:\n{text}\n\nReturn via the LabelDecision tool."
    })
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=messages,
        tools=[LABEL_TOOL],
        tool_choice={"type":"function","function":{"name":"LabelDecision"}}
    )
    args = resp.choices[0].message.tool_calls[0].function.arguments
    return json.loads(args)

def call_recency_tool(text: str, created_iso: str) -> Dict[str, Any]:
    messages = [
        {"role":"system","content": RECENCY_INSTRUCTIONS},
        {"role":"user","content": f"created_utc: {created_iso}\nTEXT:\n{text}\n\nReturn via the RecencyCheck tool."}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=messages,
        tools=[RECENCY_TOOL],
        tool_choice={"type":"function","function":{"name":"RecencyCheck"}}
    )
    args = resp.choices[0].message.tool_calls[0].function.arguments
    return json.loads(args)

# --------------------------------------------------
# PREFILTER (cheap)
# --------------------------------------------------
GRAD_RE = re.compile(r"\bgraduat(?:e|ed|ing|ion)\b|convocation|finished my degree|completed my program", re.I)
INT_RE  = re.compile(r"\binterview(?:ed|ing)?\b|phone[- ]?screen|onsite|on[- ]?site|hr interview|technical interview", re.I)
OFR_RE  = re.compile(r"\boffer(?:ed)?\b|got an offer|received an offer|return offer", re.I)

def looks_like_event(text: str) -> bool:
    """Return True if text seems worth sending to LLM."""
    if not text or not text.strip():
        return False
    if GRAD_RE.search(text): return True
    if INT_RE.search(text):  return True
    if OFR_RE.search(text):  return True
    return False

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    random.seed(SEED)

    # 0) Load/repair previously labeled store and collect seen keys
    prev_df: Optional[pd.DataFrame] = None
    seen_keys = set()
    if os.path.exists(ROW_OUTPUT):
        try:
            prev_df = pd.read_csv(ROW_OUTPUT)
            prev_df = repair_and_dedup(prev_df)
            seen_keys = set(prev_df["row_key"].astype(str))
            # Keep only essential cols going forward
            keep_cols = ["row_key", "author", "created_utc", "body", "event_label"]
            for c in keep_cols:
                if c not in prev_df.columns:
                    prev_df[c] = None
            prev_df = prev_df[keep_cols]
            print(f"[prev] loaded {len(prev_df)} previously labeled rows ({len(seen_keys)} unique keys)")
        except Exception as e:
            print(f"[prev] could not load previous labels ({e}); proceeding as fresh run")

    # 1) build few-shots once
    fewshots = build_fewshots_from_seed(SEED_PATH, per_label=FEWSHOTS_PER_LABEL, rng_seed=SEED)

    # 2) load all candidates
    df = pd.read_csv(INPUT_PATH)

    # ensure required columns
    if "author" not in df.columns:
        df["author"] = None
    if "created_utc" not in df.columns:
        df["created_utc"] = None

    # choose text + compute canonical keys (using ISO-normalized created_utc for keying)
    df["body"] = df.apply(pick_text_from_row, axis=1)
    df = df[df["body"].astype(str).str.strip().ne("")].copy()
    df["row_key"] = df.apply(lambda r: make_content_key(r.get("author"), r.get("created_utc"), r.get("body")), axis=1)

    # exclude anything we've already labeled in prior runs
    before_exclude = len(df)
    if seen_keys:
        df = df[~df["row_key"].astype(str).isin(seen_keys)].copy()
    # drop dupes within this pool
    df = df.drop_duplicates("row_key")
    print(f"[pool] candidates total: {before_exclude} -> unseen remaining: {len(df)}")

    # maybe limit
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED).copy()

    labeled_rows = []
    total = len(df)
    print(f"Labeling {total} new rows...")

    # helper for checkpoint/finish writes (merge with prev_df to preserve history)
    def write_progress(partial_new: Optional[pd.DataFrame], is_final: bool = False):
        # If nothing new, still write repaired prev_df on final to clean file
        if partial_new is None or partial_new.empty:
            if is_final and prev_df is not None:
                repaired = repair_and_dedup(prev_df)
                repaired.to_csv(ROW_OUTPUT, index=False)
            return

        combined = partial_new
        if prev_df is not None and not prev_df.empty:
            combined = pd.concat([prev_df, partial_new], ignore_index=True)

        # normalize labels + dedup preferring labeled rows
        combined = repair_and_dedup(combined)
        # IMPORTANT: do NOT reformat created_utc here; preserve as-is in row-level file
        combined.to_csv(ROW_OUTPUT, index=False)

    # 3) label loop
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text = row["body"]
        created = str(row["created_utc"] or "")

        # fast path: no clear career-event signals -> None
        if not looks_like_event(text):
            event_label = "None"
        else:
            # 1) semantic label
            llm_label = backoff_try(lambda: call_label_tool(text, fewshots)).get("label", "None")
            # 2) temporal/author check
            rec = backoff_try(lambda: call_recency_tool(text, created))
            within = bool(rec.get("within_one_month", False))
            event_label = llm_label if (llm_label != "None" and within) else "None"

        labeled_rows.append({
            "row_key":     row["row_key"],
            "author":      row["author"],
            "created_utc": row["created_utc"],
            "body":        text,
            "event_label": event_label,
        })

        # checkpoint
        if (i % CHECKPOINT_EVERY) == 0:
            tmp_df = pd.DataFrame(labeled_rows)
            write_progress(tmp_df, is_final=False)
            print(f"[checkpoint] wrote {len(tmp_df)} new rows (merged & cleaned) to {ROW_OUTPUT}")

    # write final labeled file (merged with any previous)
    out_new_df = pd.DataFrame(labeled_rows)
    write_progress(out_new_df, is_final=True)
    print(f"[done] wrote merged & cleaned labeled rows to {ROW_OUTPUT}")

    # 4) build per-user timeline (from merged labels)
    merged_df = pd.read_csv(ROW_OUTPUT)
    merged_df = repair_and_dedup(merged_df)  # safety pass
    tl_df = merged_df[merged_df["event_label"] != "None"].copy()

    if not tl_df.empty:
        # Parse created_utc into a sortable datetime column, but keep original column intact
        tl_df["created_dt"] = pd.to_datetime(tl_df["created_utc"], errors="coerce", utc=True)
        # If still NaT but looks numeric epoch, try as seconds
        mask_num = tl_df["created_dt"].isna() & tl_df["created_utc"].astype(str).str.fullmatch(r"\d+")
        if mask_num.any():
            tl_df.loc[mask_num, "created_dt"] = pd.to_datetime(tl_df.loc[mask_num, "created_utc"].astype(float), unit="s", utc=True)

        tl_df["created_dt_iso"] = tl_df["created_dt"].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        tl_df = tl_df.sort_values(by=["author", "created_dt", "row_key"], kind="mergesort")
        # Keep a tidy subset plus both time columns
        keep_tl = ["author", "created_utc", "created_dt_iso", "body", "event_label", "row_key"]
        tl_df[keep_tl].to_csv(TL_OUTPUT, index=False)
        print(f"[timeline] wrote {len(tl_df)} event rows to {TL_OUTPUT}")
    else:
        print("[timeline] no labeled events, timeline not created.")

if __name__ == "__main__":
    main()
