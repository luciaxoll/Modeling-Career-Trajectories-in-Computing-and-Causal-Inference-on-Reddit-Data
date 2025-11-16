# label_1k_llm.py
# Purpose: Use an LLM (no training) + my seed examples to label 1,000 rows.
# Output CSV columns: author, created_utc, body, event_label

import os, json, random, time
from typing import List, Dict, Any
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
INPUT_PATH  = "output/candidates_v2.csv"
SEED_PATH   = "output/label_500.csv"         # has event_type and body/text
OUTPUT_PATH = "output/candidates_v2_labeled_1k_llm_v3.csv"

N_ROWS                = 1000                  # randomly sample this many rows
FEWSHOTS_PER_LABEL    = 3                     
TEMPERATURE           = 0.0
MODEL                 = "gpt-4o-mini"         # may change
SEED                  = 42                    # reproducible sampling

# ----------------------------
# PROMPT
# ----------------------------
RUBRIC = (
    "You are a precise annotator for short career-related comments.\n"
    "Assign exactly ONE label from {Graduation, Interview, Got an Offer} or 'None' if unsure.\n\n"
    "Use meaning, not keywords:\n"
    "- Graduation: The AUTHOR states they graduated, finished their degree/program, had convocation, "
    "  completed requirements, or are officially graduating.\n"
    "- Interview: The AUTHOR reports having (or scheduling/attending) an interview (phone/onsite/technical/HR), "
    "  or discusses performance on one they had.\n"
    " A scheduled/confirmed interview in the near future (e.g., onsite Friday, interview tomorrow/next week) is Interview."
    "- Got an Offer: The AUTHOR states they received an offer (intern/full-time/return offer), including accepted/declined.\n\n"
    "Disambiguation rules:\n"
    "- If someone only gives advice without saying the event has happened to THEM, return 'None'.\n"
    "- If it is second-hand, hypothetical, unclear, or lacks a concrete event, return 'None'.\n\n"
    "Temporal rule (strict): Provide a label only if you are confident the event is past within ~31 days or a confirmed/scheduled event within the next ~31 days relative to the comment date. Otherwise return 'None'. \n"
    "Be conservative—when in doubt, return 'None'.\n"
    "Return the decision via the provided JSON tool schema only."
)

RECENCY_INSTRUCTIONS = (
    "Decide if the event is temporally close to the comment date (±31 days)."
    "Past mentions within 31 days → true."
    "Confirmed/scheduled events within the next 31 days → true."
    "Distant past (>31 days) or vague/unscheduled future → false."
    "Be conservative—if uncertain, answer false. "
    "Consider phrases like: today, yesterday, tomorrow, next week, upcoming, this week/month, last week/month, N days/weeks ago, explicit month/day. "
    "But you don't have to always reject future/scheduled mentions as long as it is within 31 days (e.g., next week, upcoming, scheduled for... These phrases are most likey fine). "
    "Be conservative—if uncertain, answer false. Return JSON with the provided tool schema."
)

# ----------------------------
# OPENAI CLIENT
# ----------------------------
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# ----------------------------
# UTILITIES
# ----------------------------
def pick_text_from_row(row: pd.Series) -> str:
    # Prefer 'body' then 'text'
    b = str(row.get("body")) if "body" in row else ""
    t = str(row.get("text")) if "text" in row else ""
    b = b if b and b.strip() else ""
    t = t if t and t.strip() else ""
    return b if b else t

def normalize_seed_label(x: str) -> str:
    if not isinstance(x, str):
        return "None"
    v = x.strip().lower()
    if v in {"graduation","graduate","graduated","convocation"}: return "Graduation"
    if v in {"interview","interviewed","phone screen","onsite","on-site","technical interview","hr interview"}: return "Interview"
    if v in {"offer","got an offer","received an offer","return offer","full-time offer","intern offer"}: return "Got an Offer"
    if x in {"Graduation","Interview","Got an Offer"}: return x
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
            # user → assistant style few-shots (assistant mirrors the tool output)
            shots.append({"role":"user", "content": f"TEXT:\n{txt}"})
            shots.append({"role":"assistant", "content": json.dumps({"label": lbl, "reason": "Clear, explicit instance."})})
            shots.append({"role":"user","content":
                "TEXT:\nGot my onsite with Amazon on Friday, got rejected 6 months ago. The recruiter stressed LPs when she called today. Any other things to keep in mind? I'm nervous >>"})
            shots.append({"role":"assistant","content": json.dumps({
                "label": "Interview",
                "reason": "Confirmed onsite later this week (within 31 days)."
            })})

            shots.append({"role":"user","content":
                "TEXT:\nI have an interview with Microsoft on site tomorrow, any tips? I'm worried I'll fail and don't deserve this interview."})
            shots.append({"role":"assistant","content": json.dumps({
                "label": "Interview",
                "reason": "Confirmed onsite tomorrow (within 31 days)."
            })})

    return shots

def backoff_try(fn, tries=4, base=1.6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(base**i + random.random()*0.25)

# ----------------------------
# LLM CALLS
# ----------------------------
def call_label_tool(text: str, fewshots: List[Dict[str, str]]) -> Dict[str, Any]:
    messages = [{"role":"system","content": RUBRIC}]
    messages.extend(fewshots)
    messages.append({"role":"user","content": f"TEXT:\n{text}\n\nReturn via the LabelDecision tool."})

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

# ----------------------------
# MAIN
# ----------------------------
def main():
    random.seed(SEED)

    # Build few-shots from your 100+ manual labels
    fewshots = build_fewshots_from_seed(SEED_PATH, per_label=FEWSHOTS_PER_LABEL, rng_seed=SEED)

    # Load candidates and select 1k random rows
    df = pd.read_csv(INPUT_PATH)

    # Ensure required columns exist
    if "author" not in df.columns:      df["author"] = None
    if "created_utc" not in df.columns: df["created_utc"] = None

    df["__text__"] = df.apply(pick_text_from_row, axis=1)
    df = df[df["__text__"].astype(str).str.strip().ne("")]

    if len(df) > N_ROWS:
        df = df.sample(N_ROWS, random_state=SEED).copy()

    labels = []
    for i, row in df.iterrows():
        text    = row["__text__"]
        created = str(row.get("created_utc") or "")

        # 1) Meaning label (with few-shots)
        lab = backoff_try(lambda: call_label_tool(text, fewshots)).get("label", "None")

        # 2) Temporal/author recency gate (strict one-month)
        within = backoff_try(lambda: call_recency_tool(text, created)).get("within_one_month", False)
        if lab != "None" and not within:
            lab = "None"

        labels.append(lab)

    out = pd.DataFrame({
        "author":      df["author"].values,
        "created_utc": df["created_utc"].values,
        "body":        [pick_text_from_row(r) for _, r in df.iterrows()],
        "event_label": labels
    })
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(out)} rows.")

if __name__ == "__main__":
    main()
