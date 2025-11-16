#!/usr/bin/env python3
import json, argparse, os, re, sys, datetime
import pandas as pd

# ---------- EDUCATION ----------
START_PROG_RE = re.compile(
    r"\b(declaring|declared|starting|started|begin(ning)?|enroll(ed|ling)|admitted|offer of admission)\b.*\b(cs|computer science|se|software engineering|ece|ce|comp sci|data science|bootcamp)\b",
    re.I)
SWITCH_PROG_RE = re.compile(
    r"\b(switch(ing|ed)?|transfer(ring|red)?|moved?)\b.*\b(major|program|to|into)\b.*\b(cs|computer science|se|software engineering|data science)\b",
    re.I)
DROPOUT_RE = re.compile(r"\b(drop(ped)?\s*out|withdrew\s*from|quit\s+(school|uni|university|college))\b", re.I)
GRAD_RE = re.compile(r"\b(graduate|graduated|graduation|convocation|commencement)\b", re.I)

# ---------- INTERVIEW ----------
PHONE_RE = re.compile(r"\b(phone\s*screen|recruiter\s*screen|hr\s*screen)\b", re.I)
OA_RE = re.compile(r"\b(online\s*assessment|oa|coding\s*challenge|take[-\s]?home)\b", re.I)
ONSITE_RE = re.compile(r"\b(onsite|on[-\s]?site|loop|final\s*round)\b", re.I)

# ---------- OFFERS ----------
OFFER_RCVD_RE = re.compile(r"\b(got|received|have)\s+(an?\s*)?(offer|return\s+offer)\b", re.I)
OFFER_ACC_RE  = re.compile(r"\b(accepted|sign(ed|ing))\s+(the\s*)?(offer|return\s+offer)\b", re.I)
OFFER_DEC_RE  = re.compile(r"\b(declined|turned\s*down|rejected)\s+(the\s*)?(offer|return\s+offer)\b", re.I)
RETURN_OFFER_RE = re.compile(r"\b(return\s+offer)\b", re.I)

def normalize_time(ts):
    if isinstance(ts, (int, float)):
        return datetime.datetime.utcfromtimestamp(ts).isoformat()
    if isinstance(ts, str):
        try: return datetime.datetime.fromisoformat(ts.replace("Z","")).isoformat()
        except Exception: return ts
    return None

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except: continue

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="r_cscareerquestions_posts.jsonl (NOT the .crswap)")
    ap.add_argument("--out", default="output/candidates_v2.csv")
    ap.add_argument("--min_chars", type=int, default=15)
    args = ap.parse_args()

    if args.input.endswith(".crswap"):
        print("Pass the real .jsonl file, not the .crswap temp file."); sys.exit(1)

    rows=[]
    for d in stream_jsonl(args.input):
        author = d.get("author") or d.get("author_name") or d.get("user")
        if not author or author == "[deleted]": 
            continue
        title = (d.get("title") or "").strip()
        body  = (d.get("selftext") or d.get("body") or d.get("text") or "").strip()
        text  = (title + " " + body).strip()
        if len(text) < args.min_chars:
            continue

        # --- section hits ---
        # Education
        hit_start   = bool(START_PROG_RE.search(text))
        hit_switch  = bool(SWITCH_PROG_RE.search(text))
        hit_drop    = bool(DROPOUT_RE.search(text))
        hit_grad    = bool(GRAD_RE.search(text))

        # Interviews
        hit_phone   = bool(PHONE_RE.search(text))
        hit_oa      = bool(OA_RE.search(text))
        hit_onsite  = bool(ONSITE_RE.search(text))

        # Offers
        hit_offer_rcvd = bool(OFFER_RCVD_RE.search(text))
        hit_offer_acc  = bool(OFFER_ACC_RE.search(text))
        hit_offer_dec  = bool(OFFER_DEC_RE.search(text))
        hit_return     = bool(RETURN_OFFER_RE.search(text))

        section_any = (
            hit_start or hit_switch or hit_drop or hit_grad or
            hit_phone or hit_oa or hit_onsite or
            hit_offer_rcvd or hit_offer_acc or hit_offer_dec or hit_return
        )
        if not section_any:
            continue

        rows.append({
            "author": author,
            "created_utc": normalize_time(d.get("created_utc") or d.get("created") or d.get("timestamp")),
            "type": "post",
            "title": title,
            "body": body,
            "id": d.get("id") or d.get("name") or "",
            "permalink": d.get("permalink") or d.get("url") or "",
            # Education flags
            "hit_StartProgram": int(hit_start),
            "hit_SwitchProgram": int(hit_switch),
            "hit_Dropout": int(hit_drop),
            "hit_Graduation": int(hit_grad),
            # Interview flags
            "hit_PhoneScreen": int(hit_phone),
            "hit_OA": int(hit_oa),
            "hit_Onsite": int(hit_onsite),
            # Offer flags
            "hit_OfferReceived": int(hit_offer_rcvd),
            "hit_OfferAccepted": int(hit_offer_acc),
            "hit_OfferDeclined": int(hit_offer_dec),
            "hit_ReturnOffer": int(hit_return),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not rows:
        print("No matches found — consider loosening patterns or checking the file path.")
        sys.exit(0)
    df = pd.DataFrame(rows).drop_duplicates(subset=["author","created_utc","id"])
    # convenience columns
    txt = (df["title"].fillna("") + " " + df["body"].fillna("")).str.lower()
    df["has_qmark"] = txt.str.contains(r"\?")
    df["has_first_person"] = txt.str.contains(r"\b(i|my|me|mine|i've|i got|i accepted|i graduated|i switched)\b")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} candidate rows → {args.out}")
