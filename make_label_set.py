#!/usr/bin/env python3
import pandas as pd, numpy as np, os, argparse

EDU_COLS   = ["hit_StartProgram","hit_SwitchProgram","hit_Dropout","hit_Graduation"]
INTV_COLS  = ["hit_PhoneScreen","hit_OA","hit_Onsite"]
OFFER_COLS = ["hit_OfferReceived","hit_OfferAccepted","hit_OfferDeclined","hit_ReturnOffer"]

def add_buckets(df):
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.lower().str.slice(0,2000)
    df["bucket"] = "other"
    df.loc[df[EDU_COLS].sum(axis=1)  > 0, "bucket"] = "education"
    df.loc[df[INTV_COLS].sum(axis=1) > 0, "bucket"] = "interview"
    df.loc[df[OFFER_COLS].sum(axis=1)> 0, "bucket"] = "offers"
    # quick heuristics to rank for easier labeling
    df["has_qmark"] = df["text"].str.contains(r"\?")
    df["has_first_person"] = df["text"].str.contains(r"\b(i|my|me|mine|i've|i got|i accepted|i graduated|i switched)\b")
    df["rank_score"] = 2*df["has_first_person"].astype(int) + (1 - df["has_qmark"].astype(int))
    return df

def stratified_take(df, plan):
    picked=[]
    for b, n in plan.items():
        sub = df[df["bucket"]==b].copy()
        if sub.empty: continue
        sub = sub.sort_values("rank_score", ascending=False)
        # keep top 70% by rank + 30% random to preserve variety
        top_n = int(n*0.7)
        head = sub.head(top_n)
        rest = sub.iloc[top_n:]
        if len(rest) > 0:
            tail = rest.sample(min(n - len(head), len(rest)), random_state=42)
            block = pd.concat([head, tail])
        else:
            block = head
        picked.append(block.head(n))
    return pd.concat(picked) if picked else pd.DataFrame([])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="output/candidates_v2.csv")
    ap.add_argument("--out", default="output/label_500.csv")
    ap.add_argument("--total", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.candidates)
    df = add_buckets(df)

    # target split across the 3 sections
    per = args.total // 3
    plan = {"education": per, "interview": per, "offers": args.total - 2*per}

    picked = stratified_take(df, plan)
    short = args.total - len(picked)
    if short > 0:
        pool = df[df["bucket"].isin(["education","interview","offers"]) & ~df.index.isin(picked.index)]
        if not pool.empty:
            picked = pd.concat([picked, pool.sample(min(short, len(pool)), random_state=7)])

    picked = picked.sample(frac=1.0, random_state=123)

    # labeling columns you will fill by hand
    cols = ["author","created_utc","type","title","body","text","bucket","has_first_person","has_qmark","id","permalink"]
    picked = picked[cols]
    picked["event_type"] = ""   # Education: StartProgram/SwitchProgram/Dropout/Graduation; Interview: PhoneScreen/OA/Onsite; Offers: OfferReceived/OfferAccepted/OfferDeclined/ReturnOffer; or None
    picked["personal"] = ""     # Personal / Not-personal
    picked["temporal"] = ""     # Retrospective / Contemporaneous / Prospective
    picked["event_time_hint"] = ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    picked.to_csv(args.out, index=False)
    print(f"Wrote labeling file → {args.out} (rows={len(picked)})")
