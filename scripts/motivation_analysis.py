import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT = _REPO_ROOT / "output"
_MOT = _OUT / "motivation_outputs"

# ---------------- Paths ----------------
USER_ACTIVITY = _OUT / "user_activity.json"
COMMENT_MOT = _MOT / "comment_motivations.jsonl"
USER_MOT = _MOT / "user_motivations.csv"

# ---------------- Settings ----------------
HORIZON_DAYS = 1460  
DROP_NONE = True

INTERVIEW_EVENT = "Interview"
OFFER_EVENT = "Got an Offer"

MOTIVATION_ORDER = ["MONEY", "CAREER_SWITCH", "INTEREST", "STABILITY", "PRESTIGE"]

# ---------------- Load ----------------
with open(USER_ACTIVITY, "r", encoding="utf-8") as f:
    activity = json.load(f)

comment_mot = pd.read_json(COMMENT_MOT, lines=True)
comment_mot["created_utc"] = pd.to_datetime(comment_mot["created_utc"], errors="coerce")

user_mot = pd.read_csv(USER_MOT)

comment_mot_by_user = dict(tuple(comment_mot.groupby("user")))
top_mot_by_user = dict(zip(user_mot["user"], user_mot["top_motivation"]))

# ---------------- Build user-level times ----------------
rows = []
for user, comments in activity.items():
    df = pd.DataFrame(comments)
    if df.empty:
        continue

    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
    df = df.sort_values("created_utc").copy()

    top_mot = top_mot_by_user.get(user, "NONE")

    # comment-level motivation rows for this user
    if user in comment_mot_by_user:
        cmu = comment_mot_by_user[user].copy()
        cmu["created_utc"] = pd.to_datetime(cmu["created_utc"], errors="coerce")
        cmu = cmu.sort_values("created_utc")
    else:
        cmu = pd.DataFrame(columns=["created_utc", "label"])

    # A) start definition 1: first expression of user's dominant motivation
    if top_mot != "NONE" and not cmu.empty:
        dominant_start_candidates = cmu.loc[cmu["label"] == top_mot, "created_utc"]
        start_time_dominant = (
            dominant_start_candidates.min()
            if not dominant_start_candidates.empty
            else pd.NaT
        )
    else:
        start_time_dominant = pd.NaT

    # B) start definition 2: first motivation-labeled comment of any kind
    first_motivation_time = cmu["created_utc"].min() if not cmu.empty else pd.NaT

    # -------- Event definitions --------
    # interview-only
    interview_candidates = df.loc[df["event_label"] == INTERVIEW_EVENT, "created_utc"]
    first_interview_only = (
        interview_candidates.min() if not interview_candidates.empty else pd.NaT
    )

    # offer-only
    offer_candidates = df.loc[df["event_label"] == OFFER_EVENT, "created_utc"]
    first_offer = offer_candidates.min() if not offer_candidates.empty else pd.NaT

    # interview-like = interview OR offer
    interview_like_candidates = df.loc[
        df["event_label"].isin([INTERVIEW_EVENT, OFFER_EVENT]), "created_utc"
    ]
    first_interview_like = (
        interview_like_candidates.min() if not interview_like_candidates.empty else pd.NaT
    )

    rows.append(
        {
            "user": user,
            "top_motivation": top_mot,
            "start_time_dominant": start_time_dominant,
            "start_time_first_motivation": first_motivation_time,
            "first_interview_only": first_interview_only,
            "first_offer": first_offer,
            "first_interview_like": first_interview_like,
        }
    )

timeline = pd.DataFrame(rows)

if DROP_NONE:
    timeline = timeline[timeline["top_motivation"].fillna("NONE") != "NONE"].copy()

# ---------------- Helper: build censored durations ----------------
def make_survival_df(df, start_col, event_col, label):
    """
    Creates:
      {label}_duration_days  : observed duration if event happens within horizon else HORIZON_DAYS
      {label}_event_observed : 1 if event observed within horizon else 0
    """
    tmp = df.dropna(subset=[start_col]).copy()

    true_dur = (tmp[event_col] - tmp[start_col]).dt.total_seconds() / (3600 * 24)
    event_obs = true_dur.notna() & (true_dur >= 0) & (true_dur <= HORIZON_DAYS)

    tmp[f"{label}_duration_days"] = true_dur.where(event_obs, HORIZON_DAYS)
    tmp[f"{label}_event_observed"] = event_obs.astype(int)
    return tmp

# ---------------- Reporting helpers ----------------
def print_sizes(df, duration_col, event_col, title):
    print(f"\n{title}")
    g = (
        df.groupby("top_motivation")
        .agg(
            N=("user", "size"),
            events=(event_col, "sum"),
            censored=(event_col, lambda x: (1 - x).sum()),
            median_duration=(duration_col, "median"),
        )
        .sort_values("N", ascending=False)
    )
    print(g)

def plot_km(df, duration_col, event_col, title):
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for mot in [m for m in MOTIVATION_ORDER if m in df["top_motivation"].unique()]:
        sub = df[df["top_motivation"] == mot]
        if sub.empty:
            continue
        kmf.fit(
            sub[duration_col],
            event_observed=sub[event_col],
            label=f"{mot} (N={len(sub)})",
        )
        kmf.plot_survival_function()

    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Survival probability (not yet reached event)")
    plt.tight_layout()
    plt.show()

def km_logrank_tests(df, duration_col, event_col, group_col="top_motivation", do_pairwise=True):
    global_res = multivariate_logrank_test(
        event_durations=df[duration_col],
        groups=df[group_col],
        event_observed=df[event_col],
    )

    print("\n========== Log-rank test (global) ==========")
    print(f"Outcome: {duration_col} / {event_col} grouped by {group_col}")
    print(f"Chi-square: {global_res.test_statistic:.4f}")
    print(f"p-value:    {global_res.p_value:.6g}")

    if do_pairwise:
        pw = pairwise_logrank_test(
            df[duration_col],
            df[group_col],
            df[event_col],
        )
        print("\n========== Log-rank tests (pairwise p-values) ==========")
        print(pw.p_value)

    return global_res

def fit_cox(df, duration_col, event_col, title):
    cox_df = df[["top_motivation", duration_col, event_col]].copy()
    cox_df = pd.get_dummies(cox_df, columns=["top_motivation"], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)

    print(f"\nCox PH model: {title}")
    print(cph.summary)

def run_survival_analysis(df, start_col, event_col, label, title_prefix):
    surv = make_survival_df(df, start_col, event_col, label)

    duration_col = f"{label}_duration_days"
    observed_col = f"{label}_event_observed"

    print_sizes(
        surv, duration_col, observed_col,
        f"{title_prefix} (with censoring at {HORIZON_DAYS} days)"
    )

    plot_km(
        surv, duration_col, observed_col,
        f"Kaplan–Meier: {title_prefix} (censor at {HORIZON_DAYS} days)"
    )

    km_logrank_tests(surv, duration_col, observed_col, do_pairwise=True)
    fit_cox(surv, duration_col, observed_col, title_prefix)

    return surv

# ---------------- Analyses to run ----------------
ANALYSES = [
    # Current main analysis, renamed explicitly
    {
        "start_col": "start_time_dominant",
        "event_col": "first_interview_like",
        "label": "dom_to_interviewlike",
        "title": "Dominant-motivation start → Interview-or-Offer",
    },

    # Event-definition sensitivity
    {
        "start_col": "start_time_dominant",
        "event_col": "first_interview_only",
        "label": "dom_to_interviewonly",
        "title": "Dominant-motivation start → Interview only",
    },
    {
        "start_col": "start_time_dominant",
        "event_col": "first_offer",
        "label": "dom_to_offeronly",
        "title": "Dominant-motivation start → Offer only",
    },

    # Start-time sensitivity
    {
        "start_col": "start_time_first_motivation",
        "event_col": "first_interview_like",
        "label": "firstmot_to_interviewlike",
        "title": "First-motivation-comment start → Interview-or-Offer",
    },
    {
        "start_col": "start_time_first_motivation",
        "event_col": "first_interview_only",
        "label": "firstmot_to_interviewonly",
        "title": "First-motivation-comment start → Interview only",
    },
    {
        "start_col": "start_time_first_motivation",
        "event_col": "first_offer",
        "label": "firstmot_to_offeronly",
        "title": "First-motivation-comment start → Offer only",
    },
]

all_surv_results = {}

for spec in ANALYSES:
    print("\n" + "=" * 90)
    print(f"RUNNING: {spec['title']}")
    print("=" * 90)

    surv_df = run_survival_analysis(
        timeline,
        start_col=spec["start_col"],
        event_col=spec["event_col"],
        label=spec["label"],
        title_prefix=spec["title"],
    )
    all_surv_results[spec["label"]] = surv_df

# ============================================================
# Final percentage: % of each motivation that got interview within 2 years
# ============================================================

def print_event_percentages(df, event_col, title):
    summary = (
        df.groupby("top_motivation")
        .agg(
            total_users=("user", "size"),
            users_with_event=(event_col, "sum"),
        )
        .reset_index()
    )

    summary["percent_with_event"] = (
        100 * summary["users_with_event"] / summary["total_users"]
    )

    summary = summary.sort_values("percent_with_event", ascending=False)

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(summary.to_string(index=False))

    return summary


# Example 1: dominant motivation start -> interview OR offer within 2 years
main_surv = all_surv_results["dom_to_interviewlike"]

main_percentage_summary = print_event_percentages(
    main_surv,
    event_col="dom_to_interviewlike_event_observed",
    title="Percentage of users in each motivation group who got an interview-or-offer within 2 years"
)

print("\nFinal percentages by motivation:")
for _, row in main_percentage_summary.iterrows():
    print(
        f"{row['top_motivation']}: "
        f"{row['percent_with_event']:.1f}% "
        f"({int(row['users_with_event'])}/{int(row['total_users'])})"
    )