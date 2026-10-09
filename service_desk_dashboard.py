"""
Service Desk Command Centre - ONE-FILE version (ServiceNow + Genesys dashboard)
Run:   pip install streamlit pandas numpy scikit-learn plotly openpyxl xlrd
       streamlit run service_desk_dashboard.py
Sections:  1) file reading & cleaning  2) KPI formulas  3) scikit-learn forecasting  4) Streamlit pages
"""
from __future__ import annotations

import io
import itertools
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              IsolationForest, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# set_page_config must be the first Streamlit call
st.set_page_config(page_title="Service Desk Command Centre", page_icon="📊", layout="wide")


# ============================================================================
# 1) FILE READING, DETECTION & CLEANING
# ============================================================================
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}

HEADER_VOCAB = {
    "number", "opened", "state", "assigned to", "agent name", "agent id", "agent",
    "created", "work item", "catalogue name", "caller", "priority", "channel",
    "short description", "assignment group", "updated", "logged in", "on queue",
    "row labels", "interval start", "answered", "handle", "stage", "request",
    "opened for", "type", "wait time", "duration", "reason", "user", "category",
    "requested for", "opened by", "due date", "quantity", "sum of answered",
    "sum of handle", "sum of alert - no answer", "email", "idle", "available",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def norm_col(c) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")


def _dedupe(cols):
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 1
            out.append(c)
    return out


def smart_dt(s: pd.Series, default_dayfirst: bool = True) -> pd.Series:
    """Parse dates robustly (real Excel dates, Excel serials, or text)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    txt = s.dropna().astype(str)
    sample = txt.head(2000).str.extract(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})")
    a = pd.to_numeric(sample[0], errors="coerce")
    b = pd.to_numeric(sample[1], errors="coerce")
    dayfirst = default_dayfirst
    if txt.head(2000).str.match(r"^\s*\d{4}[-/]").mean() > 0.5:      # ISO yyyy-mm-dd
        return pd.to_datetime(s, errors="coerce")
    if (a > 12).any() and not (b > 12).any():
        dayfirst = True
    elif (b > 12).any() and not (a > 12).any():
        dayfirst = False
    try:
        return pd.to_datetime(s, dayfirst=dayfirst, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")


def dur_to_sec(v) -> float:
    """'11h 23m 48s' -> seconds ; '-' / blank -> 0"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    if isinstance(v, (int, float, np.number)):
        return float(v)
    t = str(v).strip().lower()
    if t in ("", "-", "nan", "none"):
        return 0.0
    h = re.search(r"(\d+(?:\.\d+)?)\s*h", t)
    m = re.search(r"(\d+(?:\.\d+)?)\s*m(?!s)", t)
    s = re.search(r"(\d+(?:\.\d+)?)\s*s", t)
    if not (h or m or s):
        try:
            return float(t)
        except ValueError:
            return 0.0
    return (float(h.group(1)) * 3600 if h else 0) + (float(m.group(1)) * 60 if m else 0) + (
        float(s.group(1)) if s else 0)


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def _ensure(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _find_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    best, best_score = 0, -1
    for i in range(min(max_scan, len(raw))):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist() if pd.notna(v)]
        score = sum(v in HEADER_VOCAB for v in vals)
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= 2 else 0


def read_raw(name: str, data: bytes):
    """-> list of (label, raw_df_without_header)"""
    out = []
    if name.lower().endswith((".csv", ".txt")):
        try:
            raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object,
                              encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object,
                              encoding="latin-1", low_memory=False)
        out.append((name, raw))
    else:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        for sh, raw in sheets.items():
            out.append((f"{name} › {sh}", raw))
    return out


def _to_frame(raw: pd.DataFrame):
    """Find header, return (df with normalised columns, context_text above header)."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame(), ""
    h = _find_header_row(raw)
    context = " ".join(str(v) for v in raw.iloc[:h].stack().tolist()) if h else ""
    cols = [norm_col(c) if pd.notna(c) else f"unnamed_{i}" for i, c in enumerate(raw.iloc[h].tolist())]
    df = raw.iloc[h + 1:].copy()
    df.columns = _dedupe(cols)
    df = df.loc[:, [c for c in df.columns if not c.startswith("unnamed_") or df[c].notna().any()]]
    df = df.dropna(how="all").reset_index(drop=True).infer_objects()
    return df, context


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
PREFIX_MAP = {"INC": "incident", "RITM": "request", "REQ": "request", "IMS": "interaction",
              "CHG": "change", "PRB": "problem"}


def classify(df: pd.DataFrame) -> str:
    c = set(df.columns)
    if "row_labels" in c and any(x.startswith("sum_of_") for x in c):
        return "genesys_pivot"
    if {"agent", "logged_in", "on_queue"} <= c:
        return "genesys_presence"
    if {"work_item", "reason", "user"} <= c:
        return "timeout"
    if "agent_name" in c and ({"handle"} <= c or {"answered"} <= c):
        return "genesys_perf"
    if "number" in c:
        pref = (df["number"].dropna().astype(str).str.extract(r"^([A-Za-z]+)")[0]
                .str.upper().value_counts())
        if len(pref) and pref.index[0] in PREFIX_MAP:
            return PREFIX_MAP[pref.index[0]]
    if "catalogue_name" in c:
        return "request"
    if "wait_time" in c and "opened_for" in c:
        return "interaction"
    if "opened" in c and ("caller" in c or "priority" in c):
        return "incident"
    return "unknown"


# --------------------------------------------------------------------------- #
# cleaners
# --------------------------------------------------------------------------- #
def state_group(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.lower()
    closed = t.str.contains("resolved|closed|cancel|complete|success|fail", regex=True)
    return pd.Series(np.where(closed, "Closed", "Open"), index=s.index)


def clean_incident(df, dayfirst):
    d = df.copy()
    _ensure(d, ["number", "opened", "updated", "channel", "short_description", "caller", "priority",
                "state", "category", "assignment_group", "assigned_to", "business_service_application",
                "business_resolve_time", "business_duration", "email"])
    d = d[d["number"].notna()].copy()
    d["opened"] = smart_dt(d["opened"], dayfirst)
    d["updated"] = smart_dt(d["updated"], dayfirst)
    d["priority_num"] = pd.to_numeric(d["priority"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    d["state"] = d["state"].fillna("Unknown").astype(str).str.strip()
    d["state_group"] = state_group(d["state"])
    d["channel"] = d["channel"].fillna("Unknown").astype(str).str.strip().str.title()
    d["resolve_hours"] = num(d, "business_resolve_time") / 3600
    d["business_hours"] = num(d, "business_duration") / 3600
    for c in ["assignment_group", "assigned_to", "category"]:
        d[c] = d[c].fillna("Unassigned" if c != "category" else "Uncategorised").astype(str)
    d["date"] = d["opened"].dt.normalize()
    d["month"] = d["opened"].dt.to_period("M").dt.to_timestamp()
    return d.drop_duplicates("number")


def clean_request(df, dayfirst):
    d = df.copy()
    _ensure(d, ["number", "created", "catalogue_name", "short_description", "stage", "request",
                "requested_for", "opened_by", "due_date", "quantity", "assigned_to"])
    d = d[d["number"].notna()].copy()
    d["created"] = smart_dt(d["created"], dayfirst)
    d["due_date"] = smart_dt(d["due_date"], dayfirst)
    d["opened"] = d["created"]
    d["catalogue_name"] = d["catalogue_name"].fillna("Unknown").astype(str)
    d["assigned_to"] = d["assigned_to"].fillna("Unassigned").astype(str)
    d["stage"] = d["stage"].fillna("Unknown").astype(str)
    d["state_group"] = np.where(d["stage"].str.lower().str.contains("complete|closed|cancel"), "Closed", "Open")
    d["date"] = d["opened"].dt.normalize()
    d["month"] = d["opened"].dt.to_period("M").dt.to_timestamp()
    return d.drop_duplicates("number")


def clean_change(df, dayfirst):
    d = df.copy()
    _ensure(d, ["number", "state", "opened", "close_code", "type", "priority", "assignment_group"])
    d = d[d["number"].notna()].copy()
    d["opened"] = smart_dt(d["opened"], dayfirst)
    d["month"] = d["opened"].dt.to_period("M").dt.to_timestamp()
    outcome_src = d["close_code"].fillna(d["state"]).astype(str).str.lower()
    d["outcome"] = np.select(
        [outcome_src.str.contains("fail|unsuccess|back.?out|rolled"), outcome_src.str.contains("success|complete|closed")],
        ["Failed", "Successful"], default="Other")
    return d.drop_duplicates("number")


def clean_interaction(df, dayfirst):
    d = df.copy()
    _ensure(d, ["number", "opened", "updated", "state", "type", "assigned_to", "opened_for",
                "wait_time", "duration", "first_response_wait_time", "state_reason", "short_description"])
    d = d[d["number"].notna()].copy()
    d["opened"] = smart_dt(d["opened"], dayfirst)
    d["updated"] = smart_dt(d["updated"], dayfirst)
    d["state"] = d["state"].fillna("Unknown").astype(str)
    d["abandoned"] = d["state"].str.lower().str.contains("abandon")
    for c in ["wait_time", "duration", "first_response_wait_time"]:
        d[c] = num(d, c)
    d["assigned_to"] = d["assigned_to"].fillna("Unassigned").astype(str)
    d["state_reason"] = d["state_reason"].fillna("n/a").astype(str)
    d["date"] = d["opened"].dt.normalize()
    d["hour"] = d["opened"].dt.hour
    d["weekday"] = d["opened"].dt.day_name()
    d["month"] = d["opened"].dt.to_period("M").dt.to_timestamp()
    return d.drop_duplicates("number")


def clean_timeout(df, dayfirst):
    d = df.copy()
    d["created"] = smart_dt(d["created"], dayfirst)
    d["interaction"] = d["work_item"].astype(str).str.extract(r"(IMS\d+)")[0]
    d["date"] = d["created"].dt.normalize()
    d["month"] = d["created"].dt.to_period("M").dt.to_timestamp()
    d["user"] = d["user"].fillna("Unknown").astype(str)
    return d


def clean_perf(df):
    d = df.copy()
    _ensure(d, ["agent_name", "interval_start", "media_type", "answered", "handle", "avg_handle", "avg_talk",
                "avg_hold", "avg_acw", "held", "transferred", "alert_no_answer"])
    d = d[d["agent_name"].notna()].copy()
    d["agent_name"] = d["agent_name"].astype(str).str.strip()
    d = d[~d["agent_name"].str.lower().isin(["grand total", "row labels"])]
    d["interval_start"] = smart_dt(d["interval_start"], default_dayfirst=False)
    for c in ["answered", "handle", "held", "transferred", "alert_no_answer"]:
        d[c] = num(d, c).fillna(0)
    for c in ["avg_handle", "avg_talk", "avg_hold", "avg_acw"]:
        d[c] = num(d, c)
    d["month"] = d["interval_start"].dt.to_period("M").dt.to_timestamp()
    return d


def pivot_to_perf(df, context, label):
    d = df[df["row_labels"].notna()].copy()
    out = pd.DataFrame({"agent_name": d["row_labels"].astype(str)})
    out["answered"] = num(d, "sum_of_answered") if "sum_of_answered" in d else 0
    out["handle"] = num(d, "sum_of_handle") if "sum_of_handle" in d else 0
    out["alert_no_answer"] = num(d, "sum_of_alert_no_answer") if "sum_of_alert_no_answer" in d else 0
    text = f"{context} {label}".lower()
    month = next((n for m, n in MONTHS.items() if m in text), None)
    ym = re.search(r"(20\d{2})", text)
    year = int(ym.group(1)) if ym else datetime.now().year
    out["interval_start"] = pd.Timestamp(year, month, 1) if month else pd.NaT
    out["media_type"] = "voice"
    out["src"] = "pivot"
    return out


def clean_presence(df):
    d = df.copy()
    d = d[d["agent"].notna()].copy()
    d["agent"] = d["agent"].astype(str).str.strip()
    for c in ["logged_in", "on_queue", "idle", "available", "away", "break", "meal", "not_responding", "off_queue"]:
        d[c + "_s"] = d[c].map(dur_to_sec) if c in d else 0.0
    return d.drop_duplicates()


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def load_files(files, dayfirst: bool = True):
    """files: list[(filename, bytes)] -> (dict[str, DataFrame], catalog DataFrame)"""
    buckets, catalog = {}, []
    for name, data in files:
        try:
            sheets = read_raw(name, data)
        except Exception as e:  # unreadable file
            catalog.append({"source": name, "detected": f"ERROR: {e}", "rows": 0})
            continue
        for label, raw in sheets:
            df, ctx = _to_frame(raw)
            if df.empty:
                continue
            kind = classify(df)
            rows = len(df)
            if kind == "genesys_pivot":
                buckets.setdefault("genesys_perf", []).append(pivot_to_perf(df, ctx, label))
                kind_out = "genesys_perf (pivot)"
            elif kind == "unknown":
                kind_out = "skipped (not recognised)"
            else:
                buckets.setdefault(kind, []).append(df)
                kind_out = kind
            catalog.append({"source": label, "detected": kind_out, "rows": rows})

    out = {}
    for kind, frames in buckets.items():
        raw = pd.concat(frames, ignore_index=True)
        if kind == "incident":
            out[kind] = clean_incident(raw, dayfirst)
        elif kind == "request":
            out[kind] = clean_request(raw, dayfirst)
        elif kind == "change":
            out[kind] = clean_change(raw, dayfirst)
        elif kind == "interaction":
            out[kind] = clean_interaction(raw, dayfirst)
        elif kind == "timeout":
            out[kind] = clean_timeout(raw, dayfirst)
        elif kind == "genesys_perf":
            p = clean_perf(raw)
            p["_nn"] = p.notna().sum(axis=1)
            p = p.sort_values("_nn", ascending=False)
            has_m = p["month"].notna()
            p = pd.concat([p[has_m].drop_duplicates(["agent_name", "month", "media_type"]), p[~has_m]])
            out[kind] = p.drop(columns="_nn").sort_values(["month", "agent_name"]).reset_index(drop=True)
        elif kind == "genesys_presence":
            out[kind] = clean_presence(raw)
        else:
            out[kind] = raw
    return out, pd.DataFrame(catalog)


# ============================================================================
# 2) KPI FORMULAS
# ============================================================================
DEFAULT_SLA_HOURS = {1: 4, 2: 8, 3: 24, 4: 40}      # business hours to resolve, by priority


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def find_col(df, candidates=None, contains=None):
    """first column whose name equals one of `candidates` or contains one of `contains`."""
    if df is None or df.empty:
        return None
    for c in candidates or []:
        if c in df.columns:
            return c
    for c in df.columns:
        if contains and any(k in c for k in contains):
            return c
    return None


def truthy(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.lower().isin(["true", "yes", "y", "t", "1", "1.0"])
    return (n.fillna(0) > 0) | t


def pct(a, b):
    return float(a) / float(b) * 100 if b else None


def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def fmt_hms(seconds):
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "N/A"
    seconds = int(round(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")


def repeat_contact_rate(df, who_col, time_col, days=7):
    """% of records where the same person contacted us again within `days` of their previous contact."""
    if df is None or df.empty or who_col not in df or time_col not in df:
        return None
    t = df[[who_col, time_col]].dropna().sort_values([who_col, time_col])
    if t.empty:
        return None
    gap = t.groupby(who_col)[time_col].diff()
    return float((gap <= pd.Timedelta(days=days)).sum() / len(t) * 100)


def concurrency_by_agent(df, agent="assigned_to", start="opened", end="updated"):
    rows = []
    d = df.dropna(subset=[start, end, agent])
    d = d[(d[end] >= d[start]) & (d[agent] != "Unassigned")]
    for a, g in d.groupby(agent):
        s = g[start].values.astype("datetime64[s]").astype("int64")
        e = g[end].values.astype("datetime64[s]").astype("int64")
        t = np.concatenate([s, e])
        delta = np.concatenate([np.ones(len(s)), -np.ones(len(e))])
        order = np.lexsort((delta, t))                # ends before starts on ties
        t, delta = t[order], delta[order]
        conc = np.cumsum(delta)
        span = t[-1] - t[0]
        dt = np.diff(t)
        busy = dt[conc[:-1] > 0].sum()
        avg_busy = float((conc[:-1] * dt).sum() / busy) if busy > 0 else 1.0
        rows.append({"agent": a, "chats": len(g), "max_concurrent": int(conc.max()),
                     "avg_concurrent_when_busy": round(avg_busy, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ServiceNow - incidents
# --------------------------------------------------------------------------- #
def _bus_hours_to_ref(opened: pd.Series, ref) -> np.ndarray:
    ref = pd.Timestamp(ref)
    s = opened.fillna(ref).values.astype("datetime64[D]")
    e = np.datetime64(ref.date(), "D")
    return np.clip(np.busday_count(s, e), 0, None) * 8.0


def enrich_incidents(inc: pd.DataFrame, targets: dict, ref_date, fcr_minutes: int = 15) -> pd.DataFrame:
    d = inc.copy()
    ref = pd.Timestamp(ref_date)
    d["is_open"] = d["state_group"].eq("Open")
    d["age_days"] = np.where(d["is_open"], (ref - d["opened"]).dt.total_seconds() / 86400, np.nan)
    d["target_h"] = d["priority_num"].map(targets).fillna(max(targets.values()))
    elapsed_closed = d["resolve_hours"].where(d["resolve_hours"].notna(),
                                              (d["updated"] - d["opened"]).dt.total_seconds() / 3600 * 40 / 168)
    d["elapsed_h"] = np.where(d["is_open"], _bus_hours_to_ref(d["opened"], ref), elapsed_closed)

    # SLA: use a real SLA column if the extract has one, otherwise compare with the targets above
    met = find_col(d, ["made_sla", "sla_met", "met_sla"])
    brk = find_col(d, ["has_breached", "sla_breached", "breached", "sla_breach"])
    if met:
        d["sla_met"] = truthy(d[met])
    elif brk:
        d["sla_met"] = ~truthy(d[brk])
    else:
        d["sla_met"] = d["elapsed_h"] <= d["target_h"]
    d["breached"] = ~d["sla_met"]

    fcr = find_col(d, ["first_contact_resolution", "fcr", "first_call_resolution"])
    if fcr:
        d["fcr"] = truthy(d[fcr])
    else:
        d["fcr"] = (~d["is_open"]) & (d["resolve_hours"] * 60 <= fcr_minutes)

    ro = find_col(d, ["reopen_count", "reopened", "reopened_count", "u_reopened"])
    d["reopened"] = (truthy(d[ro]) if ro else np.nan)

    esc = find_col(d, ["escalation", "escalated", "escalation_count"], contains=["escalat"])
    rea = find_col(d, ["reassignment_count"], contains=["reassign"])
    d["escalated"] = truthy(d[esc]) if esc else (truthy(d[rea]) if rea else np.nan)

    d["age_bucket"] = pd.cut(d["age_days"], [-0.01, 7, 15, 30, 10 ** 6],
                             labels=["0-7 days", "8-15 days", "16-30 days", ">30 days"])
    d["is_major"] = d["priority_num"].isin([1, 2])
    return d


def incident_summary(d: pd.DataFrame, req: pd.DataFrame | None = None) -> dict:
    n = len(d)
    closed = d[~d["is_open"]]
    elig = d[(~d["is_open"]) | d["breached"]]
    resolved_valid = closed[closed["resolve_hours"].notna()]
    out = {
        "incidents": n,
        "requests": 0 if req is None else len(req),
        "open": int(d["is_open"].sum()),
        "closed": int((~d["is_open"]).sum()),
        "sla_pct": float(elig["sla_met"].mean() * 100) if len(elig) else None,
        "breached": int(d["breached"].sum()),
        "backlog": int(d["is_open"].sum()),
        "mttr_h": safe_mean(resolved_valid["resolve_hours"]),
        "fcr_pct": pct(resolved_valid["fcr"].sum(), len(resolved_valid)),
        "reopened": None if d["reopened"].isna().all() else int(d["reopened"].fillna(False).sum()),
        "major": int(d["is_major"].sum()),
        "escalations": None if d["escalated"].isna().all() else int(d["escalated"].fillna(False).sum()),
        "self_service_pct": pct((d["channel"].str.lower().str.contains("self")).sum(), n),
        "repeat_contact_pct": repeat_contact_rate(d, "caller", "opened"),
    }
    # knowledge usage
    kcol = find_col(d, ["knowledge", "kb", "knowledge_article"], contains=["knowledge", "kb_"])
    if kcol:
        out["kb_usage"] = int(d[kcol].notna().sum())
        out["kb_source"] = kcol
    else:
        text = d[[c for c in ["short_description", "description", "work_notes", "comments"] if c in d]].astype(str).agg(" ".join, axis=1)
        out["kb_usage"] = int(text.str.contains(r"\bKB\d{5,}\b", case=False, regex=True).sum())
        out["kb_source"] = "KB numbers found in text"
    return out


def aging_table(d):
    o = d[d["is_open"]]
    g = o.groupby("age_bucket", observed=False).size().reindex(
        ["0-7 days", "8-15 days", "16-30 days", ">30 days"], fill_value=0).reset_index(name="tickets")
    over = {">7 days": int((o["age_days"] > 7).sum()), ">15 days": int((o["age_days"] > 15).sum()),
            ">30 days": int((o["age_days"] > 30).sum())}
    return g, over


def backlog_trend(d):
    """open-ticket backlog at the end of each day = opened so far - closed so far."""
    if d.empty:
        return pd.DataFrame()
    days = pd.date_range(d["opened"].min().normalize(), max(d["opened"].max(), d["updated"].max()).normalize())
    opened = d.groupby(d["opened"].dt.normalize()).size().reindex(days, fill_value=0).cumsum()
    closed_rows = d[~d["is_open"]]
    closed = closed_rows.groupby(closed_rows["updated"].dt.normalize()).size().reindex(days, fill_value=0).cumsum()
    return pd.DataFrame({"date": days, "backlog": (opened - closed).values})


def repeat_incidents(d, min_count=3, top=15):
    t = d.copy()
    t["pattern"] = (t["short_description"].fillna("").astype(str).str.lower()
                    .str.replace(r"\d+", "", regex=True).str.replace(r"[^a-z ]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    t = t[t["pattern"].str.len() > 3]
    if t.empty:
        return pd.DataFrame()
    g = t.groupby("pattern").agg(incidents=("number", "size"), callers=("caller", "nunique"),
                                 last_seen=("opened", "max")).reset_index()
    svc = (t.dropna(subset=["business_service_application"]).groupby("pattern")["business_service_application"]
           .agg(lambda s: s.mode().iat[0]))
    g["service"] = g["pattern"].map(svc)
    return g[g["incidents"] >= min_count].sort_values("incidents", ascending=False).head(top)


def top_categories(d, n=10):
    cat = d["category"].value_counts().head(n).rename_axis("category").reset_index(name="tickets")
    svc_col = "business_service_application"
    svc = (d[svc_col].dropna().astype(str).value_counts().head(n).rename_axis("service").reset_index(name="tickets")
           if svc_col in d else pd.DataFrame())
    return cat, svc


def top_drivers(d, n=10):
    t = d.copy()
    t["driver"] = t["business_service_application"].fillna(t["category"])
    return t["driver"].value_counts().head(n).rename_axis("driver").reset_index(name="tickets")


def change_summary(chg):
    if chg is None or chg.empty:
        return None
    return chg["outcome"].value_counts().rename_axis("outcome").reset_index(name="changes")


# --------------------------------------------------------------------------- #
# Genesys - calls (agent performance + presence)
# --------------------------------------------------------------------------- #
def _wavg(df, col, w):
    m = df[col].notna() & (df[w] > 0)
    return float((df.loc[m, col] * df.loc[m, w]).sum() / df.loc[m, w].sum()) if m.any() else None


def perf_agent_table(perf: pd.DataFrame, ms: bool = True, base: str = "handle") -> pd.DataFrame:
    f = 1000.0 if ms else 1.0
    if perf is None or perf.empty:
        return pd.DataFrame()
    rows = []
    for a, g in perf.groupby("agent_name"):
        b = g[base].sum()
        rows.append({
            "agent": a, "answered": g["answered"].sum(), "handled": g["handle"].sum(),
            "not_answered": g["alert_no_answer"].sum(), "transferred": g["transferred"].sum(),
            "aht_s": (_wavg(g, "avg_handle", "handle") or np.nan) / f,
            "talk_s": (_wavg(g, "avg_talk", "handle") or np.nan) / f,
            "hold_s": (_wavg(g, "avg_hold", "held") or np.nan) / f,
            "acw_s": (_wavg(g, "avg_acw", "handle") or np.nan) / f,
            "base": b,
        })
    t = pd.DataFrame(rows)
    t["offered"] = t["base"] + t["not_answered"]
    t["abandon_pct"] = np.where(t["offered"] > 0, t["not_answered"] / t["offered"] * 100, np.nan)
    t["transfer_pct"] = np.where(t["handled"] > 0, t["transferred"] / t["handled"] * 100, np.nan)
    return t[(t["offered"] > 0) | (t["handled"] > 0)].sort_values("offered", ascending=False)


def perf_summary(perf: pd.DataFrame, ms: bool = True, base: str = "handle") -> dict:
    if perf is None or perf.empty:
        return {}
    f = 1000.0 if ms else 1.0
    answered_base = perf[base].sum()
    nr = perf["alert_no_answer"].sum()
    offered = answered_base + nr
    scale = lambda v: None if v is None else v / f
    asa_col = find_col(perf, ["avg_speed_of_answer", "asa", "avg_asa"], contains=["speed_of_answer"])
    sl_col = find_col(perf, ["service_level", "service_level_pct"], contains=["service_level"])
    cb_col = find_col(perf, ["callback", "callbacks", "callback_requests"], contains=["callback"])
    adh_col = find_col(perf, ["adherence", "adherence_pct"], contains=["adherence"])
    return {
        "offered": int(offered), "answered": int(answered_base), "abandoned": int(nr),
        "abandon_pct": pct(nr, offered),
        "asa_s": scale(_wavg(perf, asa_col, "handle")) if asa_col else None,
        "service_level_pct": safe_mean(perf[sl_col]) if sl_col else None,
        "aht_s": scale(_wavg(perf, "avg_handle", "handle")),
        "talk_s": scale(_wavg(perf, "avg_talk", "handle")),
        "hold_s": scale(_wavg(perf, "avg_hold", "held")),
        "acw_s": scale(_wavg(perf, "avg_acw", "handle")),
        "transfer_pct": pct(perf["transferred"].sum(), perf["handle"].sum()),
        "callbacks": int(pd.to_numeric(perf[cb_col], errors="coerce").sum()) if cb_col else None,
        "adherence_pct": safe_mean(perf[adh_col]) if adh_col else None,
    }


def presence_summary(pres: pd.DataFrame) -> dict:
    if pres is None or pres.empty:
        return {}
    oq, idle, li = pres["on_queue_s"].sum(), pres["idle_s"].sum(), pres["logged_in_s"].sum()
    adh = find_col(pres, ["adherence", "adherence_pct"], contains=["adherence"])
    return {
        "occupancy_pct": pct(oq - idle, oq),
        "on_queue_pct": pct(oq, li),
        "adherence_pct": safe_mean(pres[adh]) if adh else None,
        "logged_in_h": li / 3600,
    }


def presence_agent_table(pres: pd.DataFrame) -> pd.DataFrame:
    if pres is None or pres.empty:
        return pd.DataFrame()
    t = pres.groupby("agent")[[c for c in pres.columns if c.endswith("_s")]].sum().reset_index()
    t["occupancy_pct"] = np.where(t["on_queue_s"] > 0, (t["on_queue_s"] - t["idle_s"]) / t["on_queue_s"] * 100, np.nan)
    t["on_queue_pct"] = np.where(t["logged_in_s"] > 0, t["on_queue_s"] / t["logged_in_s"] * 100, np.nan)
    return t[t["logged_in_s"] > 0].sort_values("occupancy_pct", ascending=False)


# --------------------------------------------------------------------------- #
# Genesys - chat (interactions)
# --------------------------------------------------------------------------- #
def chat_summary(ix: pd.DataFrame, timeouts: pd.DataFrame | None = None, sl_seconds: int = 60) -> dict:
    if ix is None or ix.empty:
        return {}
    done = ix[~ix["abandoned"]]
    to_n = 0 if timeouts is None else len(timeouts)
    out = {
        "offered": len(ix), "handled": len(done), "abandoned": int(ix["abandoned"].sum()),
        "abandon_pct": pct(ix["abandoned"].sum(), len(ix)),
        "missed": int(ix["abandoned"].sum()) + to_n,
        "timeouts": to_n,
        "asa_s": safe_mean(done["wait_time"]),
        "first_response_s": safe_mean(done["first_response_wait_time"]),
        "aht_s": safe_mean(done["duration"]),
        "service_level_pct": pct((done["wait_time"] <= sl_seconds).sum(), done["wait_time"].notna().sum()),
        "queue_wait_s": safe_mean(ix["wait_time"]),
        "repeat_contact_pct": repeat_contact_rate(ix, "opened_for", "opened"),
    }
    conc = concurrency_by_agent(done)
    out["max_concurrent"] = int(conc["max_concurrent"].max()) if len(conc) else None
    out["avg_concurrent"] = float(conc["avg_concurrent_when_busy"].mean()) if len(conc) else None
    csat = find_col(ix, ["csat", "customer_satisfaction", "satisfaction", "survey_score"], contains=["csat", "satisf"])
    out["csat"] = safe_mean(ix[csat]) if csat else None
    sent = find_col(ix, ["sentiment", "sentiment_score"], contains=["sentiment"])
    out["sentiment"] = safe_mean(ix[sent]) if sent else None
    return out


def chat_agent_table(ix: pd.DataFrame) -> pd.DataFrame:
    if ix is None or ix.empty:
        return pd.DataFrame()
    done = ix[~ix["abandoned"] & (ix["assigned_to"] != "Unassigned")]
    t = done.groupby("assigned_to").agg(chats=("number", "size"), avg_duration_s=("duration", "mean"),
                                        avg_wait_s=("wait_time", "mean"),
                                        avg_first_response_s=("first_response_wait_time", "mean")).reset_index()
    return t.sort_values("chats", ascending=False)


def hour_heatmap(ix: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    t = ix.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    return t.reindex(order).fillna(0)


# ============================================================================
# 3) FORECASTING & ANOMALY DETECTION (scikit-learn)
# ============================================================================
MODELS = ["Auto (best on hold-out)", "Gradient Boosting", "Random Forest", "Hist Gradient Boosting", "Ridge Regression"]


def get_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    if name == "Hist Gradient Boosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def count_series(df: pd.DataFrame, date_col: str, freq: str = "D") -> pd.Series:
    t = df[date_col].dropna()
    if t.empty:
        return pd.Series(dtype=float)
    t = t.dt.floor("D")
    s = pd.Series(1, index=t).sort_index()
    s = s.resample(freq).sum()
    full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(full, fill_value=0).astype(float)


def _cfg(freq):
    return ([1, 2, 3, 7, 14], 7) if freq == "D" else ([1, 2, 3, 4], 4)


def make_features(s: pd.Series, freq: str = "D") -> pd.DataFrame:
    lags, win = _cfg(freq)
    idx = s.index
    X = pd.DataFrame(index=idx)
    X["dow"] = idx.dayofweek
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    X["dom"] = idx.day
    X["month"] = idx.month
    X["week"] = idx.isocalendar().week.astype(int).values
    for l in lags:
        X[f"lag_{l}"] = s.shift(l)
    X["roll_mean"] = s.shift(1).rolling(win, min_periods=1).mean()
    X["roll_std"] = s.shift(1).rolling(win, min_periods=2).std().fillna(0)
    return X


def recursive_forecast(model, history: pd.Series, steps: int, freq: str, cols) -> pd.Series:
    vals = history.copy()
    off = pd.tseries.frequencies.to_offset(freq)
    out_idx, out_val = [], []
    for _ in range(steps):
        nxt = vals.index[-1] + off
        tmp = pd.concat([vals, pd.Series([np.nan], index=[nxt])])
        row = make_features(tmp, freq).iloc[[-1]][cols]
        p = max(float(model.predict(row)[0]), 0.0)
        vals = pd.concat([vals, pd.Series([p], index=[nxt])])
        out_idx.append(nxt)
        out_val.append(p)
    return pd.Series(out_val, index=pd.DatetimeIndex(out_idx))


def _metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    nz = actual > 0
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "WAPE %": float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9) * 100),
        "MAPE %": float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100) if nz.any() else float("nan"),
        "R²": float(r2_score(actual, pred)) if len(actual) > 1 else float("nan"),
    }


def _fit_eval(name, s, X, y, test_n, freq):
    cols = list(X.columns)
    train_y = y.iloc[:-test_n]
    model = get_model(name).fit(X.loc[train_y.index], train_y)
    hist = s.loc[: train_y.index[-1]]
    pred = recursive_forecast(model, hist, test_n, freq, cols)
    test_y = s.loc[pred.index]
    return model, pred, test_y


def train_and_forecast(series: pd.Series, model_name: str = "Auto (best on hold-out)", horizon: int = 14,
                       test_size: int = 14, freq: str = "D"):
    """returns dict with metrics, test table, forecast table, importances, leaderboard"""
    min_len = 28 if freq == "D" else 16
    if len(series) < min_len:
        return {"error": f"Need at least {min_len} {'days' if freq == 'D' else 'weeks'} of history "
                         f"(found {len(series)}). Upload more data."}
    s = series.copy()
    lags, _ = _cfg(freq)
    X = make_features(s, freq).dropna()
    y = s.loc[X.index]
    test_n = int(min(test_size, max(len(X) // 4, 3)))
    if len(X) - test_n < 12:
        return {"error": "Not enough rows left for training after creating lag features."}

    candidates = MODELS[1:] if model_name.startswith("Auto") else [model_name]
    board, best = [], None
    for name in candidates:
        _, pred, test_y = _fit_eval(name, s, X, y, test_n, freq)
        m = _metrics(test_y, pred)
        board.append({"model": name, **m})
        if best is None or m["MAE"] < best[0]:
            best = (m["MAE"], name, pred, test_y, m)
    _, best_name, pred, test_y, metrics = best

    naive = s.shift(lags[-2] if freq == "D" else 1).loc[test_y.index]            # seasonal-naive baseline
    naive = naive.fillna(s.mean())
    metrics["Baseline MAE (seasonal naive)"] = float(mean_absolute_error(test_y, naive))

    # final model on all data -> future
    final = get_model(best_name).fit(X, y)
    fut = recursive_forecast(final, s, horizon, freq, list(X.columns))
    resid_sd = float(np.std(test_y.values - pred.values)) or 1.0
    z = 1.645
    forecast = pd.DataFrame({"forecast": fut.round(1),
                             "lower": (fut - z * resid_sd).clip(lower=0).round(1),
                             "upper": (fut + z * resid_sd).round(1)})

    # importances
    if hasattr(final, "feature_importances_"):
        imp = pd.Series(final.feature_importances_, index=X.columns)
    elif hasattr(final, "named_steps"):
        imp = pd.Series(np.abs(final.named_steps["ridge"].coef_), index=X.columns)
    else:
        pi = permutation_importance(final, X, y, n_repeats=5, random_state=42)
        imp = pd.Series(pi.importances_mean, index=X.columns)
    imp = (imp / imp.sum() * 100 if imp.sum() else imp).sort_values(ascending=False)

    test_tbl = pd.DataFrame({"actual": test_y, "predicted": pred.round(1)})
    return {"model": best_name, "metrics": metrics, "test": test_tbl, "forecast": forecast,
            "importance": imp.reset_index().rename(columns={"index": "feature", 0: "importance_%"}),
            "leaderboard": pd.DataFrame(board).sort_values("MAE"), "history": s, "test_n": test_n}


def detect_anomalies(s: pd.Series, contamination: float = 0.05) -> pd.Series:
    if len(s) < 14:
        return pd.Series(False, index=s.index)
    dow_med = s.groupby(s.index.dayofweek).transform("median")
    X = pd.DataFrame({"v": s.values, "dev": (s - dow_med).values})
    flag = IsolationForest(contamination=contamination, random_state=42).fit_predict(X) == -1
    return pd.Series(flag, index=s.index)


def fte_required(daily_volume: float, aht_seconds: float, productive_hours: float = 7.5,
                 shrinkage: float = 0.30, occupancy: float = 0.85) -> float:
    """agents needed per day = workload hours / (productive hours x (1-shrinkage) x occupancy)"""
    workload_h = daily_volume * aht_seconds / 3600
    return workload_h / (productive_hours * (1 - shrinkage) * occupancy)


# ============================================================================
# 4) STREAMLIT APP
# ============================================================================

_NEW_ST = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)
_ctr = itertools.count()
PALETTE = ["#2E86AB", "#F6AE2D", "#F26419", "#33658A", "#55DDE0", "#758E4F", "#9B5DE5", "#EF476F"]
px.defaults.color_discrete_sequence = PALETTE

st.markdown("""
<style>
 [data-testid="stMetric"] {background:#f7f9fc;border:1px solid #e3e8ef;border-radius:10px;padding:10px 14px;}
 [data-testid="stMetricLabel"] p {font-size:0.80rem;color:#586174;}
 .block-container {padding-top:1.4rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# small UI helpers
# --------------------------------------------------------------------------- #
def show(fig, height=340):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=height, legend_title_text="")
    key = f"chart_{next(_ctr)}"
    if _NEW_ST:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, height=None):
    kw = {"height": height} if height else {}
    if _NEW_ST:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def f_int(v):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(round(v)):,}"


def f_pct(v):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}%"


def f_dec(v, suffix=""):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.1f}{suffix}"


def f_time(v):
    return fmt_hms(v)


def metric_row(items, per_row=5):
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, it in zip(cols, items[i:i + per_row]):
            label, value = it[0], it[1]
            col.metric(label, value, help=it[2] if len(it) > 2 else None)


def need(msg):
    st.info(f"ℹ️ {msg}")


def bar(df, x, y, title, color=None, orient="v", **kw):
    fig = px.bar(df, x=x, y=y, color=color, orientation=orient, title=title, text_auto=True, **kw)
    if orient == "h":
        fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# --------------------------------------------------------------------------- #
# sidebar : upload, settings, filters
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Service Desk Command Centre")
uploads = st.sidebar.file_uploader(
    "Upload Excel / CSV files (select many at once)", type=["xlsx", "xlsm", "xls", "csv"],
    accept_multiple_files=True,
    help="Incidents, requests, chat interactions, time-outs, Genesys agent performance, agent presence ... "
         "the app detects what each sheet is.")
use_sample = False

dayfirst = st.sidebar.checkbox("Text dates are day-first (dd/mm/yyyy)", value=True,
                               help="Only used for dates stored as text. Real Excel dates are never ambiguous.")

with st.sidebar.expander("⚙️ KPI settings", expanded=False):
    st.caption("SLA target - business hours to resolve")
    c1, c2 = st.columns(2)
    sla = {1: c1.number_input("P1", 1, 500, DEFAULT_SLA_HOURS[1]), 2: c2.number_input("P2", 1, 500, DEFAULT_SLA_HOURS[2]),
           3: c1.number_input("P3", 1, 500, DEFAULT_SLA_HOURS[3]), 4: c2.number_input("P4", 1, 500, DEFAULT_SLA_HOURS[4])}
    fcr_min = st.number_input("FCR = resolved within (minutes)", 0, 600, 15)
    chat_sl = st.number_input("Chat service level: answered within (sec)", 5, 600, 60)
    repeat_min = st.number_input("Repeat-incident threshold (count)", 2, 50, 3)
    unit = st.selectbox("Genesys 'Avg ...' columns are in", ["Milliseconds", "Seconds"])
    base = st.selectbox("Calls answered / handled based on", ["Handle", "Answered"],
                        help="Offered = this column + 'Alert - No Answer'")
    ref_mode = st.selectbox("Ageing / backlog reference date", ["Latest date in data", "Today"])

if not uploads and not use_sample:
    st.title("Service Desk Command Centre")
    st.markdown("""
Upload your files in the sidebar to begin - **you can select many Excel and CSV files at once**.

| What the app understands | How it recognises it |
|---|---|
| ServiceNow **incidents** | `Number` starts with `INC` |
| ServiceNow **requests** | `Number` starts with `RITM` (or a `Catalogue name` column) |
| ServiceNow **changes** (optional) | `Number` starts with `CHG` |
| Genesys **chat interactions** | `Number` starts with `IMS` |
| Chat **time-outs** | columns `Work item`, `Reason`, `User` |
| Genesys **agent performance** (raw or pivot) | `Agent Name`, `Answered`, `Handle` ... or `Row Labels` + `Sum of ...` |
| Genesys **agent presence** | `Agent`, `Logged In`, `On Queue`, `Idle` ... |

Title rows above the header, extra sheets and pivot tables are handled automatically.
""")
    st.stop()


@st.cache_data(show_spinner="Reading and classifying files ...")
def _load(file_tuples, dayfirst_flag):
    return load_files(list(file_tuples), dayfirst_flag)


file_tuples = tuple((u.name, u.getvalue()) for u in uploads)
DATA, CATALOG = _load(file_tuples, dayfirst)

with st.expander(f"📁 Files recognised ({len(CATALOG)} sheets)", expanded=False):
    table(CATALOG)
    st.caption("Sheets marked 'skipped' (e.g. summary pivots with no agent / ticket detail) are ignored.")

inc_all, req_all, chg_all = DATA.get("incident"), DATA.get("request"), DATA.get("change")
ix_all, to_all = DATA.get("interaction"), DATA.get("timeout")
perf_all, pres_all = DATA.get("genesys_perf"), DATA.get("genesys_presence")

# ---- global filters --------------------------------------------------------
stamps = [x["opened"] for x in (inc_all, req_all, ix_all) if x is not None and len(x)]
if stamps:
    allts = pd.concat(stamps).dropna()
    lo, hi = allts.min().date(), allts.max().date()
    st.sidebar.markdown("### Filters")
    rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        d0, d1 = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1)
    else:
        d0, d1 = pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)
else:
    d0, d1 = pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")


def _rng(df, col):
    if df is None or df.empty:
        return None
    return df[(df[col] >= d0) & (df[col] < d1)].copy()


inc_f, req_f, chg_f = _rng(inc_all, "opened"), _rng(req_all, "opened"), _rng(chg_all, "opened")
ix_f, to_f = _rng(ix_all, "opened"), _rng(to_all, "created")
perf_f = None
if perf_all is not None and len(perf_all):
    m = perf_all["month"].isna() | ((perf_all["month"] + pd.offsets.MonthEnd(0) >= d0) & (perf_all["month"] < d1))
    perf_f = perf_all[m].copy()
pres_f = pres_all

if inc_f is not None and len(inc_f):
    groups = sorted(inc_f["assignment_group"].unique())
    sel_groups = st.sidebar.multiselect("Assignment group", groups, default=groups)
    inc_f = inc_f[inc_f["assignment_group"].isin(sel_groups)]
    sel_ch = st.sidebar.multiselect("Channel (tickets)", sorted(inc_f["channel"].unique()),
                                    default=sorted(inc_f["channel"].unique()))
    inc_f = inc_f[inc_f["channel"].isin(sel_ch)]

# ---- enrich ----------------------------------------------------------------
REF = None
D = None
if inc_f is not None and len(inc_f):
    REF = pd.Timestamp.today() if ref_mode == "Today" else max(inc_f["opened"].max(), inc_f["updated"].max())
    D = enrich_incidents(inc_f, sla, REF, fcr_min)
SN = incident_summary(D, req_f) if D is not None else {}
CALLS = perf_summary(perf_f, unit == "Milliseconds", base.lower()) if perf_f is not None else {}
PRES = presence_summary(pres_f)
CHAT = chat_summary(ix_f, to_f, chat_sl) if ix_f is not None and len(ix_f) else {}

page = st.sidebar.radio("Dashboard", ["1 · Executive dashboard", "2 · Self-service dashboard",
                                      "3 · Call data dashboard", "4 · Chat data dashboard",
                                      "5 · Forecast vs actual demand"])
st.sidebar.caption(f"Window: {d0.date()} → {(d1 - pd.Timedelta(days=1)).date()}")


# --------------------------------------------------------------------------- #
# shared pieces
# --------------------------------------------------------------------------- #
def servicenow_kpis():
    s = SN
    metric_row([
        ("Total Incidents Raised", f_int(s["incidents"])), ("Total Requests Raised", f_int(s["requests"])),
        ("Open Tickets", f_int(s["open"])), ("Closed Tickets", f_int(s["closed"])),
        ("SLA Achievement %", f_pct(s["sla_pct"]), "Closed tickets + open tickets already past target, vs. the SLA hours in the sidebar "
                                                   "(or the 'made_sla' column if your extract has one)."),
        ("Breached Tickets", f_int(s["breached"])), ("Ticket Backlog", f_int(s["backlog"])),
        ("MTTR (hrs, business)", f_dec(s["mttr_h"]), "Mean 'Business resolve time' of closed tickets"),
        ("FCR %", f_pct(s["fcr_pct"]), f"Closed tickets resolved within {fcr_min} min (or the FCR column if present)"),
        ("Reopened Tickets", f_int(s["reopened"]), "Needs a reopen_count / reopened column in the extract"),
        ("Major Incidents (P1/P2)", f_int(s["major"])),
        ("Escalation Count", f_int(s["escalations"]), "Needs an escalation / reassignment-count column in the extract"),
        ("Self-Service Adoption", f_pct(s["self_service_pct"]), "Share of incidents with Channel = Self-service"),
        ("Knowledge Article Usage", f_int(s["kb_usage"]), f"Source: {s['kb_source']}"),
        ("Repeat Contact Rate", f_pct(s["repeat_contact_pct"]), "Same caller raising another ticket within 7 days"),
    ])


def servicenow_charts():
    d = D
    c1, c2, c3 = st.columns(3)
    with c1:
        oc = d["state_group"].value_counts().rename_axis("state").reset_index(name="tickets")
        show(px.pie(oc, names="state", values="tickets", hole=.55, title="Open vs Closed Tickets"))
    with c2:
        ag, over = aging_table(d)
        show(bar(ag, "age_bucket", "tickets", "Open ticket ageing"))
        st.caption(f"Open **>7 days: {over['>7 days']}** · **>15 days: {over['>15 days']}** · **>30 days: {over['>30 days']}**")
    with c3:
        pr = d.groupby("priority_num").size().reset_index(name="tickets")
        pr["priority"] = "P" + pr["priority_num"].fillna(0).astype(int).astype(str)
        show(bar(pr, "priority", "tickets", "Ticket Volume by Priority"))

    c1, c2 = st.columns(2)
    with c1:
        g = d.groupby(["assignment_group", "state_group"]).size().reset_index(name="tickets")
        show(bar(g, "tickets", "assignment_group", "Ticket Volume by Assignment Group", color="state_group", orient="h"))
    with c2:
        cat, svc = top_categories(d)
        show(bar(cat, "tickets", "category", "Top 10 Categories", orient="h"))
    c1, c2 = st.columns(2)
    with c1:
        if len(svc):
            show(bar(svc, "tickets", "service", "Top 10 Business Services / Applications", orient="h"))
    with c2:
        m = d.groupby("month").size().rename("Incidents").to_frame()
        if req_f is not None and len(req_f):
            m["Requests"] = req_f.groupby("month").size()
        m = m.fillna(0).reset_index().melt("month", var_name="type", value_name="tickets")
        fig = px.bar(m, x="month", y="tickets", color="type", barmode="group", title="Month-on-Month Ticket Trend", text_auto=True)
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        bt = backlog_trend(d)
        show(px.area(bt, x="date", y="backlog", title="Ticket Backlog Trend (open at end of day)"))
    with c2:
        sl = d.groupby("month")["sla_met"].mean().mul(100).reset_index(name="SLA %")
        fig = px.line(sl, x="month", y="SLA %", markers=True, title="SLA Achievement % by Month")
        fig.update_yaxes(range=[0, 100])
        show(fig)

    st.subheader("Problem candidates / repeat incidents")
    ri = repeat_incidents(d, repeat_min)
    if len(ri):
        table(ri.rename(columns={"pattern": "issue pattern"}))
    else:
        st.caption(f"No issue pattern repeats {repeat_min}+ times in the selected window.")

    st.subheader("Change success vs failed")
    cs = change_summary(chg_f)
    if cs is None:
        need("Upload a ServiceNow Change extract (numbers starting CHG, with a State or Close code column) to see this.")
    else:
        show(px.pie(cs, names="outcome", values="changes", hole=.5, title="Change outcomes"), 300)


def genesys_kpis():
    st.markdown("#### 📞 Calls")
    if not CALLS:
        need("Upload the Genesys agent performance summary to see call KPIs.")
    else:
        c = CALLS
        metric_row([
            ("Calls Offered", f_int(c["offered"]), "Handled/answered + 'Alert - No Answer'"),
            ("Calls Answered", f_int(c["answered"])), ("Calls Abandoned / Missed", f_int(c["abandoned"]), "Uses 'Alert - No Answer' unless an abandoned column exists"),
            ("Abandon Rate %", f_pct(c["abandon_pct"])),
            ("ASA", f_time(c["asa_s"]), "Needs an 'Avg Speed of Answer' column"),
            ("Service Level %", f_pct(c["service_level_pct"]), "Needs a 'Service Level' column"),
            ("AHT", f_time(c["aht_s"])), ("Avg Talk Time", f_time(c["talk_s"])), ("Hold Time", f_time(c["hold_s"])),
            ("After Call Work (ACW)", f_time(c["acw_s"])),
            ("Agent Occupancy %", f_pct(PRES.get("occupancy_pct")), "(On queue − Idle) ÷ On queue, from the presence file"),
            ("Agent Adherence %", f_pct(c["adherence_pct"] or PRES.get("adherence_pct")),
             "Needs an adherence column (WFM export). On-queue % of logged-in time is on page 3."),
            ("Transfers %", f_pct(c["transfer_pct"])), ("Callback Requests", f_int(c["callbacks"]), "Needs a callback column"),
        ])
    st.markdown("#### 💬 Chats")
    if not CHAT:
        need("Upload the chat interaction extract (IMS...) to see chat KPIs.")
    else:
        c = CHAT
        metric_row([
            ("Chats Offered", f_int(c["offered"])), ("Chats Handled", f_int(c["handled"])),
            ("Missed Chats", f_int(c["missed"]), "Abandoned + time-outs"), ("Chat Abandon %", f_pct(c["abandon_pct"])),
            ("Chat Response Time", f_time(c["first_response_s"]), "Mean 'First response wait time'"),
            ("Queue Wait Time", f_time(c["queue_wait_s"])), ("Chat ASA", f_time(c["asa_s"])),
            ("Chat Service Level %", f_pct(c["service_level_pct"]), f"Answered within {chat_sl}s"),
            ("Chat AHT", f_time(c["aht_s"])), ("Concurrent Chats (max)", f_int(c["max_concurrent"])),
            ("Avg Concurrency", f_dec(c["avg_concurrent"]), "Average chats an agent holds at once while busy"),
            ("CSAT", f_dec(c["csat"]), "Needs a CSAT / survey column"), ("Sentiment Score", f_dec(c["sentiment"]), "Needs a sentiment column"),
            ("Repeat Contact Rate", f_pct(c["repeat_contact_pct"]), "Same person chatting again within 7 days"),
        ])


def workload_table():
    parts = []
    clean = lambda s: s.astype(str).str.strip().str.title()
    if D is not None:
        parts.append(D.assign(a=clean(D["assigned_to"])).groupby("a").size().rename("tickets"))
    if req_f is not None and len(req_f):
        parts.append(req_f.assign(a=clean(req_f["assigned_to"])).groupby("a").size().rename("requests"))
    if ix_f is not None and len(ix_f):
        x = ix_f[~ix_f["abandoned"]]
        parts.append(x.assign(a=clean(x["assigned_to"])).groupby("a").size().rename("chats"))
    if perf_f is not None and len(perf_f):
        parts.append(perf_f.assign(a=clean(perf_f["agent_name"])).groupby("a")["handle"].sum().rename("calls"))
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, axis=1).fillna(0)
    t = t[~t.index.str.lower().isin(["unassigned", "nan", "none"])]
    t["total_workload"] = t.sum(axis=1)
    mu, sd = t["total_workload"].mean(), t["total_workload"].std() or 0
    t["status"] = np.select([t["total_workload"] > mu + sd, t["total_workload"] < mu - sd],
                            ["Overloaded", "Under-utilised"], default="Balanced")
    return t.sort_values("total_workload", ascending=False).reset_index().rename(columns={"a": "agent"})


@st.cache_data(show_spinner="Training forecast model ...")
def _quick_forecast(series: pd.Series, horizon: int):
    return train_and_forecast(series, "Gradient Boosting", horizon=horizon, test_size=14, freq="D")


def forecast_chart(res, title="Forecast vs Actual Demand", unit_label="volume"):
    hist, test, fc = res["history"], res["test"], res["forecast"]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Actual", mode="lines", line=dict(color="#2E86AB"))
    fig.add_scatter(x=test.index, y=test["predicted"], name="Model on hold-out", mode="lines", line=dict(color="#F26419", dash="dot"))
    fig.add_scatter(x=fc.index, y=fc["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(117,142,79,0.25)", name="90% range")
    fig.add_scatter(x=fc.index, y=fc["forecast"], name="Forecast", mode="lines+markers", line=dict(color="#758E4F"))
    fig.update_layout(title=title, yaxis_title=unit_label)
    return fig


# --------------------------------------------------------------------------- #
# PAGE 1 - executive dashboard
# --------------------------------------------------------------------------- #
def page_exec():
    st.title("📊 Executive dashboard")
    tab_c, tab_s, tab_g = st.tabs(["🧩 Combined view", "🛠️ ServiceNow", "☎️ Genesys"])

    with tab_c:
        vol = (SN.get("incidents", 0) + SN.get("requests", 0)) if SN else None
        use_calls = bool(CALLS)
        metric_row([
            ("Ticket Volume", f_int(vol), "Incidents + requests"),
            ("Backlog", f_int(SN.get("backlog"))), ("SLA %", f_pct(SN.get("sla_pct"))), ("FCR %", f_pct(SN.get("fcr_pct"))),
            ("Calls Offered", f_int(CALLS.get("offered") if use_calls else None)),
            ("ASA", f_time(CALLS.get("asa_s") if use_calls and CALLS.get("asa_s") else CHAT.get("asa_s")),
             "Call ASA when available, otherwise chat ASA"),
            ("Abandon Rate", f_pct(CALLS.get("abandon_pct") if use_calls else CHAT.get("abandon_pct"))),
            ("AHT", f_time(CALLS.get("aht_s") if use_calls else CHAT.get("aht_s"))),
            ("CSAT", f_dec(CHAT.get("csat")), "Needs a CSAT column"), ("Sentiment Score", f_dec(CHAT.get("sentiment")), "Needs a sentiment column"),
        ])
        if D is None:
            need("Upload the ServiceNow incident extract to populate the ticket charts.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                daily = D.groupby("date").size().rename("Incidents").to_frame()
                if req_f is not None and len(req_f):
                    daily["Requests"] = req_f.groupby("date").size()
                daily = daily.fillna(0).reset_index().melt("date", var_name="type", value_name="tickets")
                show(px.line(daily, x="date", y="tickets", color="type", title="Ticket Volume (daily)"))
            with c2:
                show(bar(top_drivers(D), "tickets", "driver", "Top Ticket Drivers", orient="h"))
            c1, c2 = st.columns(2)
            with c1:
                q = D.groupby(["assignment_group", "state_group"]).size().reset_index(name="tickets")
                show(bar(q, "tickets", "assignment_group", "Queue load - tickets by assignment group", color="state_group", orient="h"))
            with c2:
                ag, over = aging_table(D)
                show(bar(ag, "age_bucket", "tickets", "Aging Tickets (open)"))
                esc = SN.get("escalations")
                st.caption(f"Open >7d **{over['>7 days']}** · >15d **{over['>15 days']}** · >30d **{over['>30 days']}** · "
                           + (f"Escalations **{esc}**" if esc is not None else f"Escalations: no column in extract - P1/P2 tickets = **{SN['major']}**"))
        wl = workload_table()
        if len(wl):
            st.subheader("Top performing / overloaded agents & queues")
            c1, c2 = st.columns([3, 2])
            with c1:
                long = wl.melt(["agent", "status"], [c for c in ["tickets", "requests", "chats", "calls"] if c in wl],
                               var_name="work type", value_name="items")
                fig = px.bar(long, x="items", y="agent", color="work type", orientation="h", title="Workload by agent")
                fig.update_layout(yaxis=dict(autorange="reversed", categoryorder="total ascending"))
                show(fig, 420)
            with c2:
                table(wl[["agent", "total_workload", "status"]], 420)
                st.caption("Overloaded = more than 1 standard deviation above the team average workload "
                           "(tickets + requests + chats + calls handled).")
        # forecast preview
        st.subheader("Forecast vs Actual Demand (incident volume)")
        if inc_all is not None and len(inc_all):
            s = count_series(inc_all, "opened", "D")
            res = _quick_forecast(s, 14)
            if "error" in res:
                need(res["error"])
            else:
                show(forecast_chart(res, "Incident volume - actual vs forecast (next 14 days)", "incidents / day"))
                st.caption("Full model controls and other demand streams are on page 5.")

    with tab_s:
        if D is None:
            need("Upload the ServiceNow incident extract (numbers starting INC).")
        else:
            servicenow_kpis()
            st.divider()
            servicenow_charts()

    with tab_g:
        genesys_kpis()
        st.divider()
        c1, c2 = st.columns(2)
        if CALLS:
            at = perf_agent_table(perf_f, unit == "Milliseconds", base.lower())
            with c1:
                show(px.bar(at.head(15), x="agent", y=["base", "not_answered"], title="Calls by agent (answered vs not answered)",
                            labels={"value": "calls", "variable": ""}))
        if CHAT:
            with c2:
                ct = chat_agent_table(ix_f)
                show(bar(ct.head(15), "assigned_to", "chats", "Chats handled by agent"))


# --------------------------------------------------------------------------- #
# PAGE 2 - self-service
# --------------------------------------------------------------------------- #
def page_self_service():
    st.title("🧑‍💻 Self-service dashboard")
    if D is None:
        need("Upload the ServiceNow incident extract (it must contain the Channel column).")
        return
    is_ss = D["channel"].str.lower().str.contains("self")
    ss, other = D[is_ss], D[~is_ss]
    if ss.empty:
        need("No tickets with Channel = Self-service in the selected window.")
        return
    ss_closed = ss[~ss["is_open"]]
    metric_row([
        ("Self-Service Tickets", f_int(len(ss))), ("Self-Service Adoption", f_pct(len(ss) / len(D) * 100), "Self-service ÷ all incidents"),
        ("Open Self-Service Tickets", f_int(ss["is_open"].sum())),
        ("SLA % (self-service)", f_pct(ss[(~ss['is_open']) | ss['breached']]['sla_met'].mean() * 100)),
        ("MTTR (hrs)", f_dec(safe_mean(ss_closed["resolve_hours"]))),
        ("Resolved at first touch", f_pct(pct(ss_closed["fcr"].sum(), len(ss_closed)))),
        ("Portal Requests (RITM)", f_int(0 if req_f is None else len(req_f))),
        ("Repeat contact rate", f_pct(repeat_contact_rate(ss, "caller", "opened"))),
        ("Unique callers using portal", f_int(ss["caller"].nunique())),
        ("Knowledge article usage", f_int(SN.get("kb_usage")), SN.get("kb_source")),
    ])
    c1, c2 = st.columns(2)
    with c1:
        by_ch = D.groupby("channel").size().reset_index(name="tickets")
        show(px.pie(by_ch, names="channel", values="tickets", hole=.5, title="Channel mix"))
    with c2:
        w = D.assign(week=D["opened"].dt.to_period("W").dt.start_time, is_ss=is_ss)
        a = (w.groupby("week")["is_ss"].mean() * 100).reset_index(name="adoption %")
        show(px.line(a, x="week", y="adoption %", markers=True, title="Weekly self-service adoption %"))
    c1, c2 = st.columns(2)
    with c1:
        show(px.line(ss.groupby("date").size().reset_index(name="tickets"), x="date", y="tickets", title="Self-service tickets per day"))
    with c2:
        h = ss.groupby(ss["opened"].dt.hour).size().reset_index(name="tickets").rename(columns={"opened": "hour"})
        show(bar(h, "hour", "tickets", "Self-service tickets by hour of day"))
    c1, c2 = st.columns(2)
    with c1:
        cat = ss["category"].value_counts().head(10).rename_axis("category").reset_index(name="tickets")
        show(bar(cat, "tickets", "category", "Top 10 self-service categories", orient="h"))
    with c2:
        dsc = (ss["short_description"].fillna("(blank)").astype(str).str.slice(0, 60).value_counts().head(10)
               .rename_axis("issue").reset_index(name="tickets"))
        show(bar(dsc, "tickets", "issue", "Top 10 self-service issues", orient="h"))
    c1, c2 = st.columns(2)
    with c1:
        m = D.assign(ch=np.where(is_ss, "Self-service", D["channel"])).groupby("ch")["resolve_hours"].mean().reset_index(name="avg hrs")
        show(bar(m, "ch", "avg hrs", "Average business resolve time by channel (hrs)"))
    with c2:
        show(px.pie(ss["state"].value_counts().rename_axis("state").reset_index(name="tickets"), names="state", values="tickets",
                    title="Self-service ticket state", hole=.5))

    st.subheader("Deflection opportunities - assisted-channel issues that could move to self-service")
    t = (other.assign(issue=other["short_description"].fillna("").astype(str).str.lower().str.replace(r"\d+", "", regex=True).str.strip())
         .query("issue.str.len() > 3", engine="python").groupby("issue").agg(tickets=("number", "size"), channels=("channel", lambda s: ", ".join(sorted(set(s)))))
         .reset_index().sort_values("tickets", ascending=False).head(15))
    if len(t):
        table(t)
    st.subheader("Requests raised through the catalogue")
    if req_f is None or req_f.empty:
        need("Upload the Requests (RITM) extract to see catalogue analytics.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            show(bar(req_f["catalogue_name"].value_counts().head(10).rename_axis("catalogue item").reset_index(name="requests"),
                     "requests", "catalogue item", "Top catalogue items", orient="h"))
        with c2:
            show(px.bar(req_f.groupby("month").size().reset_index(name="requests"), x="month", y="requests", title="Requests per month", text_auto=True))


# --------------------------------------------------------------------------- #
# PAGE 3 - call data
# --------------------------------------------------------------------------- #
def page_calls():
    st.title("☎️ Call data dashboard")
    if not CALLS:
        need("Upload the Genesys agent performance summary (raw export or pivot).")
    else:
        c = CALLS
        metric_row([
            ("Calls Offered", f_int(c["offered"])), ("Calls Answered", f_int(c["answered"])), ("Abandoned / Missed", f_int(c["abandoned"])),
            ("Abandon Rate %", f_pct(c["abandon_pct"])), ("ASA", f_time(c["asa_s"]), "Needs an 'Avg Speed of Answer' column"),
            ("Service Level %", f_pct(c["service_level_pct"]), "Needs a 'Service Level' column"), ("AHT", f_time(c["aht_s"])),
            ("Avg Talk", f_time(c["talk_s"])), ("Hold Time", f_time(c["hold_s"])), ("ACW", f_time(c["acw_s"])),
            ("Occupancy %", f_pct(PRES.get("occupancy_pct"))), ("On-queue % of logged-in", f_pct(PRES.get("on_queue_pct"))),
            ("Adherence %", f_pct(c["adherence_pct"] or PRES.get("adherence_pct")), "Needs an adherence column"),
            ("Transfers %", f_pct(c["transfer_pct"])), ("Callback Requests", f_int(c["callbacks"])),
        ])
        at = perf_agent_table(perf_f, unit == "Milliseconds", base.lower())
        mt = perf_f.dropna(subset=["month"]).groupby("month").agg(answered=(base.lower(), "sum"), not_answered=("alert_no_answer", "sum")).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            if len(mt):
                m = mt.melt("month", var_name="type", value_name="calls")
                show(px.bar(m, x="month", y="calls", color="type", barmode="stack", title="Calls per month", text_auto=True))
        with c2:
            show(px.bar(at.head(20), x="agent", y=["base", "not_answered"], title="Calls by agent - answered vs not answered", labels={"value": "calls", "variable": ""}))
        c1, c2 = st.columns(2)
        with c1:
            show(bar(at.dropna(subset=["abandon_pct"]).round(1).head(20), "agent", "abandon_pct", "Abandon / missed % by agent"))
        with c2:
            comp = at.melt("agent", ["talk_s", "hold_s", "acw_s"], var_name="component", value_name="seconds")
            show(px.bar(comp.dropna(), x="agent", y="seconds", color="component", barmode="stack", title="Handle-time components by agent (sec)"))
        c1, c2 = st.columns(2)
        with c1:
            show(bar(at.dropna(subset=["transfer_pct"]).round(1).head(20), "agent", "transfer_pct", "Transfers % by agent"))
        with c2:
            show(bar(at.dropna(subset=["aht_s"]).round(0).head(20), "agent", "aht_s", "AHT by agent (sec)"))
        with st.expander("Agent table"):
            table(at.round(1))
            st.download_button("Download agent table (CSV)", at.to_csv(index=False).encode(), "call_agent_table.csv", "text/csv")

    st.subheader("Agent presence & occupancy")
    pt = presence_agent_table(pres_f)
    if pt.empty:
        need("Upload the Genesys agent presence / status export (Agent, Logged In, On Queue, Idle, Break ...).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            cols = [c for c in ["available_s", "idle_s", "away_s", "break_s", "meal_s", "not_responding_s", "off_queue_s"] if c in pt]
            long = pt.melt("agent", cols, var_name="status", value_name="sec")
            long["hours"] = long["sec"] / 3600
            long["status"] = long["status"].str.replace("_s", "", regex=False)
            show(px.bar(long, x="agent", y="hours", color="status", title="Time by presence status (hrs)"), 380)
        with c2:
            show(bar(pt.round(1), "agent", "occupancy_pct", "Occupancy % by agent  ( (On queue − Idle) ÷ On queue )"), 380)
        table(pt[["agent", "occupancy_pct", "on_queue_pct"]].round(1))

    st.subheader("Phone-channel tickets in ServiceNow (call demand proxy)")
    if D is not None:
        ph = D[D["channel"].str.lower().str.contains("phone")]
        if len(ph):
            c1, c2 = st.columns(2)
            with c1:
                show(px.line(ph.groupby("date").size().reset_index(name="tickets"), x="date", y="tickets", title="Phone tickets per day"))
            with c2:
                hm = ph.assign(weekday=ph["opened"].dt.day_name(), hour=ph["opened"].dt.hour).groupby(["weekday", "hour"]).size().unstack(fill_value=0)
                hm = hm.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).fillna(0)
                show(px.imshow(hm, aspect="auto", title="Phone tickets - weekday × hour", color_continuous_scale="Blues"))
        else:
            st.caption("No phone-channel tickets in the window.")


# --------------------------------------------------------------------------- #
# PAGE 4 - chat data
# --------------------------------------------------------------------------- #
def page_chat():
    st.title("💬 Chat data dashboard")
    if not CHAT:
        need("Upload the chat interaction extract (numbers starting IMS).")
        return
    c = CHAT
    metric_row([
        ("Chats Offered", f_int(c["offered"])), ("Chats Handled", f_int(c["handled"])), ("Abandoned", f_int(c["abandoned"])),
        ("Abandon %", f_pct(c["abandon_pct"])), ("Missed (abandoned + time-outs)", f_int(c["missed"])),
        ("Chat Response Time", f_time(c["first_response_s"])), ("Queue Wait / ASA", f_time(c["asa_s"])),
        ("Service Level %", f_pct(c["service_level_pct"]), f"Answered within {chat_sl}s"), ("AHT", f_time(c["aht_s"])),
        ("Max concurrent chats", f_int(c["max_concurrent"])), ("Avg concurrency", f_dec(c["avg_concurrent"])),
        ("CSAT", f_dec(c["csat"])), ("Sentiment", f_dec(c["sentiment"])), ("Repeat contact %", f_pct(c["repeat_contact_pct"])),
        ("Time-outs", f_int(c["timeouts"])),
    ])
    ix = ix_f
    c1, c2 = st.columns(2)
    with c1:
        d = ix.assign(outcome=np.where(ix["abandoned"], "Abandoned", "Handled")).groupby(["date", "outcome"]).size().reset_index(name="chats")
        show(px.bar(d, x="date", y="chats", color="outcome", title="Chats per day - handled vs abandoned"))
    with c2:
        h = ix.groupby("hour").agg(chats=("number", "size"), abandon=("abandoned", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=h["hour"], y=h["chats"], name="Chats")
        fig.add_scatter(x=h["hour"], y=h["abandon"] * 100, name="Abandon %", yaxis="y2", mode="lines+markers", line=dict(color="#F26419"))
        fig.update_layout(title="Chats by hour & abandon %", yaxis2=dict(overlaying="y", side="right", title="%"))
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        hm = hour_heatmap(ix)
        show(px.imshow(hm, aspect="auto", color_continuous_scale="Blues", title="Chat demand - weekday × hour"))
    with c2:
        done = ix[~ix["abandoned"]]
        show(px.histogram(done, x="wait_time", nbins=40, title="Queue wait time distribution (sec)"))
    c1, c2 = st.columns(2)
    with c1:
        ct = chat_agent_table(ix)
        show(bar(ct.head(20), "assigned_to", "chats", "Chats handled by agent"))
    with c2:
        show(px.bar(ct.head(20).round(0), x="assigned_to", y=["avg_wait_s", "avg_first_response_s"], barmode="group",
                    title="Average wait & first response by agent (sec)", labels={"value": "sec", "variable": ""}))
    c1, c2 = st.columns(2)
    with c1:
        cc = concurrency_by_agent(ix[~ix["abandoned"]])
        if len(cc):
            show(px.bar(cc, x="agent", y=["max_concurrent", "avg_concurrent_when_busy"], barmode="group",
                        title="Concurrent chats by agent", labels={"value": "chats", "variable": ""}))
    with c2:
        if to_f is not None and len(to_f):
            show(bar(to_f["user"].value_counts().head(15).rename_axis("agent").reset_index(name="time-outs"), "agent", "time-outs", "Chat time-outs by agent"))
        else:
            show(px.pie(ix["state_reason"].replace("", "n/a").value_counts().rename_axis("reason").reset_index(name="chats"),
                        names="reason", values="chats", hole=.5, title="State reason"))
    c1, c2 = st.columns(2)
    with c1:
        show(px.histogram(ix[~ix["abandoned"]], x="duration", nbins=40, title="Chat duration distribution (sec)"))
    with c2:
        mm = ix.groupby("month").agg(chats=("number", "size"), abandoned=("abandoned", "sum")).reset_index()
        show(px.bar(mm, x="month", y=["chats", "abandoned"], barmode="group", title="Monthly chats", labels={"value": "chats", "variable": ""}))
    if D is not None:
        chat_t = D[D["channel"].str.lower().str.contains("chat")]
        if len(chat_t):
            st.subheader("Chat-channel incidents in ServiceNow")
            show(px.line(chat_t.groupby("date").size().reset_index(name="tickets"), x="date", y="tickets", title="Incidents logged via chat per day"), 280)


# --------------------------------------------------------------------------- #
# PAGE 5 - forecast vs actual (scikit-learn)
# --------------------------------------------------------------------------- #
def page_forecast():
    st.title("🔮 Forecast vs actual demand")
    st.caption("Machine-learning demand forecast built with scikit-learn: calendar + lag + rolling features, "
               "hold-out testing on the most recent days, then a recursive forecast with a 90% range.")
    streams = {}
    if inc_all is not None and len(inc_all):
        streams["Incidents (all channels)"] = (inc_all, "opened")
        for ch in ["Phone", "Chat", "Self-Service"]:
            sub = inc_all[inc_all["channel"].str.lower().str.contains(ch.lower().split("-")[0])]
            if len(sub) > 30:
                streams[f"Incidents - {ch}"] = (sub, "opened")
    if req_all is not None and len(req_all):
        streams["Requests (RITM)"] = (req_all, "opened")
    if ix_all is not None and len(ix_all):
        streams["Chats offered"] = (ix_all, "opened")
    if not streams:
        need("Upload incident, request or chat data first.")
        return
    c1, c2, c3, c4 = st.columns(4)
    src = c1.selectbox("Demand stream", list(streams))
    freq_lbl = c2.selectbox("Granularity", ["Daily", "Weekly"])
    horizon = c3.slider("Forecast horizon", 7, 60, 14) if freq_lbl == "Daily" else c3.slider("Forecast horizon (weeks)", 2, 12, 4)
    model = c4.selectbox("Model", MODELS)
    freq = "D" if freq_lbl == "Daily" else "W"
    test_n = st.slider("Hold-out period used to test the model", 7, 42, 14) if freq == "D" else st.slider("Hold-out (weeks)", 2, 8, 4)
    df, col = streams[src]
    series = count_series(df, col, freq)
    res = train_and_forecast(series, model, horizon, test_n, freq)
    if "error" in res:
        st.warning(res["error"])
        return
    m = res["metrics"]
    metric_row([("Model used", res["model"]), ("MAE (avg miss)", f_dec(m["MAE"]), "Average absolute error per period on the hold-out"),
                ("WAPE %", f_pct(m["WAPE %"]), "Total absolute error ÷ total actual volume"), ("RMSE", f_dec(m["RMSE"])),
                ("R²", f"{m['R²']:.2f}"), ("Baseline MAE", f_dec(m["Baseline MAE (seasonal naive)"]),
                                           "Error of simply repeating last week - the model should beat this")], per_row=6)
    show(forecast_chart(res, f"{src} - actual vs forecast", "per day" if freq == "D" else "per week"), 420)

    c1, c2 = st.columns(2)
    with c1:
        t = res["test"]
        fig = go.Figure()
        fig.add_bar(x=t.index, y=t["actual"], name="Actual")
        fig.add_bar(x=t.index, y=t["predicted"], name="Predicted")
        fig.update_layout(barmode="group", title="Hold-out: actual vs predicted")
        show(fig)
    with c2:
        show(bar(res["importance"].head(8).round(1), "importance_%" if "importance_%" in res["importance"] else res["importance"].columns[1],
                 "feature", "What drives the forecast (feature importance %)", orient="h"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Forecast table")
        out = res["forecast"].copy()
        out.index.name = "date"
        table(out.reset_index())
        st.download_button("Download forecast (CSV)", out.reset_index().to_csv(index=False).encode(), "forecast.csv", "text/csv")
    with c2:
        st.subheader("Model leaderboard")
        table(res["leaderboard"].round(2))

    st.subheader("Anomaly detection (IsolationForest)")
    contam = st.slider("Share of days to flag", 0.01, 0.15, 0.05, 0.01)
    flag = detect_anomalies(series, contam)
    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series.values, mode="lines", name="Volume", line=dict(color="#2E86AB"))
    fig.add_scatter(x=series.index[flag], y=series[flag], mode="markers", name="Anomaly", marker=dict(color="#EF476F", size=10))
    fig.update_layout(title="Unusual volume days")
    show(fig, 300)

    st.subheader("Capacity planning (agents needed for the forecast)")
    default_aht = (CALLS.get("aht_s") if CALLS.get("aht_s") else CHAT.get("aht_s")) or 600
    c1, c2, c3, c4 = st.columns(4)
    aht = c1.number_input("AHT (sec)", 30, 7200, int(default_aht))
    shr = c2.slider("Shrinkage %", 0, 60, 30) / 100
    occ = c3.slider("Target occupancy %", 50, 95, 85) / 100
    hrs = c4.number_input("Productive hrs / agent / day", 1.0, 12.0, 7.5)
    fc = res["forecast"]
    per_day = fc["forecast"] / (7 if freq == "W" else 1)
    fte = per_day.apply(lambda v: fte_required(v, aht, hrs, shr, occ))
    st.metric("Average agents required per day", f"{fte.mean():.1f}", help="Workload hours ÷ (productive hours × (1 − shrinkage) × occupancy)")
    show(px.bar(pd.DataFrame({"date": fc.index, "agents required": fte.round(1).values}), x="date", y="agents required",
                title="Agents required per day"), 280)


# --------------------------------------------------------------------------- #
if page.startswith("1"):
    page_exec()
elif page.startswith("2"):
    page_self_service()
elif page.startswith("3"):
    page_calls()
elif page.startswith("4"):
    page_chat()
else:
    page_forecast()
