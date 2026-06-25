import hashlib
import json
import difflib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import ollama

st.set_page_config(page_title="Data Intelligence Pro", layout="wide", initial_sidebar_state="expanded")
pio.templates.default = "plotly_white"

try:
    ollama.list()
except Exception:
    st.error("Ollama is not running or the model is missing. Start Ollama and pull llama3.2.")
    st.stop()

CHART_TYPES = ["bar", "line", "scatter", "histogram", "box", "heatmap", "pie", "area", "treemap"]
SEVERITY_COLORS = {"info": "#17a2b8", "warning": "#ffc107", "critical": "#dc3545"}
INSIGHT_SCHEMA = {"type", "insight", "severity", "related_columns", "chart_suggestion"}
VALID_TYPES = {"correlation", "outlier", "trend", "distribution", "pattern", "recommendation", "general"}
VALID_SEVERITIES = {"info", "warning", "critical"}
VALID_CHARTS = {"bar", "line", "scatter", "histogram", "box", "heatmap", "pie", "area", "treemap", "null"}
MAX_SAMPLE_ROWS = 50_000


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "conversation": [],
        "last_result": None,
        "dismissed_insights": set(),
        "chart_cache": {},
        "negative_flag_decision": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def _df_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_bytes) if file_name.endswith(".csv") else pd.read_excel(file_bytes)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()


@st.cache_data(show_spinner=False)
def prepare(file_bytes: bytes, file_name: str, confirm_negatives: frozenset) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    df = load_raw(file_bytes, file_name)
    original_columns = df.columns.str.strip().tolist()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    logs: list[str] = []

    before = len(df)
    df = df.dropna(how="all").drop_duplicates()
    removed = before - len(df)
    if removed:
        logs.append(f"Removed {removed} empty or duplicate rows.")

    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logs.append(f"Parsed date column: '{col}'.")

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    mixed_cols = []
    for col in obj_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        success_rate = converted.notna().sum() / max(len(df[col].dropna()), 1)
        if success_rate >= 0.8:
            df[col] = converted
        elif 0 < success_rate < 0.8:
            mixed_cols.append((col, round(success_rate * 100)))
    for col, pct in mixed_cols:
        logs.append(f"Mixed-type column '{col}': {pct}% numeric — kept as text. Review manually.")

    negative_flags: dict[str, int] = {}
    for col in df.select_dtypes(include="number").columns:
        neg = int((df[col] < 0).sum())
        if neg:
            negative_flags[col] = neg
            if col in confirm_negatives:
                df.loc[df[col] < 0, col] = None
                logs.append(f"Nullified {neg} negative values in '{col}' (confirmed by user).")
            else:
                logs.append(f"'{col}' has {neg} negative values — flagged for review.")

    preview_df = df.rename(columns=dict(zip(df.columns, original_columns)))
    return df, preview_df, logs, negative_flags


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def detect_outliers(file_hash: str, df: pd.DataFrame) -> dict[str, dict]:
    results = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (s < lower) | (s > upper)
        count = int(outlier_mask.sum())
        if count:
            z = (s - s.mean()) / s.std()
            extreme = int((z.abs() > 3).sum())
            results[col] = {
                "count": count,
                "pct": round(count / len(s) * 100, 1),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "extreme_count": extreme,
            }
    return results


# ---------------------------------------------------------------------------
# Metadata and context for AI
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_metadata(file_hash: str, df: pd.DataFrame) -> str:
    sample = df if len(df) <= MAX_SAMPLE_ROWS else df.sample(MAX_SAMPLE_ROWS, random_state=42)
    parts = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"]
    for col in sample.columns:
        s = sample[col].dropna()
        nulls = int(df[col].isna().sum())
        null_pct = round(nulls / len(df) * 100, 1)
        if s.empty:
            parts.append(f"{col}: no valid values (100% null)")
        elif pd.api.types.is_numeric_dtype(s):
            parts.append(
                f"{col} (numeric): min={s.min():.2f}, max={s.max():.2f}, "
                f"mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}, "
                f"nulls={nulls} ({null_pct}%)"
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            parts.append(f"{col} (datetime): {s.min()} to {s.max()}, nulls={nulls} ({null_pct}%)")
        else:
            mode_val = s.mode().iloc[0] if not s.mode().empty else "N/A"
            top5 = s.value_counts().head(5).to_dict()
            parts.append(
                f"{col} (categorical): {s.nunique()} unique, most_common='{mode_val}', "
                f"top_5={top5}, nulls={nulls} ({null_pct}%)"
            )
    return "\n".join(parts)


def build_sample_csv(df: pd.DataFrame) -> str:
    sample = df if len(df) <= MAX_SAMPLE_ROWS else df.sample(MAX_SAMPLE_ROWS, random_state=42)
    return sample.head(10).to_csv(index=False)


# ---------------------------------------------------------------------------
# AI analysis with retry
# ---------------------------------------------------------------------------

def _call_ollama(messages: list[dict], retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model="llama3.2",
                format="json",
                options={"temperature": 0.1},
                messages=messages,
            )
            return json.loads(response["message"]["content"])
        except json.JSONDecodeError as e:
            if attempt == retries - 1:
                return {"error": f"JSON parse error after {retries} attempts: {e}"}
            time.sleep(1.5 ** attempt)
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(1.5 ** attempt)
    return {"error": "Ollama did not respond."}


def _validate_result(result: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if "error" in result:
        return result, warnings
    insights = result.get("insights", [])
    valid = []
    for i, ins in enumerate(insights):
        missing_keys = INSIGHT_SCHEMA - ins.keys()
        if missing_keys:
            warnings.append(f"Insight {i+1} missing fields: {missing_keys} — skipped.")
            continue
        if ins.get("type") not in VALID_TYPES:
            ins["type"] = "general"
        if ins.get("severity") not in VALID_SEVERITIES:
            ins["severity"] = "info"
        cs = ins.get("chart_suggestion", {})
        if cs.get("chart_type") not in VALID_CHARTS:
            cs["chart_type"] = "null"
        valid.append(ins)
    result["insights"] = valid
    if not result.get("overall_summary"):
        warnings.append("AI returned no overall summary.")
        result["overall_summary"] = "No summary provided."
    return result, warnings


def run_ai_analysis(metadata: str, sample_csv: str, conversation: list[dict], query: str) -> tuple[dict, list[str]]:
    system_prompt = f"""You are a senior data scientist performing exploratory data analysis.

DATA PROFILE:
{metadata}

SAMPLE ROWS (up to 10):
{sample_csv}

Return strictly valid JSON only — no prose, no markdown fences:
{{
  "insights": [
    {{
      "type": "correlation | outlier | trend | distribution | pattern | recommendation | general",
      "insight": "detailed observation in plain English with concrete numbers",
      "severity": "info | warning | critical",
      "related_columns": ["col1", "col2"],
      "chart_suggestion": {{
        "chart_type": "bar | line | scatter | histogram | box | heatmap | pie | area | treemap | null",
        "x_column": "exact column name or null",
        "y_column": "exact column name or null",
        "color_column": "exact column name or null"
      }}
    }}
  ],
  "overall_summary": "concise executive summary of all findings"
}}

Rules:
- Use only column names from the data profile.
- Provide at least 3 insights when data supports it.
- Base all observations strictly on the provided profile and sample; never invent data.
- Set chart_type to null when no chart adds value.
- When the user asks a follow-up, respond only to that question and update insights accordingly.
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": query})
    result = _call_ollama(messages)
    return _validate_result(result)


# ---------------------------------------------------------------------------
# Column fuzzy matching with logging
# ---------------------------------------------------------------------------

def fuzzy_col(target: str | None, columns, fuzzy_log: list[str] | None = None) -> str | None:
    if not target or target in (None, "null", ""):
        return None
    if target in columns:
        return target
    match = difflib.get_close_matches(target, columns, n=1, cutoff=0.6)
    if match:
        if fuzzy_log is not None and match[0] != target:
            fuzzy_log.append(f"Column '{target}' not found; substituted with '{match[0]}'.")
        return match[0]
    return None


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_chart(df: pd.DataFrame, chart_type: str, x: str | None, y: str | None, color: str | None = None):
    try:
        if chart_type == "heatmap":
            corr = df.select_dtypes("number").corr()
            return px.imshow(corr, text_auto=True, title="Correlation Heatmap") if not corr.empty else None
        if chart_type == "histogram" and x:
            return px.histogram(df, x=x, color=color, title=f"Distribution of {x}")
        if chart_type == "box" and x:
            return px.box(df, x=x, y=y, color=color, title=f"Box plot: {x}" + (f" by {y}" if y else ""))
        if chart_type == "pie" and x and y:
            return px.pie(df, names=x, values=y, title=f"Pie chart: {y} by {x}")
        if chart_type == "area" and x and y:
            return px.area(df, x=x, y=y, color=color, title=f"Area chart: {y} over {x}")
        if chart_type == "treemap" and x and y:
            path = [x] + ([color] if color and color != x else [])
            return px.treemap(df, path=path, values=y, title=f"Treemap: {y} by {x}")
        if chart_type in ("bar", "line", "scatter") and x and y:
            fn = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[chart_type]
            return fn(df, x=x, y=y, color=color, title=f"{chart_type.capitalize()}: {y} vs {x}")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Quality scoring with per-column completeness
# ---------------------------------------------------------------------------

def quality_score(df: pd.DataFrame, logs: list[str]) -> int:
    if df.size == 0:
        return 0
    missing_pct = df.isna().sum().sum() / df.size * 100
    return max(0, min(100, round(100 - missing_pct * 2 - len(logs) * 2)))


def column_completeness(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        pct = round((total - nulls) / total * 100, 1) if total else 0
        rows.append({"Column": col, "Non-null": total - nulls, "Null": nulls, "Completeness (%)": pct})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def render_html_report(
    query: str,
    conversation: list[dict],
    insights: list[dict],
    summary: str,
    figs: list,
    logs: list[str],
    score: int,
    outliers: dict,
) -> str:
    def _sev_color(s):
        return SEVERITY_COLORS.get(s, "#17a2b8")

    insights_html = "".join(
        f"""<div style="border-left:4px solid {_sev_color(i.get('severity','info'))};
                padding:10px;margin:10px 0;background:#f8f9fa;">
            <strong>{i.get('type','Finding').capitalize()}</strong>
            <span style="color:{_sev_color(i.get('severity','info'))};">({i.get('severity','info')})</span>
            <p>{i.get('insight','')}</p>
            {"<small>Columns: " + ", ".join(i.get("related_columns",[])) + "</small>" if i.get("related_columns") else ""}
        </div>"""
        for i in insights
    )

    charts_html = "".join(
        pio.to_html(f, full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
        for idx, f in enumerate(figs) if f
    ) or "<p>No visualizations generated.</p>"

    outlier_html = ""
    if outliers:
        rows = "".join(
            f"<tr><td>{col}</td><td>{v['count']} ({v['pct']}%)</td>"
            f"<td>{v['lower_bound']} – {v['upper_bound']}</td><td>{v['extreme_count']}</td></tr>"
            for col, v in outliers.items()
        )
        outlier_html = f"""<h3>Outlier Summary</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;">
            <thead><tr><th>Column</th><th>Outlier Count</th><th>Expected Range (IQR)</th><th>Extreme (z&gt;3)</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    conv_html = "".join(
        f"<p><strong>{m['role'].capitalize()}:</strong> {m['content']}</p>"
        for m in conversation if m["role"] in ("user", "assistant")
    )

    logs_html = "".join(f"<li>{l}</li>" for l in logs)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Data Intelligence Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #2c3e50; line-height: 1.6; }}
    h1, h3 {{ color: #2c3e50; }}
    .box {{ background:#f4f6f7; padding:20px; border-radius:6px; }}
    table {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Data Intelligence Report</h1>
  <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <h3>Query</h3><p>{query}</p>
  <h3>Conversation History</h3><div class="box">{conv_html or "<p>Single-turn analysis.</p>"}</div>
  <h3>Executive Summary</h3><div class="box">{summary}</div>
  <h3>Detailed Findings</h3>{insights_html}
  {outlier_html}
  <h3>Visualizations</h3>{charts_html}
  <h3>Data Quality</h3>
  <p>Quality Score: <strong>{score}/100</strong></p>
  <ul>{logs_html}</ul>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sidebar: persistent dataset summary
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame, cleaning_logs: list[str], outliers: dict) -> None:
    with st.sidebar:
        st.header("Dataset Summary")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        mem_mb = round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        st.metric("Memory", f"{mem_mb} MB")
        score = quality_score(df, cleaning_logs)
        color = "normal" if score >= 80 else "inverse"
        st.metric("Quality Score", f"{score}/100", delta_color=color)
        if outliers:
            st.warning(f"{len(outliers)} column(s) have outliers detected.")
        st.divider()
        st.subheader("Column Completeness")
        comp = column_completeness(df)
        st.dataframe(comp[["Column", "Completeness (%)"]].set_index("Column"), use_container_width=True)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Data Intelligence Pro")
st.caption("AI-powered analytics, cleaning, and dynamic reporting")

uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

file_bytes = uploaded_file.read()
file_hash = _df_hash(file_bytes)

if st.session_state.get("_last_file_hash") != file_hash:
    st.session_state["conversation"] = []
    st.session_state["last_result"] = None
    st.session_state["dismissed_insights"] = set()
    st.session_state["chart_cache"] = {}
    st.session_state["negative_flag_decision"] = {}
    st.session_state["_last_file_hash"] = file_hash

confirm_negatives = frozenset(st.session_state["negative_flag_decision"].get("confirmed", []))
df, preview_df, cleaning_logs, negative_flags = prepare(file_bytes, uploaded_file.name, confirm_negatives)

if df.empty:
    st.error("The dataset is empty after cleaning. Please upload a valid file.")
    st.stop()

outliers = detect_outliers(file_hash, df)
metadata = build_metadata(file_hash, df)
render_sidebar(df, cleaning_logs, outliers)

# Negative value confirmation
if negative_flags:
    unflagged = {col: cnt for col, cnt in negative_flags.items() if col not in confirm_negatives}
    if unflagged:
        st.warning("The following columns contain negative values. Select which to nullify:")
        to_nullify = []
        for col, cnt in unflagged.items():
            if st.checkbox(f"Nullify {cnt} negative value(s) in '{col}'", key=f"neg_{col}"):
                to_nullify.append(col)
        if st.button("Apply Negative Value Decisions"):
            st.session_state["negative_flag_decision"]["confirmed"] = list(confirm_negatives) + to_nullify
            st.rerun()

with st.expander("Data Preview"):
    st.dataframe(preview_df.head(20), use_container_width=True)

with st.expander("Cleaning Report"):
    for log in cleaning_logs:
        st.write(f"- {log}")

with st.expander("Statistical Metadata"):
    st.text(metadata)

if outliers:
    with st.expander("Outlier Report"):
        comp_df = pd.DataFrame([
            {
                "Column": col,
                "Outlier Count": v["count"],
                "Outlier %": v["pct"],
                "Expected Range": f"{v['lower_bound']} – {v['upper_bound']}",
                "Extreme Outliers (z>3)": v["extreme_count"],
            }
            for col, v in outliers.items()
        ])
        st.dataframe(comp_df.set_index("Column"), use_container_width=True)

st.divider()

# Chart override
st.subheader("Chart Override (optional)")
c1, c2, c3, c4 = st.columns(4)
chart_override = c1.selectbox("Chart type", ["auto"] + CHART_TYPES)
x_override = c2.selectbox("X axis", ["auto"] + list(df.columns))
y_override = c3.selectbox("Y axis", ["auto"] + list(df.columns))
color_override = c4.selectbox("Color / group by", ["none"] + list(df.columns))

# Query input
cols = df.columns
if len(cols) >= 2:
    default_query = f"Analyze the relationship between {cols[0]} and {cols[1]}"
elif len(cols) == 1:
    default_query = f"Describe the distribution of {cols[0]}"
else:
    default_query = "Summarize the dataset"

query = st.text_input("Ask a data question", value=default_query, max_chars=500)

if len(query.strip()) < 5:
    st.warning("Please enter a question of at least 5 characters.")
    st.stop()

col_btn1, col_btn2 = st.columns([1, 5])
run = col_btn1.button("Run Analysis", type="primary")
reset = col_btn2.button("Reset Conversation")

if reset:
    st.session_state["conversation"] = []
    st.session_state["last_result"] = None
    st.session_state["dismissed_insights"] = set()
    st.session_state["chart_cache"] = {}
    st.rerun()

if not run:
    if st.session_state["last_result"] is not None:
        st.info("Showing previous analysis results. Ask a follow-up question and click Run Analysis.")
    else:
        st.stop()

if run:
    with st.spinner("Analyzing..."):
        result, schema_warnings = run_ai_analysis(
            metadata,
            build_sample_csv(df),
            st.session_state["conversation"],
            query,
        )

    if "error" in result:
        st.error(result["error"])
        st.stop()

    if schema_warnings:
        for w in schema_warnings:
            st.warning(f"Schema warning: {w}")

    st.session_state["conversation"].append({"role": "user", "content": query})
    st.session_state["conversation"].append({
        "role": "assistant",
        "content": result.get("overall_summary", ""),
    })
    st.session_state["last_result"] = result

result = st.session_state["last_result"]
if result is None:
    st.stop()

insights = result.get("insights", [])
summary = result.get("overall_summary", "No summary provided.")
dismissed = st.session_state["dismissed_insights"]

st.subheader("Executive Summary")
st.markdown(
    f"<div style='background:#f0f2f6;padding:15px;border-radius:8px;'>{summary}</div>",
    unsafe_allow_html=True,
)

severity_fn = {"info": st.info, "warning": st.warning, "critical": st.error}
fuzzy_log: list[str] = []
all_insight_figs = []

if insights:
    st.subheader("Detailed Findings")
    active_insights = [ins for i, ins in enumerate(insights) if i not in dismissed]
    dismissed_count = len(insights) - len(active_insights)
    if dismissed_count:
        st.caption(f"{dismissed_count} insight(s) dismissed.")

    for i, ins in enumerate(active_insights):
        original_idx = insights.index(ins)
        ins_type = ins.get("type", "general").capitalize()
        severity = ins.get("severity", "info")
        text = ins.get("insight", "")
        label = f"{'[!] ' if severity == 'critical' else ''}{ins_type}: {text[:80]}..."

        with st.expander(label, expanded=(severity == "critical")):
            severity_fn.get(severity, st.info)(text)

            cols_ref = ins.get("related_columns", [])
            if cols_ref:
                st.caption(f"Related columns: {', '.join(cols_ref)}")

            # Per-insight chart controls
            cs = ins.get("chart_suggestion", {}) or {}
            ai_chart_type = cs.get("chart_type", "null")
            ai_x = fuzzy_col(cs.get("x_column"), df.columns, fuzzy_log)
            ai_y = fuzzy_col(cs.get("y_column"), df.columns, fuzzy_log)
            ai_color = fuzzy_col(cs.get("color_column"), df.columns, fuzzy_log)

            ic1, ic2, ic3, ic4 = st.columns(4)
            override_ctype = ic1.selectbox(
                "Chart", ["ai"] + CHART_TYPES,
                key=f"ct_{original_idx}",
                index=0,
            )
            override_x = ic2.selectbox(
                "X", ["ai"] + list(df.columns),
                key=f"cx_{original_idx}",
                index=0,
            )
            override_y = ic3.selectbox(
                "Y", ["ai"] + list(df.columns),
                key=f"cy_{original_idx}",
                index=0,
            )
            override_color = ic4.selectbox(
                "Color", ["ai", "none"] + list(df.columns),
                key=f"cc_{original_idx}",
                index=0,
            )

            use_ctype = override_ctype if override_ctype != "ai" else (ai_chart_type if ai_chart_type not in (None, "null") else None)
            use_x = fuzzy_col(override_x, df.columns) if override_x != "ai" else ai_x
            use_y = fuzzy_col(override_y, df.columns) if override_y != "ai" else ai_y
            use_color_raw = override_color if override_color != "ai" else (ai_color if ai_color else "none")
            use_color = None if use_color_raw == "none" else fuzzy_col(use_color_raw, df.columns)

            cache_key = f"{original_idx}_{use_ctype}_{use_x}_{use_y}_{use_color}"
            if cache_key not in st.session_state["chart_cache"]:
                st.session_state["chart_cache"][cache_key] = (
                    build_chart(df, use_ctype, use_x, use_y, use_color) if use_ctype else None
                )
            fig_i = st.session_state["chart_cache"][cache_key]

            if fig_i:
                st.plotly_chart(fig_i, use_container_width=True, key=f"chart_insight_{original_idx}_{cache_key}")
                all_insight_figs.append(fig_i)

            if st.button("Dismiss", key=f"dismiss_{original_idx}"):
                st.session_state["dismissed_insights"].add(original_idx)
                st.rerun()
else:
    st.info("No specific insights were generated. Try a more specific question or provide richer data.")

if fuzzy_log:
    with st.expander("Column Substitution Log"):
        for entry in fuzzy_log:
            st.caption(entry)

# Conversation history
if st.session_state["conversation"]:
    with st.expander("Conversation History"):
        for msg in st.session_state["conversation"]:
            role = msg["role"].capitalize()
            st.markdown(f"**{role}:** {msg['content']}")

# Global chart override
override_active = any(v != "auto" for v in (chart_override, x_override, y_override))
manual_fig = None

if override_active:
    st.subheader("Specified Chart")
    x = fuzzy_col(x_override, df.columns) if x_override != "auto" else None
    y = fuzzy_col(y_override, df.columns) if y_override != "auto" else None
    c_type = chart_override if chart_override != "auto" else "bar"
    color_val = None if color_override == "none" else fuzzy_col(color_override, df.columns)
    manual_fig = build_chart(df, c_type, x, y, color_val)
    if manual_fig:
        st.plotly_chart(manual_fig, use_container_width=True, key="chart_manual_override")
    else:
        st.warning("Could not generate the chart with the selected settings.")

score = quality_score(df, cleaning_logs)

report_figs = all_insight_figs[:3]
if not report_figs and manual_fig:
    report_figs = [manual_fig]

report_html = render_html_report(
    query=query,
    conversation=st.session_state["conversation"],
    insights=[ins for i, ins in enumerate(insights) if i not in dismissed],
    summary=summary,
    figs=report_figs,
    logs=cleaning_logs,
    score=score,
    outliers=outliers,
)

st.divider()
dl1, dl2 = st.columns(2)
dl1.download_button("Download Report (HTML)", data=report_html, file_name="report.html", mime="text/html")
dl2.download_button("Download Cleaned CSV", data=df.to_csv(index=False), file_name="cleaned_data.csv", mime="text/csv")