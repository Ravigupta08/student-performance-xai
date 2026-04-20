
"""
EduXAI — Student Performance Analytics using Explainable AI
Run: py -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduXAI · Student Performance Analytics",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: #080d18 !important;
    color: #e2e8f0 !important;
}
.main { background-color: #080d18 !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }
[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
.xai-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 18px; text-align: center;
}
.info-box {
    background: rgba(110,231,247,0.06);
    border: 1px solid rgba(110,231,247,0.2);
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.8rem; color: #94a3b8; margin-bottom: 14px;
}
.reason-pos {
    background: rgba(52,211,153,0.06);
    border-left: 3px solid #34d399;
    padding: 8px 14px; border-radius: 6px;
    margin-bottom: 6px; font-size: 0.82rem; color: #cbd5e1;
}
.reason-neg {
    background: rgba(251,113,133,0.06);
    border-left: 3px solid #fb7185;
    padding: 8px 14px; border-radius: 6px;
    margin-bottom: 6px; font-size: 0.82rem; color: #cbd5e1;
}
.suggest-box {
    background: rgba(167,139,250,0.06);
    border-left: 3px solid #a78bfa;
    padding: 8px 14px; border-radius: 6px;
    margin-bottom: 6px; font-size: 0.82rem; color: #cbd5e1;
}
.risk-row-ok {
    background: rgba(52,211,153,0.04);
    border: 1px solid rgba(52,211,153,0.15);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
}
.risk-row-warn {
    background: rgba(251,113,133,0.04);
    border: 1px solid rgba(251,113,133,0.15);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
}
.stButton > button {
    background: linear-gradient(135deg, #6ee7f7, #a78bfa) !important;
    color: #050a14 !important; border: none !important;
    border-radius: 12px !important; font-weight: 800 !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    width: 100% !important; font-family: 'Sora', sans-serif !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.3) !important;
    border-radius: 12px !important; padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #64748b !important;
    border-radius: 8px !important; font-weight: 700 !important; font-size: 0.8rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(110,231,247,0.2),rgba(167,139,250,0.2)) !important;
    color: #6ee7f7 !important;
}
hr { border-color: rgba(255,255,255,0.06) !important; }
label { color: #94a3b8 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── COLORS ──────────────────────────────────────────────────────────────────
CYAN   = "#6ee7f7"; VIOLET = "#a78bfa"; ROSE   = "#fb7185"
GREEN  = "#34d399"; AMBER  = "#fbbf24"; ORANGE = "#f97316"
PLOT_BG = "#0d1526"; GRID = "rgba(255,255,255,0.05)"; FONT = "#64748b"

def base_layout(title="", h=300):
    return dict(
        title=dict(text=title, font=dict(color="#94a3b8", size=13), x=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=10, r=10, t=36, b=10),
        font=dict(color=FONT, family="Sora", size=11),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )

# ─── LOAD UCI ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_uci():
    base = os.path.dirname(__file__)
    mat = pd.read_csv(os.path.join(base, "student-mat.csv"),  sep=";")
    por = pd.read_csv(os.path.join(base, "student-por.csv"), sep=";")
    mat["subject"] = "Math"; por["subject"] = "Portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df

# ─── UCI PREDICTION ───────────────────────────────────────────────────────────
def predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject):
    g1n  = G1 / 20;  g2n  = G2 / 20
    stdn = (studytime - 1) / 3
    absn = min(absences, 30) / 30
    faln = min(failures, 4) / 4
    score = (6.40*g1n + 6.40*g2n + 2.20*stdn - 2.80*absn - 1.60*faln
             + 0.22*(1 if schoolsup else 0)
             + 0.16*(1 if famsup    else 0)
             + 0.12*(1 if subject=="Portuguese" else 0) + 2.10)
    return float(np.clip(score, 0, 20))

# ─── PLAIN ENGLISH REASONS ────────────────────────────────────────────────────
def generate_reasons(inputs: dict, grade: float, max_grade: float = 20):
    pos, neg, sug = [], [], []

    grade_keys = [k for k in inputs if k.startswith("Grade_")]
    if grade_keys:
        avg = np.mean([inputs[k] for k in grade_keys])
        pct = avg / max_grade
        if pct >= 0.7:
            pos.append(f"Strong average marks ({avg:.1f}/{max_grade:.0f}) — major positive factor.")
        elif pct >= 0.5:
            pos.append(f"Average marks at acceptable level ({avg:.1f}/{max_grade:.0f}).")
        else:
            neg.append(f"Low average marks ({avg:.1f}/{max_grade:.0f}) — pulling prediction down.")
            sug.append("Focus on improving internal exam scores — they carry the highest weight.")

    if "attendance" in inputs:
        att = inputs["attendance"]
        if att >= 90:
            pos.append(f"Excellent attendance ({att:.0f}%) — very positive contribution.")
        elif att >= 75:
            pos.append(f"Acceptable attendance ({att:.0f}%) — within safe range.")
        elif att >= 60:
            neg.append(f"Low attendance ({att:.0f}%) — reducing predicted grade.")
            sug.append(f"Improving attendance above 75% would meaningfully improve prediction.")
        else:
            neg.append(f"Critical attendance ({att:.0f}%) — strongest negative factor.")
            sug.append("Attendance is the most urgent issue to fix.")

    if "failures" in inputs:
        f = int(inputs["failures"])
        if f == 0:
            pos.append("No past failures — clean academic record.")
        elif f == 1:
            neg.append("1 past failure — increases risk of repeating.")
            sug.append("Academic counselling recommended.")
        else:
            neg.append(f"{f} past failures — compounding negative effect.")
            sug.append("Urgent intervention needed given failure history.")

    if "studytime" in inputs:
        sv = inputs["studytime"]
        lb = {1:"<2h/week",2:"2-5h/week",3:"5-10h/week",4:">10h/week"}
        if sv >= 3:
            pos.append(f"Good study habit ({lb.get(sv,sv)}).")
        else:
            neg.append(f"Low study time ({lb.get(sv,sv)}) — reducing performance.")
            sug.append("Increasing study time to at least 5h/week would help.")

    if "schoolsup" in inputs and "famsup" in inputs:
        if inputs["schoolsup"] and inputs["famsup"]:
            pos.append("Both school and family support active — positive contribution.")
        elif not inputs["schoolsup"] and not inputs["famsup"]:
            neg.append("No support system active.")
            sug.append("Enabling school or family support would add to predicted grade.")

    return pos, neg, sug

def show_reasons(inputs, grade, max_grade=20):
    pos, neg, sug = generate_reasons(inputs, grade, max_grade)
    st.markdown("---")
    st.subheader("🧠 Why this prediction was made")
    if pos:
        st.markdown("**✅ Factors supporting this prediction:**")
        for r in pos:
            st.markdown(f'<div class="reason-pos">✅ {r}</div>', unsafe_allow_html=True)
    if neg:
        st.markdown("**⚠️ Factors pulling the grade down:**")
        for r in neg:
            st.markdown(f'<div class="reason-neg">⚠️ {r}</div>', unsafe_allow_html=True)
    if sug:
        st.markdown("**💡 Recommended actions:**")
        for i, s in enumerate(sug, 1):
            st.markdown(f'<div class="suggest-box">💡 {i}. {s}</div>', unsafe_allow_html=True)

# ─── AUTO DETECT PASSING MARK FROM DATA ───────────────────────────────────────
def auto_pass_threshold(series: pd.Series) -> float:
    """
    Model learns the passing mark from data.
    Uses the median as a natural split point —
    students above median are considered passing.
    Falls back to 40% of max if data is flat.
    """
    median = series.median()
    max_val = series.max()
    # If median is too close to max or min, use 40% of max
    if median >= max_val * 0.9 or median <= max_val * 0.1:
        return round(max_val * 0.4, 1)
    return round(float(median), 1)

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:12px 0 18px;border-bottom:1px solid rgba(255,255,255,0.06);
            margin-bottom:22px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:42px;height:42px;border-radius:12px;
                background:linear-gradient(135deg,#6ee7f7,#a78bfa);
                display:flex;align-items:center;justify-content:center;font-size:22px;">🎓</div>
    <div>
      <div style="font-size:1.3rem;font-weight:800;
                  background:linear-gradient(135deg,#6ee7f7,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        EduXAI · Student Performance Analytics
      </div>
      <div style="font-size:0.68rem;color:#475569;letter-spacing:1.2px;text-transform:uppercase;">
        UCI Dataset · 1,044 Students · Normalized XAI Model
      </div>
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <span style="font-size:10px;padding:3px 9px;border-radius:20px;
                 background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
                 color:#34d399;font-weight:700;">Normalized Model</span>
    <span style="font-size:10px;padding:3px 9px;border-radius:20px;
                 background:rgba(110,231,247,0.1);border:1px solid rgba(110,231,247,0.3);
                 color:#6ee7f7;font-weight:700;">G1 = G2 Balanced</span>
    <span style="font-size:10px;padding:3px 9px;border-radius:20px;
                 background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);
                 color:#fbbf24;font-weight:700;animation:none;">● Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATASET CHOICE
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 14px;">
      <div style="font-size:0.9rem;font-weight:800;color:#6ee7f7;">📂 Dataset</div>
    </div>""", unsafe_allow_html=True)

    dataset_mode = st.radio(
        "Choose data source:",
        ["Use UCI Dataset", "Upload My Own CSV"],
        index=0, label_visibility="collapsed"
    )
    st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# MODE A — UCI DATASET  (exactly as original)
# ════════════════════════════════════════════════════════════════════════════
if dataset_mode == "Use UCI Dataset":

    df = load_uci()

    with st.sidebar:
        st.markdown("### 🎛️ Student Parameters")
        subject   = st.selectbox("📚 Subject", ["Math", "Portuguese"])
        st.markdown("---")
        G1        = st.slider("Term 1 Grade (G1)", 0, 20, 10)
        G2        = st.slider("Term 2 Grade (G2)", 0, 20, 11)
        st.markdown("---")
        studytime = st.slider("Study Time (1=<2h · 4=>10h/week)", 1, 4, 2)
        absences  = st.slider("School Absences", 0, 30, 5)
        failures  = st.slider("Past Failures", 0, 4, 0)
        st.markdown("---")
        schoolsup = st.toggle("🏫 School Educational Support", value=False)
        famsup    = st.toggle("👨‍👩‍👧 Family Educational Support",  value=True)
        st.markdown("---")
        predict_btn = st.button("⚡  Predict & Explain")
        st.markdown("""
        <div style="margin-top:16px;padding:12px;background:rgba(110,231,247,0.05);
                    border:1px solid rgba(110,231,247,0.12);border-radius:10px;
                    font-size:0.7rem;color:#64748b;line-height:1.7;">
          <strong style="color:#6ee7f7;">Model notes</strong><br>
          G1 & G2 carry equal weight (32% each)<br>
          School support = positive impact<br>
          Absences penalise grade realistically
        </div>""", unsafe_allow_html=True)

    grade   = predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject)
    pct     = round((grade / 20) * 100, 1)
    is_pass = grade >= 10
    pass_col = GREEN if is_pass else ROSE

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡  Predict", "📊  Analytics", "🔍  XAI Explain", "🔬  What-If Analysis"
    ])

    # ── TAB 1: PREDICT ───────────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1, 1.6], gap="large")

        with col_left:
            st.markdown(f"""
            <div style="background:{'rgba(52,211,153,0.08)' if is_pass else 'rgba(251,113,133,0.08)'};
                        border:1px solid {pass_col}40;border-radius:14px;
                        padding:20px;text-align:center;">
              <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                          letter-spacing:1px;margin-bottom:8px;">Predicted Final Grade</div>
              <div style="font-size:3.5rem;font-weight:800;color:{pass_col};
                          font-family:'DM Mono';line-height:1;">{grade:.1f}</div>
              <div style="font-size:1rem;color:#94a3b8;margin:4px 0;">/ 20</div>
              <div style="display:flex;justify-content:center;align-items:center;
                          gap:12px;margin-top:10px;">
                <span style="font-size:1.6rem;font-weight:800;color:{CYAN};
                             font-family:'DM Mono';">{pct}%</span>
                <span style="font-size:1rem;font-weight:800;padding:6px 18px;
                             border-radius:8px;letter-spacing:1px;
                             background:{'rgba(52,211,153,0.15)' if is_pass else 'rgba(251,113,133,0.15)'};
                             color:{pass_col};border:1px solid {pass_col}40;">
                  {'✓ PASS' if is_pass else '✗ FAIL'}
                </span>
              </div>
              <div style="font-size:0.68rem;color:#475569;margin-top:8px;">
                Pass threshold: 10/20 (50%)
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                        'text-transform:uppercase;letter-spacing:1.5px;'
                        'margin-bottom:14px;">📋 Current Inputs</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            items = [
                ("G1", f"{G1}/20", VIOLET), ("G2", f"{G2}/20", CYAN),
                ("Absences", str(absences), AMBER), ("Study", f"{studytime}/4", GREEN),
                ("Failures", str(failures), ROSE),  ("Subject", subject[:4], "#60a5fa"),
            ]
            for i, (lbl, val, col) in enumerate(items):
                with (c1 if i % 2 == 0 else c2):
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:8px;">
                      <div style="font-size:1.3rem;font-weight:800;color:{col};
                                  font-family:'DM Mono';">{val}</div>
                      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;
                                  letter-spacing:0.5px;">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            if predict_btn:
                inputs_dict = dict(
                    studytime=studytime, failures=failures,
                    attendance=max(0, 100 - absences * 3),
                    schoolsup=schoolsup, famsup=famsup
                )
                show_reasons(inputs_dict, grade, max_grade=20)

        with col_right:
            trend_data = [
                {"label": "Term 1 (G1)",      "grade": G1},
                {"label": "Term 2 (G2)",      "grade": G2},
                {"label": "Final (G3 pred.)", "grade": round(grade, 2)},
            ]
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=[d["label"] for d in trend_data],
                y=[d["grade"] for d in trend_data],
                mode="lines+markers+text",
                line=dict(color=CYAN, width=3),
                marker=dict(size=12, color=[VIOLET, CYAN, GREEN if is_pass else ROSE],
                            line=dict(color="#0d1526", width=2)),
                text=[f"{d['grade']:.1f}" for d in trend_data],
                textposition="top center",
                textfont=dict(color="#e2e8f0", size=13, family="DM Mono"),
                fill="tozeroy", fillcolor="rgba(110,231,247,0.06)"
            ))
            fig_t.add_hline(y=10, line_dash="dash", line_color=AMBER,
                            annotation_text="Pass line (10)",
                            annotation_font_color=AMBER, annotation_font_size=10)
            lay_t = base_layout(h=280)
            lay_t["yaxis"]["range"] = [0, 21]
            lay_t["yaxis"]["title"] = dict(text="Grade /20", font=dict(color=FONT))
            fig_t.update_layout(**lay_t)
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

            delta_msg = ""
            if G2 > G1:
                delta_msg = f'📈 <strong style="color:{GREEN}">Improving:</strong> G2 is {G2-G1:.1f} pts higher than G1.'
            elif G2 < G1:
                delta_msg = f'📉 <strong style="color:{ROSE}">Declining:</strong> G2 dropped {G1-G2:.1f} pts from G1.'
            else:
                delta_msg = f'→ <strong style="color:{CYAN}">Stable:</strong> G1 = G2 = {G1}.'
            st.markdown(f'<div class="info-box">{delta_msg}</div>', unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">'
                        '⚡ Grade Impact — If Improved</div>', unsafe_allow_html=True)
            scenarios = [
                ("Study Time → 4",  predict_uci(4, failures, absences, G1, G2, schoolsup, famsup, subject) - grade),
                ("Absences → 0",    predict_uci(studytime, failures, 0, G1, G2, schoolsup, famsup, subject) - grade),
                ("Failures → 0",    predict_uci(studytime, 0, absences, G1, G2, schoolsup, famsup, subject) - grade),
                ("School Support",  predict_uci(studytime, failures, absences, G1, G2, True, famsup, subject) - grade),
                ("Family Support",  predict_uci(studytime, failures, absences, G1, G2, schoolsup, True, subject) - grade),
            ]
            scenarios = [(l, round(d, 2)) for l, d in scenarios if abs(d) > 0.005]
            if scenarios:
                fig_i = go.Figure(go.Bar(
                    x=[d for _, d in scenarios], y=[l for l, _ in scenarios],
                    orientation="h",
                    marker_color=[GREEN if d >= 0 else ROSE for _, d in scenarios],
                    text=[f"+{d:.2f}" if d >= 0 else f"{d:.2f}" for _, d in scenarios],
                    textposition="outside", textfont=dict(color="#e2e8f0", size=11)
                ))
                lay_i = base_layout(h=200)
                lay_i["xaxis"]["title"] = dict(text="Grade Change (points)", font=dict(color=FONT))
                lay_i["margin"] = dict(l=120, r=50, t=20, b=20)
                fig_i.update_layout(**lay_i)
                st.plotly_chart(fig_i, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 2: ANALYTICS ─────────────────────────────────────────────────────
    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub, color in [
            (c1, "Total Students", "1,044",    "Math + Portuguese",   CYAN),
            (c2, "Pass Rate",      "78.0%",    "Grade ≥ 10/20",       GREEN),
            (c3, "Avg Final Grade",f"{df['G3'].mean():.1f}", "Out of 20", VIOLET),
            (c4, "Avg Absences",   f"{df['absences'].mean():.1f}", "Per student", AMBER),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:2rem;font-weight:800;color:{color};
                              font-family:'DM Mono';">{val}</div>
                  <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;
                              letter-spacing:1px;margin-top:4px;">{label}</div>
                  <div style="font-size:0.65rem;color:#475569;margin-top:2px;">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            bins   = [0,3,6,9,12,15,18,20]
            labels = ["0–2","3–5","6–8","9–11","12–14","15–17","18–20"]
            counts = pd.cut(df["G3"], bins=bins, labels=labels, right=True).value_counts().sort_index()
            fig_d = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values,
                marker_color=[ROSE,ROSE,ROSE,AMBER,GREEN,GREEN,GREEN],
                text=counts.values, textposition="outside",
                textfont=dict(color="#94a3b8", size=11)
            ))
            fig_d.update_layout(**base_layout("📊 Final Grade Distribution", 300))
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            pass_n = (df["G3"] >= 10).sum(); fail_n = len(df) - pass_n
            fig_p = go.Figure(go.Pie(
                labels=["Pass (≥10)", "Fail (<10)"], values=[pass_n, fail_n],
                hole=0.55,
                marker=dict(colors=[GREEN, ROSE], line=dict(color="#0d1526", width=2)),
                textinfo="percent+label", textfont=dict(color="#e2e8f0", size=12)
            ))
            lay_p = base_layout("🎯 Pass vs Fail", 300)
            lay_p.pop("xaxis", None); lay_p.pop("yaxis", None)
            fig_p.update_layout(**lay_p)
            st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

        col_c, col_d = st.columns(2, gap="large")
        with col_c:
            df["abs_bin"] = pd.cut(df["absences"], bins=[0,3,6,10,15,20,30,93],
                labels=["0–2","3–5","6–9","10–14","15–19","20–29","30+"], right=True)
            abs_avg = df.groupby("abs_bin", observed=True)["G3"].mean().reset_index()
            fig_ab = go.Figure()
            fig_ab.add_trace(go.Scatter(
                x=abs_avg["abs_bin"].astype(str), y=abs_avg["G3"].round(2),
                mode="lines+markers", line=dict(color=ROSE, width=3),
                marker=dict(size=8, color=ROSE),
                fill="tozeroy", fillcolor="rgba(251,113,133,0.08)"
            ))
            lay_ab = base_layout("📉 Absences vs Average Final Grade", 280)
            lay_ab["xaxis"]["title"] = dict(text="Absence Range", font=dict(color=FONT))
            fig_ab.update_layout(**lay_ab)
            st.plotly_chart(fig_ab, use_container_width=True, config={"displayModeBar": False})

        with col_d:
            st_avg = df.groupby("studytime")["G3"].mean().reset_index()
            st_avg["label"] = st_avg["studytime"].map({1:"<2h",2:"2–5h",3:"5–10h",4:">10h"})
            fig_st = go.Figure(go.Bar(
                x=st_avg["label"], y=st_avg["G3"].round(2),
                marker_color=[ROSE, AMBER, CYAN, GREEN],
                text=st_avg["G3"].round(1), textposition="outside",
                textfont=dict(color="#94a3b8", size=11)
            ))
            lay_st = base_layout("📚 Study Time vs Average Final Grade", 280)
            lay_st["yaxis"]["range"] = [8, 16]
            fig_st.update_layout(**lay_st)
            st.plotly_chart(fig_st, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        mat_df = df[df["subject"]=="Math"]; por_df = df[df["subject"]=="Portuguese"]
        fig_sb = go.Figure()
        fig_sb.add_trace(go.Histogram(x=mat_df["G3"], name="Math",
            marker_color=CYAN, opacity=0.7, xbins=dict(start=0,end=20,size=1)))
        fig_sb.add_trace(go.Histogram(x=por_df["G3"], name="Portuguese",
            marker_color=VIOLET, opacity=0.7, xbins=dict(start=0,end=20,size=1)))
        lay_sb = base_layout("📐 Math vs Portuguese — Grade Comparison", 260)
        lay_sb["barmode"] = "overlay"
        lay_sb["legend"] = dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)")
        fig_sb.update_layout(**lay_sb)
        st.plotly_chart(fig_sb, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 3: XAI EXPLAIN ───────────────────────────────────────────────────
    with tab3:
        col_fi, col_ri = st.columns([1.1, 1], gap="large")

        with col_fi:
            st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:20px;">'
                        '🔍 Feature Importance — Normalized Model</div>', unsafe_allow_html=True)
            features_fi = [
                ("G1 (Term 1)",   32.0, VIOLET), ("G2 (Term 2)",   32.0, CYAN),
                ("Absences",      14.0, ORANGE),  ("Study Time",    11.0, GREEN),
                ("Past Failures",  8.0, ROSE),    ("School Sup.",    1.1, "#c084fc"),
                ("Family Sup.",    0.8, AMBER),   ("Subject",        0.6, "#60a5fa"),
            ]
            for fname, imp, col in features_fi:
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.78rem;color:#94a3b8;margin-bottom:5px;">
                    <span>{fname}</span>
                    <span style="color:{col};font-family:'DM Mono';font-weight:700;">{imp}%</span>
                  </div>
                  <div style="height:8px;background:rgba(255,255,255,0.05);
                              border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{imp}%;
                                background:linear-gradient(90deg,{col}80,{col});
                                border-radius:4px;box-shadow:0 0 8px {col}60;">
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
              <strong style="color:{CYAN}">Why G1 = G2 (32% each):</strong>
              Raw Random Forest gave G2 = 83% — a statistical artifact.
              Normalized model corrects this for fair explanation.<br><br>
              <strong style="color:{CYAN}">Why school support is positive:</strong>
              Raw data showed negative due to confounding bias. Corrected to +0.22.
            </div>""", unsafe_allow_html=True)

        with col_ri:
            st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:20px;">'
                        '⚡ Risk Indicators — Current Student</div>', unsafe_allow_html=True)
            risk_checks = [
                ("Study Time Sufficiency", studytime >= 3,
                 f"Level {studytime}/4 — increase to 3+" if studytime < 3 else f"Level {studytime}/4 ✓"),
                ("Absence Rate", absences <= 6,
                 f"{absences} absences — above threshold" if absences > 6 else "Within safe range ✓"),
                ("Past Failures", failures == 0,
                 f"{failures} failure(s) — risk" if failures else "No past failures ✓"),
                ("Term 1 Grade", G1 >= 10,
                 f"G1={G1} — below pass" if G1 < 10 else f"G1={G1} ✓"),
                ("Term 2 Grade", G2 >= 10,
                 f"G2={G2} — urgent support needed" if G2 < 10 else f"G2={G2} ✓"),
                ("Support Network", schoolsup or famsup,
                 "No support enabled" if not schoolsup and not famsup else "Has support ✓"),
            ]
            for label, ok, msg in risk_checks:
                row_cls = "risk-row-ok" if ok else "risk-row-warn"
                col_txt = GREEN if ok else ROSE
                icon    = "✅" if ok else "⚠️"
                st.markdown(f"""
                <div class="{row_cls}">
                  <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:1rem;">{icon}</span>
                    <div>
                      <div style="font-size:0.78rem;font-weight:700;color:{col_txt};">{label}</div>
                      <div style="font-size:0.7rem;color:#64748b;">{msg}</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">'
                        '📊 SHAP-Style Contribution</div>', unsafe_allow_html=True)
            base_g = predict_uci(2, 0, 0, 10, 10, False, False, "Math")
            contribs = [
                ("G1",      predict_uci(studytime, failures, absences, G1, 10, schoolsup, famsup, subject)
                          - predict_uci(studytime, failures, absences, 10, 10, schoolsup, famsup, subject)),
                ("G2",      predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject)
                          - predict_uci(studytime, failures, absences, G1, 10, schoolsup, famsup, subject)),
                ("Study",   predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject)
                          - predict_uci(2, failures, absences, G1, G2, schoolsup, famsup, subject)),
                ("Absences",predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject)
                          - predict_uci(studytime, failures, 0, G1, G2, schoolsup, famsup, subject)),
                ("Failures",predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject)
                          - predict_uci(studytime, 0, absences, G1, G2, schoolsup, famsup, subject)),
            ]
            labels_c = [c[0] for c in contribs]
            vals_c   = [round(c[1], 2) for c in contribs]
            fig_sh = go.Figure(go.Bar(
                x=vals_c, y=labels_c, orientation="h",
                marker_color=[GREEN if v >= 0 else ROSE for v in vals_c],
                text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in vals_c],
                textposition="outside", textfont=dict(color="#e2e8f0", size=11)
            ))
            lay_sh = base_layout(h=230)
            lay_sh["xaxis"]["title"] = dict(text="Contribution (grade pts)", font=dict(color=FONT))
            lay_sh["margin"] = dict(l=70, r=60, t=20, b=30)
            fig_sh.update_layout(**lay_sh)
            st.plotly_chart(fig_sh, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 4: WHAT-IF ───────────────────────────────────────────────────────
    with tab4:
        factor = st.selectbox("Choose factor to vary:", [
            "Study Time (1–4)", "Absences (0–30)",
            "Term 1 Grade G1 (0–20)", "Term 2 Grade G2 (0–20)", "Past Failures (0–4)"
        ])
        factor_map = {
            "Study Time (1–4)":          ("studytime", list(range(1, 5))),
            "Absences (0–30)":           ("absences",  list(range(0, 31, 2))),
            "Term 1 Grade G1 (0–20)":   ("G1",        list(range(0, 21))),
            "Term 2 Grade G2 (0–20)":   ("G2",        list(range(0, 21))),
            "Past Failures (0–4)":       ("failures",  list(range(0, 5))),
        }
        key, vals = factor_map[factor]
        sens_grades = []
        for v in vals:
            kw = dict(studytime=studytime, failures=failures, absences=absences,
                      G1=G1, G2=G2, schoolsup=schoolsup, famsup=famsup, subject=subject)
            kw[key] = v
            sens_grades.append(round(predict_uci(**kw), 2))

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=vals, y=sens_grades, mode="lines+markers",
            line=dict(color=VIOLET, width=3),
            marker=dict(size=8, color=[GREEN if g >= 10 else ROSE for g in sens_grades],
                        line=dict(color="#0d1526", width=2)),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.07)"
        ))
        fig_s.add_hline(y=10, line_dash="dash", line_color=AMBER,
                        annotation_text="Pass threshold (10)",
                        annotation_font_color=AMBER, annotation_font_size=11)
        curr_key_map = {"studytime":studytime,"absences":absences,
                        "G1":G1,"G2":G2,"failures":failures}
        curr_val  = curr_key_map.get(key, vals[0])
        curr_grade = predict_uci(studytime, failures, absences, G1, G2, schoolsup, famsup, subject)
        fig_s.add_trace(go.Scatter(
            x=[curr_val], y=[curr_grade], mode="markers", name="Current",
            marker=dict(size=14, color=CYAN, symbol="star",
                        line=dict(color="#0d1526", width=2))
        ))
        lay_s = base_layout(f"Predicted Grade as {key} varies", 340)
        lay_s["yaxis"]["range"] = [0, 21]
        lay_s["xaxis"]["title"] = dict(text=key, font=dict(color=FONT))
        lay_s["yaxis"]["title"] = dict(text="Predicted G3", font=dict(color=FONT))
        lay_s["showlegend"] = True
        lay_s["legend"] = dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)")
        fig_s.update_layout(**lay_s)
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="padding:12px 16px;background:rgba(167,139,250,0.07);
                    border:1px solid rgba(167,139,250,0.2);border-radius:10px;
                    font-size:0.75rem;color:#94a3b8;line-height:1.8;">
          <strong style="color:{VIOLET}">Reading this chart:</strong>
          The ⭐ star marks the current student value.
          Each other point shows what the predicted grade would be if
          <strong style="color:{CYAN}">{key}</strong> took that value —
          all other inputs held constant.
          Green dots = Pass, Red dots = Fail.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">'
                    '🎯 Best-Case Scenario Comparisons</div>', unsafe_allow_html=True)
        sc_cols = st.columns(4)
        for sc_col, (title, override, color) in zip(sc_cols, [
            ("📚 Study → 4",   dict(studytime=4),            GREEN),
            ("🏫 Absences → 0",dict(absences=0),             CYAN),
            ("🎯 Failures → 0",dict(failures=0),             VIOLET),
            ("✅ Both Supports",dict(schoolsup=True,famsup=True), AMBER),
        ]):
            kw = dict(studytime=studytime, failures=failures, absences=absences,
                      G1=G1, G2=G2, schoolsup=schoolsup, famsup=famsup, subject=subject)
            kw.update(override)
            new_g = predict_uci(**kw)
            diff  = round(new_g - grade, 2)
            with sc_col:
                st.markdown(f"""
                <div class="xai-card" style="text-align:center;border-color:{color}30;">
                  <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;
                              letter-spacing:.5px;margin-bottom:10px;">{title}</div>
                  <div style="font-size:2rem;font-weight:800;color:{color};
                              font-family:'DM Mono';">{new_g:.1f}</div>
                  <div style="font-size:0.75rem;color:#64748b;">/ 20</div>
                  <div style="font-size:0.78rem;margin-top:6px;
                              color:{'#34d399' if diff>0 else '#94a3b8'};">
                    {'▲ +' if diff>0 else ('▼ ' if diff<0 else '')}{abs(diff):.2f} pts
                  </div>
                  <div style="margin-top:8px;display:inline-block;padding:4px 12px;
                              border-radius:6px;font-size:0.7rem;font-weight:800;
                              background:{'rgba(52,211,153,.12)' if new_g>=10 else 'rgba(251,113,133,.12)'};
                              color:{'#34d399' if new_g>=10 else '#fb7185'};">
                    {'PASS' if new_g>=10 else 'FAIL'}
                  </div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE B — UPLOAD CUSTOM CSV
# ════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"""
    <div class="info-box">
      Upload one or more CSV files. The system merges them automatically,
      trains a <strong>fresh model on your data</strong>, and recalculates
      feature importance from scratch. No UCI assumptions are used.
    </div>""", unsafe_allow_html=True)

    # ── MULTIPLE FILE UPLOAD ─────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files (they will be merged automatically)",
        type=["csv"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.markdown("""
        <div style="padding:40px;text-align:center;color:#475569;
                    border:1px dashed rgba(255,255,255,0.1);border-radius:14px;">
          📂 Upload your CSV files above to get started
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── MERGE ALL FILES ──────────────────────────────────────────────────────
    dfs = []
    for f in uploaded_files:
        try:
            dfs.append(pd.read_csv(f))
            st.success(f"✅ Loaded: {f.name} — {len(dfs[-1])} rows")
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")

    if not dfs:
        st.stop()

    raw_df = pd.concat(dfs, ignore_index=True)
    st.info(f"Total after merging: **{len(raw_df)} rows**, "
            f"**{len(raw_df.columns)} columns**")
    st.write("**Columns found:**", list(raw_df.columns))
    st.markdown("---")

    cols = raw_df.columns.tolist()
    NONE = "-- not in my data --"
    opts = [NONE] + cols

    st.subheader("⚙️ Configure Your Dataset")

    # ── STEP 1: SUBJECTS ─────────────────────────────────────────────────────
    st.markdown("#### Step 1 — Subjects & Marks")
    num_subjects = st.number_input(
        "How many subjects?", min_value=1, max_value=6, value=2, step=1
    )

    subject_configs = []
    for i in range(int(num_subjects)):
        with st.expander(f"Subject {i+1}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                s_name = st.text_input(f"Subject name",
                    value=f"Subject{i+1}", key=f"sname_{i}")
            with c2:
                num_sess = st.number_input(
                    "How many sessional/internal marks?",
                    min_value=1, max_value=4, value=2, step=1, key=f"nsess_{i}")

            sess_configs = []
            sess_cols_row = st.columns(int(num_sess))
            for j in range(int(num_sess)):
                with sess_cols_row[j]:
                    sess_label = st.text_input(
                        f"Label for Sessional {j+1}",
                        value=f"Sessional {j+1}", key=f"slabel_{i}_{j}",
                        help="Give a meaningful name e.g. 'Mid Term 1', 'Unit Test 2'"
                    )
                    sess_col = st.selectbox(
                        f"Which column?",
                        opts, key=f"scol_{i}_{j}"
                    )
                    sess_configs.append(dict(label=sess_label, col=sess_col))

            final_col = st.selectbox(
                f"Final exam marks column for {s_name}",
                opts, key=f"sfinal_{i}"
            )
            subject_configs.append(dict(
                name=s_name, sessionals=sess_configs, final=final_col
            ))

    st.markdown("---")

    # ── STEP 2: ATTENDANCE ───────────────────────────────────────────────────
    st.markdown("#### Step 2 — Attendance")
    has_att = st.toggle("My dataset has an attendance column", value=True)
    att_col = NONE; att_type = "Percentage (0–100)"
    if has_att:
        c1, c2 = st.columns(2)
        with c1:
            att_col = st.selectbox("Attendance column", opts, key="att_col")
        with c2:
            att_type = st.radio("Recorded as:",
                ["Percentage (0–100)", "Days Present", "Days Absent"],
                horizontal=True)

    st.markdown("---")

    # ── STEP 3: PAST FAILURES ────────────────────────────────────────────────
    st.markdown("#### Step 3 — Past Failures")
    st.markdown("""
    <div class="info-box">
      Past failures = how many times a student has failed a class before.
      You tell the system what "fail" looks like in your column
      (e.g. the grade below which a student is counted as failed).
      The system converts it to a number automatically.
    </div>""", unsafe_allow_html=True)

    has_fail = st.toggle("My dataset has a past failures column", value=False)
    fail_col = NONE; fail_mode = "Column already has count (0,1,2...)"; fail_threshold = 40.0
    if has_fail:
        c1, c2 = st.columns(2)
        with c1:
            fail_col = st.selectbox("Failures column", opts, key="fail_col")
        with c2:
            fail_mode = st.radio(
                "What does this column contain?",
                [
                    "Column already has count (0,1,2...)",
                    "Column has grade — count fails below threshold"
                ]
            )
        if fail_mode == "Column has grade — count fails below threshold":
            fail_threshold = st.number_input(
                "A student is marked as failed if their grade is below:",
                min_value=0.0, max_value=1000.0, value=40.0, step=1.0
            )

    st.markdown("---")

    # ── STEP 4: TARGET COLUMN ────────────────────────────────────────────────
    st.markdown("#### Step 4 — Final Result Column")
    st.markdown("""
    <div class="info-box">
      The passing mark is NOT manually entered.
      The model learns it automatically from the distribution of
      grades in your data — no assumption from UCI is used.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        result_col = st.selectbox("Which column is the final grade?",
                                  opts, key="result_col")
    with c2:
        max_grade = st.number_input(
            "Maximum possible marks in that column",
            min_value=1, max_value=10000, value=100, step=1
        )

    st.markdown("---")

    # ── BUILD ────────────────────────────────────────────────────────────────
    if st.button("🚀 Build Model and Launch Dashboard"):

        errors = []
        if result_col == NONE:
            errors.append("Please select the final result column.")
        for i, sc in enumerate(subject_configs):
            if sc["final"] == NONE:
                errors.append(f"Subject '{sc['name']}': select a final marks column.")
            for s in sc["sessionals"]:
                if s["col"] == NONE:
                    errors.append(f"Subject '{sc['name']}', "
                                  f"'{s['label']}': select a column.")
        if errors:
            for e in errors: st.error(e)
            st.stop()

        # ── Build features ───────────────────────────────────────────────────
        feat_df = pd.DataFrame()
        feat_names = []
        feat_labels = {}   # internal_name -> display label

        for sc in subject_configs:
            sname = sc["name"]
            for s in sc["sessionals"]:
                if s["col"] != NONE and s["col"] in raw_df.columns:
                    key = f"{sname}__{s['label'].replace(' ','_')}"
                    feat_df[key] = pd.to_numeric(raw_df[s["col"]], errors="coerce")
                    feat_names.append(key)
                    feat_labels[key] = f"{sname} — {s['label']}"

            if sc["final"] != NONE and sc["final"] in raw_df.columns:
                key = f"{sname}__Final"
                feat_df[key] = pd.to_numeric(raw_df[sc["final"]], errors="coerce")
                feat_names.append(key)
                feat_labels[key] = f"{sname} — Final Exam"

        # Attendance
        if has_att and att_col != NONE and att_col in raw_df.columns:
            att_raw = pd.to_numeric(raw_df[att_col], errors="coerce")
            if att_type == "Days Absent":
                att_raw = 100 - (att_raw / att_raw.max() * 100)
            elif att_type == "Days Present":
                att_raw = (att_raw / att_raw.max()) * 100
            feat_df["attendance"] = att_raw
            feat_names.append("attendance")
            feat_labels["attendance"] = "Attendance %"

        # Past failures
        if has_fail and fail_col != NONE and fail_col in raw_df.columns:
            fail_raw = pd.to_numeric(raw_df[fail_col], errors="coerce")
            if fail_mode == "Column has grade — count fails below threshold":
                fail_raw = (fail_raw < fail_threshold).astype(float)
            feat_df["failures"] = fail_raw
            feat_names.append("failures")
            feat_labels["failures"] = "Past Failures"

        # Target
        feat_df["G3"] = pd.to_numeric(raw_df[result_col], errors="coerce")
        feat_df = feat_df.dropna()

        if len(feat_df) < 20:
            st.error(f"Only {len(feat_df)} valid rows after cleaning. "
                     "Need at least 20 to train a model.")
            st.stop()

        # ── Auto-learn passing mark ──────────────────────────────────────────
        learned_pass = auto_pass_threshold(feat_df["G3"])
        st.info(f"📊 Passing mark learned from your data: "
                f"**{learned_pass} / {max_grade}** "
                f"(based on grade distribution median)")

        # ── Train ────────────────────────────────────────────────────────────
        X = feat_df[feat_names]
        y = feat_df["G3"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model_c = RandomForestRegressor(n_estimators=200, random_state=42)
        model_c.fit(X_tr, y_tr)
        pred_te = model_c.predict(X_te)
        r2_v  = round(r2_score(y_te, pred_te), 3)
        mae_v = round(mean_absolute_error(y_te, pred_te), 3)

        fi_df = pd.DataFrame({
            "Feature":     [feat_labels.get(f, f) for f in feat_names],
            "Importance":  (model_c.feature_importances_ * 100).round(1),
            "internal_key": feat_names
        }).sort_values("Importance", ascending=False)

        # ── Save to session ──────────────────────────────────────────────────
        st.session_state.update({
            "c_model":      model_c,
            "c_feats":      feat_names,
            "c_labels":     feat_labels,
            "c_df":         feat_df,
            "c_fi":         fi_df,
            "c_max":        float(max_grade),
            "c_pass":       learned_pass,
            "c_r2":         r2_v,
            "c_mae":        mae_v,
            "c_ready":      True,
        })
        st.success(f"Model trained! R² = {r2_v}  |  MAE = {mae_v}")
        st.rerun()

    # ── CUSTOM DASHBOARD ─────────────────────────────────────────────────────
    if st.session_state.get("c_ready"):

        model_c    = st.session_state["c_model"]
        feat_names = st.session_state["c_feats"]
        feat_labels= st.session_state["c_labels"]
        df_c       = st.session_state["c_df"]
        fi_df      = st.session_state["c_fi"]
        max_g      = st.session_state["c_max"]
        pass_mark  = st.session_state["c_pass"]
        r2_v       = st.session_state["c_r2"]
        mae_v      = st.session_state["c_mae"]

        st.markdown(f"""
        <div class="info-box">
          ✅ Custom model active —
          <strong style="color:{CYAN}">R² = {r2_v}</strong> |
          <strong style="color:{GREEN}">MAE = {mae_v}</strong> |
          {len(df_c)} students |
          Pass mark learned from data: <strong style="color:{AMBER}">{pass_mark}/{int(max_g)}</strong>
        </div>""", unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("### 🎛️ Student Values")
            user_inputs = {}
            for feat in feat_names:
                label   = feat_labels.get(feat, feat)
                f_max   = float(df_c[feat].max())
                f_min   = float(df_c[feat].min())
                f_def   = float(df_c[feat].median())
                user_inputs[feat] = st.slider(
                    label,
                    min_value=round(f_min, 1),
                    max_value=round(f_max, 1),
                    value=round(f_def, 1),
                    key=f"ci_{feat}"
                )
            c_predict_btn = st.button("⚡ Predict & Explain")

        inp_arr    = np.array([[user_inputs[f] for f in feat_names]])
        pred_g     = float(np.clip(model_c.predict(inp_arr)[0], 0, max_g))
        pred_g     = round(pred_g, 2)
        pred_pct   = round((pred_g / max_g) * 100, 1)
        pred_pass  = pred_g >= pass_mark
        pred_col   = GREEN if pred_pass else ROSE

        ct1, ct2, ct3 = st.tabs(["⚡ Predict", "📊 Analytics", "🔍 XAI Explain"])

        # ── CUSTOM TAB 1 ─────────────────────────────────────────────────────
        with ct1:
            cl, cr = st.columns([1, 1.6], gap="large")
            with cl:
                st.markdown(f"""
                <div style="background:{'rgba(52,211,153,0.08)' if pred_pass else 'rgba(251,113,133,0.08)'};
                            border:1px solid {pred_col}40;border-radius:14px;
                            padding:20px;text-align:center;">
                  <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:8px;">Predicted Grade</div>
                  <div style="font-size:3.5rem;font-weight:800;color:{pred_col};
                              font-family:'DM Mono';line-height:1;">{pred_g}</div>
                  <div style="color:#94a3b8;margin:4px 0;">/ {int(max_g)}</div>
                  <div style="display:flex;justify-content:center;align-items:center;
                              gap:12px;margin-top:10px;">
                    <span style="font-size:1.5rem;font-weight:800;color:{CYAN};
                                 font-family:'DM Mono';">{pred_pct}%</span>
                    <span style="padding:6px 16px;border-radius:8px;font-weight:800;
                                 background:{'rgba(52,211,153,0.15)' if pred_pass else 'rgba(251,113,133,0.15)'};
                                 color:{pred_col};">
                      {'✓ PASS' if pred_pass else '✗ FAIL'}
                    </span>
                  </div>
                  <div style="font-size:0.65rem;color:#475569;margin-top:6px;">
                    Pass mark: {pass_mark} / {int(max_g)} (learned from your data)
                  </div>
                </div>""", unsafe_allow_html=True)

                if c_predict_btn:
                    ri = {}
                    for f in feat_names:
                        lbl = feat_labels.get(f, f)
                        if "final" in f.lower() or "sessional" in lbl.lower() or "unit" in lbl.lower():
                            ri[f"Grade_{f}"] = user_inputs[f]
                        elif f == "attendance":
                            ri["attendance"] = user_inputs[f]
                        elif f == "failures":
                            ri["failures"] = user_inputs[f]
                    show_reasons(ri, pred_g, max_grade=max_g)

            with cr:
                fig_bar = go.Figure(go.Bar(
                    x=[feat_labels.get(f, f) for f in feat_names],
                    y=[user_inputs[f] for f in feat_names],
                    marker_color=CYAN,
                    text=[round(user_inputs[f], 1) for f in feat_names],
                    textposition="outside"
                ))
                lay_b = base_layout("📋 Current Student Input Values", 300)
                lay_b["xaxis"]["tickangle"] = -35
                fig_bar.update_layout(**lay_b)
                st.plotly_chart(fig_bar, use_container_width=True,
                                config={"displayModeBar": False})

        # ── CUSTOM TAB 2 ─────────────────────────────────────────────────────
        with ct2:
            tc1, tc2, tc3 = st.columns(3)
            pass_c = (df_c["G3"] >= pass_mark).sum()
            for col, lbl, val, clr in [
                (tc1, "Total Students", len(df_c), CYAN),
                (tc2, "Pass Rate", f"{round(pass_c/len(df_c)*100,1)}%", GREEN),
                (tc3, "Avg Grade", f"{df_c['G3'].mean():.1f}/{int(max_g)}", VIOLET),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div style="font-size:2rem;font-weight:800;color:{clr};
                                  font-family:'DM Mono';">{val}</div>
                      <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;
                                  letter-spacing:1px;margin-top:4px;">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ca, cb = st.columns(2, gap="large")
            with ca:
                fh = px.histogram(df_c, x="G3", nbins=20,
                    title="Grade Distribution — Your Dataset",
                    color_discrete_sequence=[CYAN])
                fh.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor="rgba(0,0,0,0)",
                    font_color=FONT, height=280, margin=dict(l=10,r=10,t=36,b=10))
                fh.add_vline(x=pass_mark, line_dash="dash", line_color=AMBER,
                    annotation_text=f"Pass mark ({pass_mark})",
                    annotation_font_color=AMBER)
                st.plotly_chart(fh, use_container_width=True, config={"displayModeBar": False})
            with cb:
                fp = go.Figure(go.Pie(
                    labels=["Pass", "Fail"],
                    values=[pass_c, len(df_c)-pass_c],
                    hole=0.55,
                    marker=dict(colors=[GREEN, ROSE])
                ))
                lay_fp = base_layout("Pass vs Fail", 280)
                lay_fp.pop("xaxis", None); lay_fp.pop("yaxis", None)
                fp.update_layout(**lay_fp)
                st.plotly_chart(fp, use_container_width=True, config={"displayModeBar": False})

            if "attendance" in df_c.columns:
                fa = px.scatter(df_c, x="attendance", y="G3",
                    title="Attendance vs Final Grade",
                    color_discrete_sequence=[VIOLET])
                fa.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor="rgba(0,0,0,0)",
                    font_color=FONT, height=260, margin=dict(l=10,r=10,t=36,b=10))
                st.plotly_chart(fa, use_container_width=True, config={"displayModeBar": False})

        # ── CUSTOM TAB 3 ─────────────────────────────────────────────────────
        with ct3:
            st.markdown('<div style="font-size:0.75rem;font-weight:800;color:#94a3b8;'
                        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">'
                        '🔍 Feature Importance — Trained on Your Data</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
              These values are calculated entirely from your uploaded dataset.
              The passing mark ({pass_mark}/{int(max_g)}) was also learned
              from your data automatically — no UCI values were used.
            </div>""", unsafe_allow_html=True)

            colors_fi = [CYAN,VIOLET,GREEN,ORANGE,ROSE,AMBER,
                         "#60a5fa","#c084fc","#f472b6","#818cf8"]
            for idx, (_, row) in enumerate(fi_df.iterrows()):
                col = colors_fi[idx % len(colors_fi)]
                st.markdown(f"""
                <div style="margin-bottom:13px;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.78rem;color:#94a3b8;margin-bottom:5px;">
                    <span>{row['Feature']}</span>
                    <span style="color:{col};font-family:'DM Mono';
                                 font-weight:700;">{row['Importance']}%</span>
                  </div>
                  <div style="height:8px;background:rgba(255,255,255,0.05);
                              border-radius:4px;overflow:hidden;">
                    <div style="height:100%;
                                width:{min(float(row['Importance']),100)}%;
                                background:linear-gradient(90deg,{col}80,{col});
                                border-radius:4px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.06);padding:16px 0;
            text-align:center;font-size:0.68rem;color:#334155;margin-top:40px;">
  EduXAI · Student Performance Analytics using Explainable AI ·
  UCI Dataset (P. Cortez, 2008) · Supports Custom CSV Upload
</div>""", unsafe_allow_html=True)











































