"""
InsightPilot - Streamlit front end.

A staged flow rather than a dashboard:
  landing -> upload -> cleaning report -> definitions -> ask

The cleaning report and the definitions screen exist because both steps
change the numbers, and a tool that changes numbers silently cannot be
trusted with a business question.
"""
from __future__ import annotations
import os, sys, time, tempfile, json, html as _html
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
import streamlit.components.v1 as components

from tools import dataset
from tools.warehouse import (schema_context, run_sql, tables, load_semantics,
                             load_profile, ambiguous_terms, save_semantics,
                             connect, column_values, date_ranges)
from tools.ingest import ingest
from agent.graph import investigate
from agent.llm import USAGE, reset_usage, budget_status

st.set_page_config(page_title="InsightPilot · agentic analytics",
                   page_icon="◆", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- palette
# Chart hues are the validated categorical set. Page surfaces are glass over
# an animated field built from the SAME hues at low alpha, so the interface
# and the data read as one system. Charts get an opaque surface ("solid") -
# a plot on frosted glass is unreadable.
PALETTE = {
    "dark": {
        "surface": "rgba(24,24,30,0.62)", "plane": "#07070b",
        "raised": "rgba(40,40,52,0.60)", "solid": "#16161c",
        "ink": "#ffffff", "ink2": "#dcdae6", "muted": "#9b98a8",
        "grid": "#2c2c34", "axis": "#3a3a46",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "seq": "#3987e5", "good": "#0ca30c", "warn": "#fab219",
        "critical": "#d03b3b", "border": "rgba(255,255,255,0.11)",
        "borderHi": "rgba(255,255,255,0.30)",
    },
    "light": {
        "surface": "rgba(255,255,255,0.74)", "plane": "#eef1f7",
        "raised": "rgba(255,255,255,0.92)", "solid": "#fcfcfb",
        "ink": "#0b0b0b", "ink2": "#3d3c42", "muted": "#6e6c78",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "seq": "#2a78d6", "good": "#0ca30c", "warn": "#fab219",
        "critical": "#d03b3b", "border": "rgba(11,11,11,0.10)",
        "borderHi": "rgba(11,11,11,0.26)",
    },
}

TIME_HINTS = ("date", "month", "quarter", "year", "week", "day", "ts",
              "time", "period")


def p() -> dict:
    return PALETTE[st.session_state.get("mode", "dark")]


# ------------------------------------------------------------------ state
for k, v in [("mode", "dark"), ("stage", "landing"), ("question", ""),
             ("pending", []), ("clarifications", {}), ("result", None)]:
    st.session_state.setdefault(k, v)


def goto(stage: str):
    st.session_state.stage = stage
    st.rerun()


def reset_run():
    st.session_state.update(pending=[], clarifications={}, result=None)


@st.cache_data(show_spinner=False)
def sql_df(sql: str, _key: str) -> pd.DataFrame:
    return run_sql(sql)


# -------------------------------------------------------------------- css
CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
  #MainMenu, footer, header {visibility: hidden;}
  /* Streamlit bolts an anchor-link icon onto headings - kill it */
  [data-testid="stHeaderActionElements"] {display:none !important;}
  h1 > a, h2 > a, h3 > a, .stMarkdown a.headerlink {display:none !important;}

  .block-container {padding-top: 1.1rem; padding-bottom: 0; max-width: 1180px;}
  /* Inter for reading, Plus Jakarta Sans for display. Set on containers,
     never on *, so Streamlit's icon font is left alone. */
  .stApp, .stMarkdown, .stButton button, .stTextInput input,
  .stSelectbox, [data-testid="stExpander"] summary, .stDownloadButton button {
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "cv11", "ss01";
  }
  .hero h1, .sec h2, .pagehead h1, .card h3, .stat .v, .brand .name,
  .ftitle, .ex .q, .stMarkdown h3, .stMarkdown h4 {
    font-family: "Plus Jakarta Sans", "Inter", system-ui, sans-serif !important;
  }
  .stMarkdown p, .stMarkdown li { color:$ink2; font-size:15px; line-height:1.7; }
  .stMarkdown h3, .stMarkdown h4 { color:$ink !important; letter-spacing:-.02em; }
  .stMarkdown strong, .stMarkdown b { color:$ink; font-weight:650; }
  [data-testid="stCaptionContainer"], .stCaption { color:$muted !important; }

  /* ---------- buttons: one system, not Streamlit red + plain white ---- */
  .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
    border-radius:12px !important; font-weight:600 !important;
    font-size:14.5px !important; padding:10px 18px !important;
    border:1px solid $borderHi !important; background:$raised !important;
    color:$ink !important; backdrop-filter: blur(12px);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
  }
  .stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-2px); border-color:$s0 !important;
    box-shadow: 0 10px 28px -14px $s0 !important;
  }
  .stButton button[kind="primary"], [data-testid="stBaseButton-primary"],
  [data-testid="baseButton-primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(120deg,$s0,$s2) !important; color:#fff !important;
    border:0 !important; box-shadow: 0 12px 30px -14px $s0 !important;
  }
  .stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 16px 36px -12px $s0 !important; filter: brightness(1.08);
  }
  .stButton button p { font-weight:600 !important; color:inherit !important; }
  .stTextInput input {
    background:$raised !important; color:$ink !important; font-size:15.5px !important;
    border:1px solid $borderHi !important; border-radius:12px !important;
    padding:14px 16px !important;
  }
  .stTextInput input:focus { border-color:$s0 !important;
                             box-shadow:0 0 0 3px $ring !important; }
  /* the wrapper around inputs carries its own background - match it */
  [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
  [data-baseweb="textarea"] {
    background:$raised !important; border-color:$borderHi !important;
    border-radius:12px !important;
  }
  [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color:$ink !important; }
  input::placeholder, textarea::placeholder { color:$muted !important; }

  /* expanders: glass card, readable header, in either mode */
  [data-testid="stExpander"] details { background:transparent !important;
                                       border:0 !important; }
  [data-testid="stExpander"] summary {
    background:transparent !important; color:$ink !important;
    padding:14px 16px !important; border-radius:14px !important;
  }
  [data-testid="stExpander"] summary:hover { background:$raised !important; }
  [data-testid="stExpander"] summary p { color:$ink !important; font-weight:600 !important;
                                         font-size:14.5px !important; }
  [data-testid="stExpander"] svg { fill:$ink2 !important; color:$ink2 !important; }

  /* ---------- the answer panel ---------- */
  .st-key-answer {
    background:$surface; backdrop-filter: blur(16px) saturate(130%);
    border:1px solid $border; border-left:3px solid $s0; border-radius:18px;
    padding:22px 26px 10px !important; margin:6px 0 4px;
    box-shadow:0 22px 50px -34px rgba(0,0,0,.9);
  }
  .st-key-answer p { font-size:16px !important; line-height:1.75 !important;
                     color:$ink2 !important; }
  .st-key-answer p > strong:first-child {
    display:inline-block; margin-right:6px; font-size:11.5px; letter-spacing:.14em;
    text-transform:uppercase; font-weight:800; color:$s0;
  }
  .st-key-answer p:nth-of-type(2) > strong:first-child { color:$s2; }
  .st-key-answer p:nth-of-type(3) > strong:first-child { color:$s3; }
  .st-key-answer blockquote { border-left:3px solid $warn; color:$ink2;
                              background:$raised; border-radius:0 10px 10px 0;
                              padding:8px 14px; }

  /* ---------- animated colour field ---------- */
  .stApp { background: $plane; }
  .stApp::before {
    content:""; position: fixed; inset: -30%; z-index: 0; pointer-events: none;
    background:
      radial-gradient(38% 42% at 20% 18%, $g0 0%, transparent 64%),
      radial-gradient(34% 38% at 82% 12%, $g1 0%, transparent 62%),
      radial-gradient(40% 44% at 72% 82%, $g2 0%, transparent 64%),
      radial-gradient(32% 36% at 14% 86%, $g3 0%, transparent 62%),
      radial-gradient(30% 34% at 50% 50%, $g4 0%, transparent 66%);
    filter: blur(34px) saturate(135%);
    animation: drift 30s ease-in-out infinite alternate,
               hue 54s linear infinite;
  }
  @keyframes drift {
    0%   { transform: translate3d(0,0,0) scale(1) rotate(0deg); }
    33%  { transform: translate3d(3%,-2.5%,0) scale(1.09) rotate(2deg); }
    66%  { transform: translate3d(-2.5%,2%,0) scale(1.04) rotate(-1.5deg); }
    100% { transform: translate3d(1.5%,1.5%,0) scale(1.12) rotate(1deg); }
  }
  @keyframes hue {
    0%,100% { filter: blur(34px) saturate(135%) hue-rotate(0deg); }
    50%     { filter: blur(34px) saturate(150%) hue-rotate(14deg); }
  }
  .stApp::after {
    content:""; position: fixed; inset:0; z-index:0; pointer-events:none;
    background-image:
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='.5'/%3E%3C/svg%3E"),
      radial-gradient(120% 90% at 50% 0%, transparent 42%, $vig 100%);
    opacity: .5; mix-blend-mode: soft-light;
  }
  .block-container { position: relative; z-index: 1; }

  @keyframes rise {
    from { opacity:0; transform: translateY(18px); }
    to   { opacity:1; transform:none; }
  }
  .rise { animation: rise .8s cubic-bezier(.22,.7,.3,1) both; }
  .d1{animation-delay:.06s}.d2{animation-delay:.14s}.d3{animation-delay:.22s}
  .d4{animation-delay:.30s}.d5{animation-delay:.38s}.d6{animation-delay:.46s}

  @media (prefers-reduced-motion: reduce) {
    .stApp::before, .rise, .flow, .orbit, .ping, .brand .dot
      { animation: none !important; }
    .rise { opacity:1 !important; transform:none !important; }
  }

  /* ---------- brand bar ---------- */
  .topbar {
    position: sticky; top: 0; z-index: 50;
    display:flex; align-items:center; justify-content:space-between;
    gap:18px; padding:13px 20px; margin:0 0 6px;
    background:$surface; backdrop-filter: blur(20px) saturate(140%);
    border:1px solid $border; border-radius:16px;
    box-shadow: 0 10px 34px -22px rgba(0,0,0,.85);
  }
  .brand { display:flex; align-items:center; gap:12px; }
  .brand .mark {
    width:34px;height:34px;border-radius:10px;display:flex;
    align-items:center;justify-content:center;
    background: linear-gradient(135deg,$s0,$s2 55%,$s1);
    box-shadow: 0 6px 20px -8px $s0;
  }
  .orbit { transform-origin: 12px 12px; animation: spin 9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .brand .name {
    font-size:19px; font-weight:700; letter-spacing:-.028em; color:$ink;
    line-height:1;
  }
  .brand .name b {
    font-weight:700;
    background: linear-gradient(95deg,$s0,$s2);
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .brand .tagline {
    font-size:11px; color:$muted; letter-spacing:.09em; text-transform:uppercase;
    margin-top:4px; font-weight:600;
  }
  .barmeta { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .chip {
    font-size:11px; letter-spacing:.06em; text-transform:uppercase;
    color:$ink2; border:1px solid $border; border-radius:999px;
    padding:6px 12px; background:$raised; font-weight:600;
  }
  .chip.live { display:flex; align-items:center; gap:7px; color:$ink; }
  .ping {
    width:7px;height:7px;border-radius:50%;background:#0ca30c;
    box-shadow:0 0 0 0 rgba(12,163,12,.6); animation: ping 2.1s infinite;
  }
  @keyframes ping {
    70%  { box-shadow:0 0 0 8px rgba(12,163,12,0); }
    100% { box-shadow:0 0 0 0 rgba(12,163,12,0); }
  }
  @media (max-width:760px) { .barmeta { display:none; } }

  /* ---------- hero ---------- */
  .hero { padding: 46px 0 8px; }
  .eyebrow { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:26px; }
  .eyebrow span {
    font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
    color:$ink2; border:1px solid $border; border-radius:999px;
    padding:6px 14px; background:$surface; backdrop-filter: blur(12px);
    font-weight:600;
  }
  .hero h1 {
    font-size: clamp(42px,6vw,74px); line-height:1.03; font-weight:800;
    letter-spacing:-.04em; color:$ink; margin:0 0 22px; max-width:17ch;
    text-shadow: 0 2px 34px rgba(0,0,0,.4);
  }
  .hero h1 em {
    font-style:normal; position:relative;
    background: linear-gradient(100deg,$s0,$s2 46%,$s1 78%,$s0);
    background-size: 260% 100%;
    -webkit-background-clip:text; background-clip:text; color:transparent;
    animation: shimmer 9s ease-in-out infinite;
  }
  @keyframes shimmer {
    0%,100% { background-position: 0% 50%; }
    50%     { background-position: 100% 50%; }
  }
  .lead {
    font-size: clamp(16.5px,1.4vw,20px); line-height:1.65; color:$ink2;
    max-width:60ch; margin:0 0 30px;
  }
  .lead em { font-style:normal; color:$ink; font-weight:640; }

  /* ---------- stat tiles ---------- */
  .stats {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
    gap:14px; margin:38px 0 6px;
  }
  .stat {
    position:relative; overflow:hidden;
    background:$surface; backdrop-filter: blur(16px) saturate(130%);
    border:1px solid $border; border-radius:16px; padding:19px 20px 16px;
    transition:.3s cubic-bezier(.22,.7,.3,1);
  }
  .stat::before {
    content:""; position:absolute; left:0; top:0; height:3px; width:100%;
    background: linear-gradient(90deg,$s0,$s2);
  }
  .stat:hover { transform:translateY(-3px); border-color:$borderHi; }
  .stat .v {
    font-size:33px; font-weight:700; letter-spacing:-.032em; color:$ink;
    line-height:1.05;
  }
  .stat .k { font-size:13px; color:$ink2; margin-top:8px; line-height:1.45;
             font-weight:500; }
  .stat .spk { width:100%; height:26px; margin-top:12px; opacity:.85; }

  /* ---------- section headings ---------- */
  .sec { margin:70px 0 24px; }
  .sec .kicker {
    display:inline-flex; align-items:center; gap:8px;
    font-size:11.5px; letter-spacing:.16em; text-transform:uppercase;
    font-weight:700; margin-bottom:12px;
    background:linear-gradient(90deg,$s0,$s2);
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .sec h2 {
    font-size:clamp(28px,2.8vw,38px); font-weight:800; letter-spacing:-.035em;
    color:$ink; margin:0 0 14px; line-height:1.15;
  }
  .sec p { color:$ink2; font-size:16px; line-height:1.7; margin:0; max-width:66ch; }
  .rule {
    height:1px; width:100%; margin-bottom:22px;
    background:linear-gradient(90deg,$borderHi,transparent 62%);
  }

  /* ---------- pipeline --------------------------------------------------
     Colours and timings live HERE, keyed by position. Some Streamlit
     versions strip CSS variables from inline style="" attributes, which
     left every step colourless and firing at once. */
  .flowbox {
    position:relative; border:1px solid $border; border-radius:22px;
    background:$surface; backdrop-filter: blur(16px) saturate(130%);
    padding:26px 22px 22px; overflow:hidden;
  }
  .flowrow { position:relative; display:grid;
             grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; }
  .fstep:nth-child(1) { --c:$s0; --d:0s;   }
  .fstep:nth-child(2) { --c:$s1; --d:1s;   }
  .fstep:nth-child(3) { --c:$s2; --d:2s;   }
  .fstep:nth-child(4) { --c:$s3; --d:3s;   }
  .fstep:nth-child(5) { --c:$s0; --d:4s;   }
  .running .fstep:nth-child(2) { --d:.5s; } .running .fstep:nth-child(3) { --d:1s; }
  .running .fstep:nth-child(4) { --d:1.5s; } .running .fstep:nth-child(5) { --d:2s; }

  .fstep {
    position:relative; border-radius:16px; padding:18px 14px 16px;
    background:$raised; border:1px solid $border; text-align:left;
    animation: activate 5s ease-in-out infinite; animation-delay:var(--d);
  }
  .running .fstep { animation-duration:2.5s; }
  /* the connector chevron between cards */
  .fstep:not(:last-child)::after {
    content:""; position:absolute; top:34px; right:-11px; width:8px; height:8px;
    border-top:2px solid $borderHi; border-right:2px solid $borderHi;
    transform: rotate(45deg); z-index:2;
  }
  @keyframes activate {
    0%, 26%, 100% { border-color:$border; transform:translateY(0);
                    box-shadow:0 0 0 0 transparent; }
    8%, 18%       { border-color:var(--c); transform:translateY(-4px);
                    box-shadow:0 18px 40px -22px var(--c),
                               inset 0 0 0 1px var(--c); }
  }
  .ftop { display:flex; align-items:center; justify-content:space-between;
          margin-bottom:14px; }
  .fico {
    width:38px; height:38px; border-radius:11px; display:flex;
    align-items:center; justify-content:center; color:var(--c);
    background:$solid; border:1px solid $border;
  }
  .fico svg { width:19px; height:19px; }
  .fnum { font: 700 11px/1 "Inter", sans-serif; letter-spacing:.14em;
          color:var(--c); }
  .ftitle { font-size:16.5px; font-weight:700; color:$ink;
            letter-spacing:-.015em; margin-bottom:6px; }
  .fsub { font-size:13.5px; line-height:1.55; color:$ink2; }
  .fbar { position:absolute; left:14px; right:14px; bottom:0; height:2px;
          border-radius:2px; background:var(--c); opacity:0;
          animation: barfill 5s ease-in-out infinite; animation-delay:var(--d);
          transform-origin:left; }
  .running .fbar { animation-duration:2.5s; }
  @keyframes barfill {
    0%, 26%, 100% { opacity:0; transform:scaleX(0); }
    6%            { opacity:1; transform:scaleX(.2); }
    18%           { opacity:1; transform:scaleX(1); }
  }
  .flowcap { display:flex; align-items:center; justify-content:center; gap:10px;
             margin-top:18px; font-size:14px; color:$ink; font-weight:600; }
  .flowcap .spin { width:14px; height:14px; border-radius:50%;
                   border:2px solid $border; border-top-color:$s0;
                   animation: spin .8s linear infinite; }
  @media (max-width:900px) {
    .flowrow { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .fstep:not(:last-child)::after { display:none; }
  }
  /* Reduced motion (Windows "Animation effects" off): a slower, gentler
     cycle rather than a dead diagram. */
  @media (prefers-reduced-motion: reduce) {
    .fstep { animation: activate 10s ease-in-out infinite !important;
             animation-delay: calc(var(--d) * 2) !important; }
    .fbar { animation: none !important; }
  }

  /* ---------- compact result tiles ---------- */
  .stats.res { grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
               margin:18px 0 8px; }
  .stats.res .stat { padding:15px 16px 13px; }
  .stats.res .stat .v { font-size:24px; white-space:nowrap; }
  .stats.res .stat .k { font-size:11.5px; letter-spacing:.07em;
                        text-transform:uppercase; font-weight:650; margin-top:5px; }
  .v.good { color:$good !important; } .v.bad { color:$crit !important; }

  /* ---------- suggestion label ---------- */
  .suglbl { font-size:11.5px; letter-spacing:.13em; text-transform:uppercase;
            color:$muted; font-weight:700; margin:4px 0 8px; }

  /* ---------- pipeline (old svg) ---------- */
  .pipe {
    border:1px solid $border; border-radius:20px; background:$surface;
    backdrop-filter: blur(16px) saturate(130%); padding:28px 24px 18px;
  }
  .flow { stroke-dasharray:5 9; animation: dash 1.5s linear infinite; }
  @keyframes dash { to { stroke-dashoffset:-28; } }
  .pnode { font-size:12.5px; fill:$ink; font-weight:650; }
  .pdesc { font-size:11px; fill:$muted; }

  /* ---------- cards ---------- */
  .grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
  @media (max-width:900px) { .grid3 { grid-template-columns:1fr; } }
  .grid2 { display:grid; grid-template-columns:1.32fr 1fr; gap:16px; }
  @media (max-width:820px) { .grid2 { grid-template-columns:1fr; } }

  .card {
    position:relative; overflow:hidden; isolation:isolate;
    border:1px solid $border; border-radius:20px; background:$surface;
    backdrop-filter: blur(16px) saturate(130%); padding:26px 24px 24px;
    transition:.32s cubic-bezier(.22,.7,.3,1);
    display:flex; flex-direction:column;
  }
  /* gradient ring, drawn only on hover */
  .card::before {
    content:""; position:absolute; inset:0; border-radius:20px; padding:1px;
    background: linear-gradient(140deg,$s0,$s2 46%,$s1);
    -webkit-mask: linear-gradient(#000 0 0) content-box,
                  linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity:0; transition:opacity .32s ease; pointer-events:none;
  }
  /* diagonal sheen sweeping across on hover */
  .card::after {
    content:""; position:absolute; top:-60%; left:-70%;
    width:45%; height:220%; pointer-events:none;
    background: linear-gradient(90deg,transparent,$sheen,transparent);
    transform: rotate(18deg); transition: left .75s cubic-bezier(.22,.7,.3,1);
  }
  .card:hover {
    transform:translateY(-6px);
    box-shadow:0 26px 60px -28px rgba(0,0,0,.8);
  }
  .card:hover::before { opacity:1; }
  .card:hover::after { left:130%; }

  /* a thin accent bar per card, and a numbered pill you can actually read */
  .card { border-top:0; }
  .grid3 .card:nth-child(1) { --a:$s0; } .grid3 .card:nth-child(2) { --a:$s2; }
  .grid3 .card:nth-child(3) { --a:$s1; } .grid2 .card { --a:$s3; }
  .card > .bar2 { position:absolute; left:0; right:0; top:0; height:3px;
                  background:linear-gradient(90deg,var(--a),transparent 85%); }
  .card .num {
    position:absolute; top:22px; right:22px;
    font: 700 12px/1 "Inter", sans-serif; letter-spacing:.12em;
    color:var(--a); padding:7px 11px; border-radius:999px;
    border:1px solid $border; background:$raised; pointer-events:none;
  }
  .card .ico {
    width:40px;height:40px;border-radius:12px;display:flex;align-items:center;
    justify-content:center;margin-bottom:17px;border:1px solid $border;
  }
  .card h3 { font-size:20px;font-weight:700;color:$ink;margin:2px 0 12px;
             letter-spacing:-.022em; line-height:1.25; }
  .card p { font-size:15px;line-height:1.72;color:$ink2;margin:0; }
  .card p b { color:$ink; font-weight:650; }
  .card p + p { margin-top:14px; }
  .card .tag {
    margin-top:auto; padding-top:16px; border-top:1px solid $border;
    font-size:11.5px; letter-spacing:.08em; text-transform:uppercase;
    color:var(--a); font-weight:700;
  }
  .card p:last-of-type { margin-bottom:20px; }

  /* ---------- worked example ---------- */
  .ex {
    border:1px solid $border;border-radius:20px;background:$surface;
    backdrop-filter: blur(16px) saturate(130%); overflow:hidden;
  }
  .ex .bar {
    display:flex;align-items:center;gap:8px;padding:11px 18px;
    background:$raised;border-bottom:1px solid $border;
  }
  .ex .bar i { width:9px;height:9px;border-radius:50%;display:block; }
  .ex .bar span {
    margin-left:8px;font-size:11px;letter-spacing:.13em;text-transform:uppercase;
    color:$muted;font-weight:650;
  }
  .ex .q {
    padding:18px 24px;border-bottom:1px solid $border;
    font-size:15.5px;color:$ink;font-weight:600;
  }
  .ex .q u {
    text-decoration:none;color:$muted;font-weight:600;margin-right:10px;
    font-size:12px;letter-spacing:.1em;
  }
  .ex .b { padding:22px 24px; }
  .ex .row { margin-bottom:18px; padding-left:14px;
             border-left:2px solid $border; }
  .ex .row:last-child { margin-bottom:0; }
  .ex .row.k0 { border-left-color:$s0; }
  .ex .row.k1 { border-left-color:$s2; }
  .ex .row.k2 { border-left-color:$s3; }
  .ex .lbl {
    font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;
    color:$muted;margin-bottom:7px;font-weight:700;
  }
  .ex .txt { font-size:15px;line-height:1.7;color:$ink2; }
  .ex .txt b { color:$ink;font-weight:660; }
  .ex .meta {
    padding:14px 24px;border-top:1px solid $border;background:$raised;
    font-size:12.5px;color:$muted;
  }

  /* ---------- format pills ---------- */
  .fmts { display:flex;gap:10px;flex-wrap:wrap;margin-top:10px; }
  .fmt {
    border:1px solid $border;border-radius:12px;background:$surface;
    backdrop-filter: blur(12px); padding:12px 18px;font-size:13.5px;color:$ink2;
    transition:.26s ease;
  }
  .fmt:hover { border-color:$borderHi; transform:translateY(-2px); }
  .fmt b { color:$ink;font-weight:660; }

  /* ---------- page heads / footer ---------- */
  .crumb { color:$muted;font-size:11.5px;letter-spacing:.15em;
           text-transform:uppercase;margin-bottom:9px;font-weight:700; }
  .pagehead { padding-top:26px; }
  .pagehead h1 { font-size:40px;font-weight:800;letter-spacing:-.035em;
                 color:$ink;margin:0 0 12px; }
  .pagehead p { color:$ink2;font-size:16px;line-height:1.7;margin:0;
                max-width:70ch; }
  .pagehead p b { color:$ink; }

  .foot {
    margin-top:76px;padding:26px 4px 34px;border-top:1px solid $border;
    display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;
    font-size:12.5px;color:$muted;
  }
  .foot b { color:$ink2;font-weight:640; }

  /* Streamlit widgets sit on glass too */
  [data-testid="stExpander"], [data-testid="stFileUploader"] {
    background:$surface !important; backdrop-filter: blur(14px);
    border:1px solid $border !important; border-radius:14px !important;
  }
  [data-testid="stMetric"] {
    background:$surface; backdrop-filter: blur(14px);
    border:1px solid $border; border-radius:14px; padding:14px 16px;
  }
  .stButton button { border-radius:11px !important; font-weight:600 !important; }
</style>
"""


def inject_css():
    c = p()
    dark = st.session_state.get("mode", "dark") == "dark"
    # the field is the categorical hues at low alpha - same family as the
    # charts, so the page and the data read as one system
    fields = (["rgba(57,135,229,.55)", "rgba(144,133,233,.45)",
               "rgba(25,158,112,.42)", "rgba(217,89,38,.36)",
               "rgba(201,133,0,.28)"] if dark else
              ["rgba(42,120,214,.30)", "rgba(74,58,167,.22)",
               "rgba(27,175,122,.24)", "rgba(235,104,52,.20)",
               "rgba(237,161,0,.18)"])
    vig = "rgba(0,0,0,.55)" if dark else "rgba(0,0,0,.10)"
    sheen = "rgba(255,255,255,.07)" if dark else "rgba(255,255,255,.55)"

    css = CSS
    for i, g in enumerate(fields):
        css = css.replace(f"$g{i}", g)
    css = (css.replace("$vig", vig).replace("$sheen", sheen)
              .replace("$surface", c["surface"]).replace("$plane", c["plane"])
              .replace("$raised", c["raised"]).replace("$solid", c["solid"])
              .replace("$ink2", c["ink2"]).replace("$ink", c["ink"])
              .replace("$muted", c["muted"])
              .replace("$borderHi", c["borderHi"]).replace("$border", c["border"])
              .replace("$s0", c["series"][0]).replace("$s1", c["series"][1])
              .replace("$s2", c["series"][2]).replace("$s3", c["series"][3])
              .replace("$grid", c["grid"]).replace("$axis", c["axis"])
              .replace("$good", c["good"]).replace("$crit", c["critical"])
              .replace("$warn", c["warn"])
              .replace("$ringc", "rgba(255,255,255,.10)" if dark
                       else "rgba(0,0,0,.07)")
              .replace("$ring", "rgba(57,135,229,.28)" if dark
                       else "rgba(42,120,214,.20)"))
    st.markdown(css, unsafe_allow_html=True)


inject_css()


# --------------------------------------------------------------- fragments
def ico(path: str, colour: str) -> str:
    return (f'<div class="ico" style="background:{colour}22;'
            f'box-shadow:0 6px 18px -10px {colour}">'
            f'<svg width="19" height="19" viewBox="0 0 24 24" fill="none" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{path}</svg></div>')


def spark(vals, colour, w: int = 100, h: int = 26) -> str:
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = max(len(vals) - 1, 1)
    pts = " ".join(f"{i * w / n:.1f},{h - (v - lo) / rng * (h - 5) - 2.5:.1f}"
                   for i, v in enumerate(vals))
    return (f'<svg class="spk" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round"/></svg>')


def stat(value: str, label: str, series=None, colour: str | None = None) -> str:
    c = p()
    line = spark(series, colour or c["series"][0]) if series else ""
    return (f'<div class="stat"><div class="v">{value}</div>'
            f'<div class="k">{label}</div>{line}</div>')


def topbar(right: str = ""):
    c = p()
    st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="mark">
      <svg width="21" height="21" viewBox="0 0 24 24" fill="none"
           stroke="#ffffff" stroke-width="2" stroke-linecap="round"
           stroke-linejoin="round">
        <path d="M12 3 L18.5 20 L12 16 L5.5 20 Z" fill="#ffffff"
              fill-opacity=".92" stroke="none"/>
        <circle class="orbit" cx="12" cy="12" r="9.2" stroke="#ffffff"
                stroke-opacity=".75" stroke-dasharray="7 10" fill="none"/>
      </svg>
    </div>
    <div>
      <div class="name">Insight<b>Pilot</b></div>
      <div class="tagline">Autonomous analytics agent</div>
    </div>
  </div>
  <div class="barmeta">{right}</div>
</div>
""", unsafe_allow_html=True)


def footer():
    st.markdown("""
<div class="foot">
  <div><b>InsightPilot</b> · agentic analytics over your own data</div>
  <div>LangGraph · DuckDB · Groq &amp; Gemini · Plotly</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- pipeline
FLOW = [("Clarify", "Asks when a term could mean two things"),
        ("Plan", "Decides exactly what to measure"),
        ("Compute", "Runs checked SQL on your data"),
        ("Critique", "Tests whether the cause is real"),
        ("Ground", "Verifies every number it quotes")]


FLOW_ICONS = [
    '<path d="M12 17h.01"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><circle cx="12" cy="12" r="10"/>',
    '<path d="M9 6h11"/><path d="M9 12h11"/><path d="M9 18h11"/><path d="M4 6h.01"/><path d="M4 12h.01"/><path d="M4 18h.01"/>',
    '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
]


def pipeline_html(running: bool = False, caption: str = "") -> str:
    """Five steps as real HTML text, lit in turn. Colours and delays come
    from the stylesheet by position, not from inline CSS variables, which
    some Streamlit versions strip."""
    steps = "".join(
        f'<div class="fstep"><div class="ftop"><div class="fico">'
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{FLOW_ICONS[i]}</svg></div><div class="fnum">0{i + 1}</div></div>'
        f'<div class="ftitle">{t}</div><div class="fsub">{d}</div>'
        f'<div class="fbar"></div></div>'
        for i, (t, d) in enumerate(FLOW))
    cap = (f'<div class="flowcap">{"<span class=spin></span>" if running else ""}'
           f'{caption}</div>') if caption else ""
    return (f'<div class="flowbox{" running" if running else ""}">'
            f'<div class="flowrow">{steps}</div>{cap}</div>')


# ------------------------------------------------------------ copy button
def copy_button(text: str, key: str, label: str = "Copy SQL"):
    """A copy button that works. Tries the Clipboard API, then falls back to
    execCommand, which still works inside the component iframe when the
    newer API is blocked."""
    c = p()
    components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@600&display=swap" rel="stylesheet">
<button id="b" style="font:600 14.5px 'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  color:{c['ink']};background:{c['raised']};border:1px solid {c['borderHi']};
  border-radius:12px;padding:10px 18px;cursor:pointer;width:100%;
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;"
  onmouseover="this.style.transform='translateY(-2px)';this.style.borderColor='{c['series'][0]}'"
  onmouseout="this.style.transform='none';this.style.borderColor='{c['borderHi']}'"
  >{_html.escape(label)}</button>
<script>
  const txt = {json.dumps(text)};
  const b = document.getElementById("b");
  function done(ok) {{
    b.textContent = ok ? "Copied \u2713" : "Select the SQL and press Ctrl+C";
    setTimeout(() => b.textContent = {json.dumps(label)}, 1800);
  }}
  function legacy() {{
    const a = document.createElement("textarea");
    a.value = txt; a.style.position = "fixed"; a.style.opacity = "0";
    document.body.appendChild(a); a.focus(); a.select();
    let ok = false; try {{ ok = document.execCommand("copy"); }} catch (e) {{}}
    a.remove(); done(ok);
  }}
  b.onclick = () => {{
    if (navigator.clipboard && window.isSecureContext) {{
      navigator.clipboard.writeText(txt).then(() => done(true), legacy);
    }} else {{ legacy(); }}
  }};
</script>
<style>body{{margin:0;padding:2px 1px;background:transparent;}}</style>
""", height=50)


# --------------------------------------------------------- dataset summary
NUMERIC_T = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT",
             "DOUBLE", "DECIMAL", "REAL", "UBIGINT", "UINTEGER")
DATE_T = ("DATE", "TIMESTAMP")


def _qi(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


@st.cache_data(show_spinner=False)
def dataset_summary(_db: str) -> dict:
    """Exact summary statistics for every column of every table.

    Computed with plain aggregates rather than DuckDB's SUMMARIZE, which
    reports APPROXIMATE distinct counts and medians - on the hospital data
    it gave 11,212 unique IDs for 12,000 unique rows. A summary people will
    quote has to be exact."""
    out = {}
    con = connect()
    try:
        for t in tables():
            n = con.execute(f"SELECT COUNT(*) FROM {_qi(t)}").fetchone()[0] or 0
            info = con.execute(f"PRAGMA table_info({_qi(t)})").fetchall()
            nums, cats, dates = [], [], []
            for _, col, typ, *_ in info:
                typ = str(typ).upper()
                q = _qi(col)
                miss = con.execute(
                    f"SELECT 100.0 * COUNT(*) FILTER (WHERE {q} IS NULL) / "
                    f"NULLIF(COUNT(*), 0) FROM {_qi(t)}").fetchone()[0] or 0.0
                if any(k in typ for k in NUMERIC_T):
                    r = con.execute(
                        f"SELECT AVG({q}), MEDIAN({q}), STDDEV_SAMP({q}), MIN({q}), "
                        f"QUANTILE_CONT({q}, 0.25), QUANTILE_CONT({q}, 0.75), MAX({q}), "
                        f"COUNT(DISTINCT {q}) FROM {_qi(t)}").fetchone()
                    nums.append({"column": col, "mean": r[0], "median": r[1],
                                 "std dev": r[2], "min": r[3], "25%": r[4],
                                 "75%": r[5], "max": r[6], "distinct": r[7],
                                 "missing %": round(miss, 1)})
                elif any(k in typ for k in DATE_T):
                    r = con.execute(f"SELECT MIN({q}), MAX({q}) FROM {_qi(t)}").fetchone()
                    span = (pd.Timestamp(r[1]) - pd.Timestamp(r[0])).days if r[0] else None
                    dates.append({"column": col, "earliest": str(r[0])[:10],
                                  "latest": str(r[1])[:10],
                                  "span (days)": span, "missing %": round(miss, 1)})
                else:
                    d = con.execute(f"SELECT COUNT(DISTINCT {q}) FROM {_qi(t)}").fetchone()[0]
                    top = con.execute(
                        f"SELECT CAST({q} AS VARCHAR), COUNT(*) FROM {_qi(t)} "
                        f"WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 1"
                    ).fetchone()
                    low = str(col).lower()
                    is_id = (low == "id" or low.endswith("_id") or low.endswith("id")
                             and d > 25 or bool(n and d >= 0.95 * n))
                    kind = ("an ID" if n and d >= 0.95 * n else
                            "an ID (repeats)" if is_id else
                            "a category" if d <= 25 else "free text")
                    cats.append({"column": col, "distinct": d,
                                 # the most common ID is trivia, not a finding
                                 "most common": None if is_id else (top[0] if top else None),
                                 "its share %": None if is_id else (
                                     round(100.0 * top[1] / n, 1) if top and n else None),
                                 "missing %": round(miss, 1), "looks like": kind})
            out[t] = {"rows": n, "numeric": nums, "categorical": cats, "dates": dates}
    finally:
        con.close()
    return out


def render_summary():
    ds = dataset.get_active()
    try:
        summ = dataset_summary(ds["db_path"])
    except Exception as e:
        st.warning(f"Could not compute the summary: {e}")
        return
    st.markdown("""
<div class="sec" style="margin-top:44px">
  <div class="kicker">Dataset summary</div>
  <h2>Every column at a glance</h2>
  <div class="rule"></div>
  <p>Exact statistics over the full table - not a sample. Check these before
  asking questions: a mean far from its median means a skewed column, and a
  high missing % limits what that column can tell you.</p>
</div>""", unsafe_allow_html=True)

    fmt = lambda v: "" if v is None else (f"{v:,.2f}" if isinstance(v, float) else v)
    for t, info in summ.items():
        with st.expander(f"**{t}** — {info['rows']:,} rows · "
                         f"{len(info['numeric'])} numeric · "
                         f"{len(info['categorical'])} text · "
                         f"{len(info['dates'])} date", expanded=len(summ) == 1):
            if info["numeric"]:
                st.markdown("**Numeric columns**")
                df = pd.DataFrame(info["numeric"])
                for col in ["mean", "median", "std dev", "min", "25%", "75%", "max"]:
                    df[col] = df[col].map(fmt)
                st.dataframe(df, use_container_width=True, hide_index=True)

                pick = st.selectbox("Distribution of", [r["column"] for r in info["numeric"]],
                                    key=f"dist_{t}")
                vals = run_sql(f"SELECT {_qi(pick)} AS v FROM {_qi(t)} "
                               f"WHERE {_qi(pick)} IS NOT NULL", limit=200_000)
                c = p()
                fig = go.Figure(go.Histogram(x=vals["v"], nbinsx=40,
                                             marker_color=c["seq"],
                                             marker_line=dict(width=1, color=c["solid"])))
                med = next(r["median"] for r in info["numeric"] if r["column"] == pick)
                if med is not None:
                    fig.add_vline(x=med, line_dash="dash", line_color=c["series"][1],
                                  annotation_text=f"median {med:,.2f}",
                                  annotation_font_color=c["ink2"])
                st.plotly_chart(style(fig, 240), use_container_width=True,
                                config={"displayModeBar": False})
            if info["categorical"]:
                st.markdown("**Text and category columns**")
                st.dataframe(pd.DataFrame(info["categorical"]),
                             use_container_width=True, hide_index=True)
            if info["dates"]:
                st.markdown("**Date columns**")
                st.dataframe(pd.DataFrame(info["dates"]),
                             use_container_width=True, hide_index=True)


# ---------------------------------------------------- suggested questions
SCOREISH = ("score", "rating", "csat", "satisfaction", "nps")
# in order of preference: revenue beats a list price, and "seats_billed" is
# a count of seats, not money, whatever its name says
MONEYISH = ("revenue", "net", "paid", "sales", "income", "amount", "price",
            "cost", "spend", "fee", "value", "billed")
NOT_MONEY = ("seat", "count", "qty", "quantity", "units", "users", "calls",
             "sessions", "hours", "days")
BADVALUES = ("denied", "failed", "cancelled", "canceled", "churned", "refunded",
             "rejected", "returned", "lost", "defaulted")


def _h(name: str) -> str:
    return str(name).replace("__", " ").replace("_", " ").strip()


@st.cache_data(show_spinner=False)
def suggested_questions(_db: str) -> list[str]:
    """Four questions built from THIS dataset's own columns - one of each kind
    the agent handles. Code, not a model call: instant, free, and the column
    names are always real."""
    try:
        dr = date_ranges()
        vals = column_values()
        con = connect()
    except Exception:
        return []
    try:
        types = {}
        for t in tables():
            for _, col, typ, *_ in con.execute(f"PRAGMA table_info({_qi(t)})").fetchall():
                types[(t, col)] = str(typ).upper()

        def numerics(t):
            return [c for (tt, c), ty in types.items() if tt == t
                    and any(k in ty for k in NUMERIC_T) and not c.lower().endswith("id")
                    and f"{t}.{c}" not in vals]

        # the fact table: a dated table with the most measures
        dated = [(k.split(".", 1)[0], k.split(".", 1)[1], d) for k, d in dr.items()
                 if d.get("previous")]
        if not dated:
            return []
        t, tcol, d = max(dated, key=lambda x: len(numerics(x[0])))
        nums = numerics(t)
        score = next((c for c in nums if any(h in c.lower() for h in SCOREISH)), None)
        money = sorted(
            (c for c in nums if any(h in c.lower() for h in MONEYISH)
             and not any(x in c.lower() for x in NOT_MONEY)),
            key=lambda c: min(i for i, h in enumerate(MONEYISH) if h in c.lower()))
        metric = score or (money[0] if money else (nums[0] if nums else None))
        agg = "average" if metric == score or not money or metric not in money else "total"
        cats = [(c, v) for k, v in vals.items() for c in [k.split(".", 1)[1]]
                if k.split(".", 1)[0] == t]
        flags = [c for c, v in cats if sorted(v) == ["false", "true"]]
        dims = [c for c, v in cats if 3 <= len(v) <= 12 and "status" not in c.lower()]
        qs = []

        if metric:
            prev, last = d["previous"], d["latest"]
            fn = "AVG" if agg == "average" else "SUM"
            r = con.execute(
                f"SELECT {fn}(CASE WHEN {_qi(tcol)} >= TIMESTAMP '{prev[1]}' AND "
                f"{_qi(tcol)} < TIMESTAMP '{prev[2]}' THEN {_qi(metric)} END), "
                f"{fn}(CASE WHEN {_qi(tcol)} >= TIMESTAMP '{last[1]}' AND "
                f"{_qi(tcol)} < TIMESTAMP '{last[2]}' THEN {_qi(metric)} END) "
                f"FROM {_qi(t)}").fetchone()
            if r[0] is not None and r[1] is not None and r[0] != r[1]:
                word = "fall" if r[1] < r[0] else "rise"
                qs.append(f"Why did {agg} {_h(metric)} {word} in {last[0]}?")

        if flags:
            qs.append(f"What predicts {_h(flags[0])}?")
        else:
            for c, v in cats:
                bad = next((x for x in v if x.lower() in BADVALUES), None)
                if bad:
                    qs.append(f"What predicts {_h(c)} being {bad}?")
                    break

        if metric and dims:
            qs.append(f"Which {_h(dims[0])} has the highest {agg} {_h(metric)}?")
        m2 = next((c for c in money if c != metric), None) or (money[0] if money else None)
        if m2 and len(dims) > 1:
            qs.append(f"What is the total {_h(m2)} by {_h(dims[1])}?")
        elif metric:
            qs.append(f"How has {agg} {_h(metric)} changed by quarter?")
        return qs[:4]
    except Exception:
        return []
    finally:
        con.close()


# ----------------------------------------------------------------- charts
def style(fig: go.Figure, height: int = 320) -> go.Figure:
    c = p()
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=c["solid"], plot_bgcolor=c["solid"],
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color=c["ink2"]),
        hoverlabel=dict(bgcolor=c["solid"], font_size=12,
                        bordercolor=c["axis"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color=c["ink2"])),
        showlegend=len(fig.data) > 1,
    )
    fig.update_xaxes(showgrid=False, linecolor=c["axis"], zeroline=False,
                     tickfont=dict(color=c["muted"], size=11))
    fig.update_yaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False,
                     tickfont=dict(color=c["muted"], size=11))
    return fig


def pick_chart(df: pd.DataFrame) -> go.Figure | None:
    """The data's job picks the form. A table is often the honest answer, so
    return None freely - a misleading chart is worse than no chart."""
    if df is None or df.empty or len(df.columns) < 2 or len(df) > 25:
        return None
    if len(df) < 2:
        return None          # a single row is a figure, not a chart

    c = p()
    label = df.columns[0]
    is_time = (pd.api.types.is_datetime64_any_dtype(df[label])
               or any(h in str(label).lower() for h in TIME_HINTS))

    # A numeric first column is DATA, not an axis. Charting it puts the value
    # on the x-axis and draws one meaningless bar labelled "181.8457278M".
    # Ordinal offsets (month_offset, weeks_before) are time-hinted, so those
    # still chart correctly as a line.
    if pd.api.types.is_numeric_dtype(df[label]) and not is_time:
        return None

    # Repeated labels mean long format - one row per label PER PERIOD. Drawn
    # as-is, both periods land on the same x position and silently overlap.
    # Pivoting it here would guess at which column is the period, so the
    # table is the honest answer.
    if df[label].duplicated().any():
        return None

    nums = [col for col in df.columns[1:]
            if pd.api.types.is_numeric_dtype(df[col])][:4]
    if not nums:
        return None

    # Quantities of different orders of magnitude cannot share an axis:
    # seats (~40) beside discount (~4,900) renders the seats as a flat line
    # on zero, with a legend entry for a series nobody can see.
    scale = {col: float(df[col].abs().max() or 0) for col in nums}
    top = max(scale.values(), default=0.0)
    if top > 0:
        nums = [col for col in nums if scale[col] >= top / 20.0]
    if not nums:
        return None

    x = df[label].astype(str)
    fig = go.Figure()

    if is_time:                           # trend -> line
        for i, col in enumerate(nums):
            fig.add_trace(go.Scatter(
                x=x, y=df[col], name=str(col), mode="lines+markers",
                line=dict(width=2, color=c["series"][i % 4]),
                marker=dict(size=8, color=c["series"][i % 4],
                            line=dict(width=2, color=c["solid"])),
                hovertemplate=f"<b>{col}</b><br>%{{x}}: %{{y:,.2f}}<extra></extra>"))
        fig.update_layout(hovermode="x unified")
    elif len(nums) == 1:                  # magnitude -> one hue
        col = nums[0]
        fig.add_trace(go.Bar(
            x=x, y=df[col], name=str(col), marker_color=c["seq"],
            marker_line=dict(width=2, color=c["solid"]),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,.2f}}<extra></extra>"))
    else:                                 # identity -> categorical, fixed order
        for i, col in enumerate(nums):
            fig.add_trace(go.Bar(
                x=x, y=df[col], name=str(col), marker_color=c["series"][i % 4],
                marker_line=dict(width=2, color=c["solid"]),
                hovertemplate=f"<b>{col}</b><br>%{{x}}: %{{y:,.2f}}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)

    return style(fig)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ◆ InsightPilot")
    st.caption("Autonomous analytics agent")
    if st.session_state.stage != "landing":
        ds = dataset.get_active()
        st.divider()
        st.caption(f"**{ds['name']}** · {len(tables())} tables")
        for s, label in [("ask", "Ask questions"), ("clean", "Data profile"),
                         ("define", "Definitions"), ("upload", "New upload")]:
            if st.button(label, use_container_width=True, key=f"nav_{s}"):
                goto(s)
        st.divider()
    st.session_state.mode = st.radio(
        "Appearance", ["dark", "light"], horizontal=True,
        index=0 if st.session_state.mode == "dark" else 1)
    if st.session_state.stage != "landing":
        if st.button("Back to start", use_container_width=True):
            goto("landing")


# ==================================================================== LANDING
if st.session_state.stage == "landing":
    c = p()
    topbar('<div class="chip live"><span class="ping"></span>Agent ready</div>'
           '<div class="chip">7-table demo loaded</div>')

    st.markdown("""
<div class="hero">
  <div class="eyebrow rise d1">
    <span>LangGraph</span><span>DuckDB</span><span>Groq · Gemini</span>
    <span>Semantic layer</span>
  </div>
  <h1 class="rise d2">Hand over a problem.<br/>Get an <em>investigation</em> back.</h1>
  <p class="lead rise d3">Most tools answer <em>what</em>. Ask InsightPilot
  <em>why</em> — it plans the analysis, writes and runs its own SQL, checks the
  result against the data, and tells you which segment moved and what drove it.</p>
</div>
""", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1.05, 1.05, 3.2])
    if b1.button("Upload my data", type="primary", use_container_width=True):
        goto("upload")
    if b2.button("Explore the demo", use_container_width=True):
        dataset.use_demo()
        st.cache_data.clear()
        reset_run()
        goto("ask")

    tiles = "".join([
        stat("86.7%", "Benchmark accuracy · 3 runs, σ 7.6",
             [62, 71, 68, 80, 85, 83, 87], c["series"][0]),
        stat("10 / 12", "Root-cause questions · baseline 1 / 4",
             [1, 2, 4, 6, 7, 9, 10], c["series"][2]),
        stat("~25s", "Median diagnostic · was 537s",
             [537, 410, 260, 120, 62, 34, 25], c["series"][1]),
        stat("5", "File formats accepted",
             [1, 2, 3, 3, 4, 5, 5], c["series"][3]),
    ])
    st.markdown(f'<div class="stats rise d4">{tiles}</div>',
                unsafe_allow_html=True)

    # ---------------- pipeline
    st.markdown("""
<div class="sec rise">
  <div class="kicker">The loop</div>
  <h2>Five stages, and two of them are checks</h2>
  <div class="rule"></div>
  <p>An agent that only generates is a liability. This one validates before it
  executes and verifies before it answers.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(f'<div class="rise">{pipeline_html()}</div>', unsafe_allow_html=True)

    # ---------------- three principles
    st.markdown("""
<div class="sec rise">
  <div class="kicker">Why it's different</div>
  <h2>Three things a chat-with-your-CSV tool doesn't do</h2>
  <div class="rule"></div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="grid3">
  <div class="card rise d1"><div class="bar2"></div>
    <div class="num">01</div>
    {ico('<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>', c['series'][0])}
    <h3>It investigates</h3>
    <p>A <b>“why” question</b> becomes a full investigation: every dimension is
    broken down, the segment that carries the change is found, and what shifted
    <b>inside it and nowhere else</b> is named as the likely cause.</p>
    <div class="tag">Headline → breakdown → cause</div>
  </div>
  <div class="card rise d2"><div class="bar2"></div>
    <div class="num">02</div>
    {ico('<path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/>', c['series'][2])}
    <h3>It checks its own numbers</h3>
    <p>Every query is <b>checked against the real schema before it runs</b>, and
    the arithmetic is done in code, not by the model. <b>Every figure</b> in the
    final answer is then matched back to the evidence.</p>
    <div class="tag">Validate · compute · verify</div>
  </div>
  <div class="card rise d3"><div class="bar2"></div>
    <div class="num">03</div>
    {ico('<path d="M12 17h.01"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><circle cx="12" cy="12" r="10"/>', c['series'][1])}
    <h3>It asks instead of guessing</h3>
    <p>When “revenue” has two valid definitions in your data, it <b>stops and asks</b>
    which you meant — before computing, not after. When a field doesn't exist,
    it <b>says so</b> instead of quietly using a lookalike.</p>
    <div class="tag">Honest by design</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ---------------- example
    st.markdown("""
<div class="sec rise">
  <div class="kicker">A real answer</div>
  <h2>What comes back</h2>
  <div class="rule"></div>
  <p>Produced from the demo dataset, unedited.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="grid2">
  <div class="ex rise d1">
    <div class="bar">
      <i style="background:{c['critical']}"></i>
      <i style="background:{c['warn']}"></i>
      <i style="background:{c['good']}"></i>
      <span>InsightPilot · run log</span>
    </div>
    <div class="q"><u>ASK</u>Why did revenue drop in Q3 2025?</div>
    <div class="b">
      <div class="row k0"><div class="lbl">Answer</div><div class="txt">
        Revenue fell by <b>Rs 2.63 M</b>, from Rs 19.60 M in Q2 2025 to
        Rs 16.97 M in Q3 2025.</div></div>
      <div class="row k1"><div class="lbl">Why</div><div class="txt">
        Primarily a volume problem — order count fell by 353. The largest single
        contributor was the <b>South × Electronics</b> segment, down
        <b>Rs 1.46 M</b>, accounting for over half the total drop.</div></div>
      <div class="row k2"><div class="lbl">Caveat</div><div class="txt">
        Correlation only; the data does not prove causation.</div></div>
    </div>
    <div class="meta">4 queries · 2 rejected before execution and rewritten ·
      1 critic-driven revision · 25s</div>
  </div>
  <div class="card rise d2"><div class="bar2"></div>
    {ico('<path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/>', c['series'][3])}
    <h3>Definitions, not guesses</h3>
    <p>A semantic layer pins down what every metric means, so the same question
    returns the same number every time. Revenue excludes freight and excludes
    cancelled orders — because that's written down, not inferred.</p>
    <p>On your own data the model drafts that layer from a column profile,
    executes every definition against the real schema, and discards the ones
    that fail. You review it before anything runs.</p>
    <div class="tag">10 metrics · validated at load</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ---------------- formats
    st.markdown("""
<div class="sec rise">
  <div class="kicker">Your data</div>
  <h2>Bring whatever you have</h2>
  <div class="rule"></div>
  <p>Several files become several tables, and the relationships between them are
  detected from keys — not assumed. Messy encodings, currency symbols and mixed
  date formats are handled on the way in.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="fmts rise">
  <div class="fmt"><b>CSV</b> · any encoding</div>
  <div class="fmt"><b>Excel</b> · every sheet</div>
  <div class="fmt"><b>TSV</b></div>
  <div class="fmt"><b>JSON</b></div>
  <div class="fmt"><b>Parquet</b></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:42px'></div>", unsafe_allow_html=True)
    g1, g2, _ = st.columns([1.05, 1.05, 3.2])
    if g1.button("Get started", type="primary", use_container_width=True,
                 key="cta2"):
        goto("upload")
    if g2.button("See the demo", use_container_width=True, key="cta3"):
        dataset.use_demo()
        st.cache_data.clear()
        reset_run()
        goto("ask")
    footer()


# ===================================================================== UPLOAD
elif st.session_state.stage == "upload":
    topbar('<div class="chip">Step 1 of 3</div>')
    st.markdown("""
<div class="pagehead rise">
  <div class="crumb">Step 1 of 3</div>
  <h1>Upload your data</h1>
  <p>Any tabular format. Upload several files at once if your data spans
  multiple tables — the relationships between them are detected for you.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    files = st.file_uploader(
        "Drop files here", accept_multiple_files=True,
        type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "parquet"])
    name = st.text_input("Name this dataset", value="my_data")

    c1, c2, _ = st.columns([1.1, 1.1, 3])
    if c1.button("Clean and profile", type="primary",
                 use_container_width=True, disabled=not files):
        tmp = tempfile.mkdtemp()
        paths = []
        for f in files:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
        with st.spinner("Reading, cleaning, profiling and detecting joins…"):
            try:
                ingest(paths, name or "my_data", activate=True)
                st.cache_data.clear()
                reset_run()
                goto("clean")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

    if c2.button("Use the demo instead", use_container_width=True):
        dataset.use_demo()
        st.cache_data.clear()
        reset_run()
        goto("ask")

    with st.expander("What happens to your file"):
        st.markdown("""
- Column names are made SQL-safe; duplicates are de-conflicted
- Numbers hiding behind currency symbols, commas or `(parentheses)` are parsed
- Dates are parsed from mixed formats; four text encodings are tried in turn
- Fully empty rows and columns are dropped
- Every column is profiled **in the database over the full table** — not a
  sample, because sampling produces false uniqueness, and a false key means a
  wrong join
- Quality problems are **reported, never silently fixed**. Dropping rows changes
  every number downstream, so that decision stays yours.
""")
    footer()


# ====================================================================== CLEAN
elif st.session_state.stage == "clean":
    prof = load_profile()
    topbar('<div class="chip">Step 2 of 3</div>')
    st.markdown("""
<div class="pagehead rise">
  <div class="crumb">Step 2 of 3</div>
  <h1>What we found in your data</h1>
  <p>Everything below was measured over the full table. Cleaning that changed
  values is listed separately from problems that were only flagged.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    if not prof:
        st.info("This dataset has no cleaning profile — it's the built-in demo, "
                "which ships with a hand-written semantic layer.")
        render_summary()
        if st.button("Go to questions", type="primary"):
            goto("ask")
    else:
        n_tables = len(prof.get("tables", []))
        n_rows = sum(t["rows"] for t in prof["tables"])
        n_cols = sum(len(t["columns"]) for t in prof["tables"])
        issues = sum(len(t.get("issues", [])) for t in prof["tables"])

        st.markdown(
            '<div class="stats">'
            + stat(f"{n_tables}", "Tables")
            + stat(f"{n_rows:,}", "Rows")
            + stat(f"{n_cols}", "Columns")
            + stat(f"{issues}", "Quality flags")
            + "</div>", unsafe_allow_html=True)

        if prof.get("joins"):
            st.markdown("#### Relationships detected")
            for j in prof["joins"]:
                mark = "🔗" if j["confidence"] == "high" else "❓"
                st.markdown(f"{mark} `{j['left']}` → `{j['right']}` "
                            f"· {j['confidence']} confidence")
            st.caption("Detected where a column name matches and one side is a "
                       "true key. Low-confidence links are guesses.")

        st.markdown("#### Per-table profile")
        for t in prof["tables"]:
            flag = f" · {len(t['issues'])} flags" if t.get("issues") else ""
            with st.expander(f"**{t['name']}** — {t['rows']:,} rows × "
                             f"{len(t['columns'])} columns{flag}"):
                cleaned = {k2: v for k2, v in (t.get("notes") or {}).items()
                           if k2.startswith("_")}
                parsed = {k2: v for k2, v in (t.get("notes") or {}).items()
                          if not k2.startswith("_")}
                if cleaned or parsed:
                    st.markdown("**Cleaning applied**")
                    for k2, v in cleaned.items():
                        st.markdown(f"- {k2.lstrip('_').replace('_',' ')}: {v}")
                    for col, what in parsed.items():
                        st.markdown(f"- `{col}` {what}")
                if t.get("issues"):
                    st.markdown("**Flagged — not changed**")
                    for issue in t["issues"]:
                        st.markdown(f"- ⚠️ {issue}")
                st.dataframe(
                    pd.DataFrame(t["columns"])[
                        ["name", "dtype", "null_pct", "distinct",
                         "is_unique", "min", "max"]],
                    use_container_width=True, hide_index=True)

        st.info("Nothing was deleted or imputed. If a flag matters — duplicate "
                "rows, a mostly-empty column — fix it at the source and "
                "re-upload, so you know exactly what changed.")

        render_summary()
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        c1, c2, _ = st.columns([1.3, 1.1, 3])
        if c1.button("Next: business definitions", type="primary",
                     use_container_width=True):
            goto("define")
        if c2.button("Skip to questions", use_container_width=True):
            goto("ask")
    footer()


# ===================================================================== DEFINE
elif st.session_state.stage == "define":
    topbar('<div class="chip">Step 3 of 3</div>')
    st.markdown("""
<div class="pagehead rise">
  <div class="crumb">Step 3 of 3</div>
  <h1>Business definitions</h1>
  <p>This is what stops the agent inventing its own version of “revenue” on
  every question. The model drafts definitions from your column profile; each
  one is executed against the real schema, and any that fail are discarded.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    sem = load_semantics()

    if not sem:
        st.warning("No definitions yet. Without them the agent works from the "
                   "column profile alone and states its own assumptions.")
        if st.button("Infer definitions", type="primary"):
            from tools.semantic import infer
            with st.spinner("Drafting and validating…"):
                try:
                    infer(write=True, verbose=False)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
    else:
        metrics = sem.get("metrics") or {}
        amb = ambiguous_terms()
        st.markdown(
            '<div class="stats">'
            + stat(f"{len(metrics)}", "Metrics defined and validated")
            + stat(f"{len(sem.get('joins') or [])}", "Joins")
            + stat(f"{len(amb)}", "Terms the agent will ask about")
            + stat(f"{len(sem.get('dimensions') or {})}", "Dimensions")
            + "</div>", unsafe_allow_html=True)

        left, right = st.columns([1.3, 1])
        with left:
            st.markdown("#### Metrics")
            for nm, m in metrics.items():
                st.markdown(f"**{nm}**")
                st.code(m.get("definition", ""), language="sql")
                if m.get("filter"):
                    st.caption(f"filter: `{m['filter']}`")
                if m.get("note"):
                    st.caption(m["note"])
        with right:
            st.markdown("#### You'll be asked about")
            if amb:
                for a in amb:
                    st.markdown(f"**{a['term']}**")
                    st.caption(a["ask"])
            else:
                st.caption("Nothing ambiguous detected.")

        with st.expander("Edit the raw definitions"):
            st.caption("These are the model's guesses. Fix anything wrong — "
                       "everything downstream uses this file.")
            edited = st.text_area(
                "semantic.yml",
                value=yaml.safe_dump(sem, sort_keys=False, allow_unicode=True),
                height=380, label_visibility="collapsed")
            if st.button("Save definitions"):
                try:
                    save_semantics(yaml.safe_load(edited))
                    st.cache_data.clear()
                    st.success("Saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid YAML: {e}")

        if st.button("Start asking questions", type="primary"):
            goto("ask")
    footer()


# ======================================================================== ASK
elif st.session_state.stage == "ask":
    ds = dataset.get_active()
    sem = load_semantics()

    topbar(f'<div class="chip live"><span class="ping"></span>'
           f'{ds["name"]}</div>'
           f'<div class="chip">{len(tables())} tables</div>'
           f'<div class="chip">'
           f'{"semantic layer on" if sem else "no definitions"}</div>')

    st.markdown("""
<div class="pagehead rise">
  <div class="crumb">Ask</div>
  <h1>Ask your data anything</h1>
  <p>Start a question with <b>why</b> or <b>what caused</b> and the agent runs a
  full investigation instead of a single query.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    EXAMPLES = (["Why did revenue drop in Q3 2025?",
                 "What caused the decline in review scores in Q3 2025?",
                 "Our shipping costs spiked in Q3 2025 — what drove it?",
                 "Which region generated the most revenue in 2025?"]
                if ds["kind"] == "demo" else suggested_questions(ds["db_path"]))

    if EXAMPLES:
        st.markdown('<div class="suglbl">Try one of these — built from your '
                    'data</div>', unsafe_allow_html=True)
        for row in range(0, len(EXAMPLES), 2):
            cols = st.columns(2)
            for col, ex in zip(cols, EXAMPLES[row:row + 2]):
                if col.button(ex, use_container_width=True, key=f"sug_{ex}"):
                    st.session_state.question = ex
                    reset_run()
                    st.rerun()
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    q = st.text_input("Business question", value=st.session_state.question,
                      placeholder="Why did revenue drop last quarter?",
                      label_visibility="collapsed")

    c1, c2, _ = st.columns([1, 1, 5])
    go_now = c1.button("Investigate", type="primary", use_container_width=True)
    if c2.button("Clear", use_container_width=True):
        st.session_state.question = ""
        reset_run()
        st.rerun()

    if go_now and q.strip():
        st.session_state.question = q
        reset_run()
        reset_usage()
        t0 = time.time()
        live = st.empty()
        live.markdown(pipeline_html(running=True,
                                    caption="Investigating your data — usually 20 to 60 seconds"),
                      unsafe_allow_html=True)
        try:
            state = investigate(q, clarifications={}, ask=True,
                                interactive=False)
        except Exception as e:
            live.empty()
            st.error(f"{type(e).__name__}: {e}")
            state = {}
        live.empty()
        if state.get("needs_clarification"):
            st.session_state.pending = state["needs_clarification"]
        elif state:
            state["_elapsed"] = time.time() - t0
            st.session_state.result = state

    # ---- clarification: ask before computing, never guess
    if st.session_state.pending and not st.session_state.result:
        st.info("One term in your question has more than one valid definition "
                "here. Your answer changes the numbers, so it's asked first.")
        with st.form("clarify"):
            answers = {}
            for a in st.session_state.pending:
                answers[a["term"]] = st.text_input(
                    a["ask"], key=f"clar_{a['term']}",
                    placeholder="Answer, or leave blank for the default")
            if st.form_submit_button("Continue", type="primary"):
                clar = {k2: v.strip() for k2, v in answers.items() if v.strip()}
                reset_usage()
                t0 = time.time()
                live = st.empty()
                live.markdown(pipeline_html(running=True,
                                            caption="Investigating…"),
                              unsafe_allow_html=True)
                try:
                    res = investigate(st.session_state.question,
                                      clarifications=clar, ask=False)
                    res["_elapsed"] = time.time() - t0
                    st.session_state.result = res
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
                live.empty()
                st.session_state.pending = []
                st.rerun()

    # ---- result
    state = st.session_state.result
    if state:
        results = state.get("results", [])
        failed = [r for r in results if not r.get("ok")]
        grounding = state.get("grounding", {})

        st.markdown("### Answer")
        if failed:
            st.warning(f"{len(failed)} of {len(results)} steps failed — this "
                       f"answer rests on partial evidence.")
        try:
            box = st.container(key="answer")
        except TypeError:                      # Streamlit older than 1.39
            box = st.container()
        with box:
            st.markdown(state.get("answer") or "_No answer produced._")

        if state.get("clarifications"):
            st.caption("Applied: " + "; ".join(
                f"**{k2}** → {v}" for k2, v in state["clarifications"].items()))

        QT = {"LOOKUP": "Lookup", "DIAGNOSTIC": "Diagnostic",
              "PREDICTIVE": "Predictive"}
        clean = grounding.get("clean")
        took = (f"{state['_elapsed']:.0f}s" if state.get("_elapsed")
                else ("Cached" if state.get("cached") else "—"))
        tile = lambda v, k, cls="": (f'<div class="stat"><div class="v {cls}">'
                                     f'{v}</div><div class="k">{k}</div></div>')
        st.markdown(
            '<div class="stats res">'
            + tile(QT.get(state.get("qtype") or "", state.get("qtype") or "—"), "Question type")
            + tile(len(results), "Steps")
            + tile(sum(r.get("attempts", 1) - 1 for r in results), "Self-corrections")
            + tile("Clean" if clean else "Flagged", "Numbers verified",
                   "good" if clean else "bad")
            + tile(took, "Time")
            + "</div>", unsafe_allow_html=True)

        chk = state.get("checklist") or {}
        LABELS = {"segment_localized": "cause localized to a segment",
                  "mechanism_identified": "mechanism identified",
                  "signal_found": "predictor found",
                  "baseline_compared": "compared against a baseline"}
        if state.get("qtype") != "LOOKUP" and chk:
            st.caption(" · ".join(
                f"{'✅' if v else '⚠️'} {LABELS.get(k, k)}"
                for k, v in chk.items()))

        st.markdown("### Evidence")
        st.caption("Every step shows the SQL that produced it — verify the "
                   "agent rather than trusting it.")
        for i, r in enumerate(results, 1):
            badge = "✅" if r.get("ok") else "❌"
            with st.expander(f"{badge} Step {i} — {r['step']}",
                             expanded=(i == 1)):
                if r.get("ok"):
                    try:
                        df = sql_df(r["sql"], f"{i}-{hash(r['sql'])}")
                        fig = pick_chart(df)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True,
                                            config={"displayModeBar": False})
                        # height follows the rows - a fixed 240px drew four
                        # empty rows under a two-row headline
                        st.dataframe(df, use_container_width=True, hide_index=True,
                                     height=min(36 * (len(df) + 1) + 4, 380))
                    except Exception as e:
                        st.warning(f"Could not re-render: {e}")
                else:
                    st.error("\n".join(r.get("errors", ["failed"])[-1:]))

                if r.get("attempts", 1) > 1:
                    st.caption(
                        f"Self-corrected after {r['attempts'] - 1} rejected "
                        f"attempt(s); {r.get('validator_catches', 0)} caught "
                        f"before execution.")
                if r.get("sql"):
                    st.code(r["sql"], language="sql")
                    b1, b2, _ = st.columns([1, 1, 3])
                    with b1:
                        copy_button(r["sql"], key=f"cp_{i}")
                    with b2:
                        st.download_button("Download .sql", r["sql"],
                                           file_name=f"step_{i}.sql",
                                           mime="text/plain", key=f"dl_{i}")

        with st.expander("Run details"):
            st.json({
                "question_type": state.get("qtype"),
                "revisions": state.get("revisions"),
                "checklist": chk,
                "grounding": grounding,
                "cached": state.get("cached") or False,
                "llm_calls": USAGE["calls"],
                "tokens": USAGE["prompt_tokens"] + USAGE["output_tokens"],
                "models": dict(USAGE["by_model"]),
                "providers": budget_status(),
            })
    footer()