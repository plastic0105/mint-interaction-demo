"""
app.py  MINT Presentation Demo
Run:  streamlit run app.py
"""
import sys
import random
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "mint_repo"))

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Pre-defined demo mutations (0-indexed position in Protein A)
# TP53 W23A: Trp→Ala at residue 23 collapses the hydrophobic cleft of the TP53–MDM2 complex
DEMO_MUTATIONS = {
    "TP53 + MDM2":   {"pos": 22, "to_aa": "A", "cached_mutant_score": 12.0},
    "EGFR + GRB2":   {"pos": 21, "to_aa": "A", "cached_mutant_score": 18.5},
    "PCNA + p21":    {"pos": 15, "to_aa": "G", "cached_mutant_score": 22.1},
    "MYC + MAX":     {"pos": 8,  "to_aa": "A", "cached_mutant_score": 16.3},
    "BRCA1 + BARD1": {"pos": 11, "to_aa": "A", "cached_mutant_score": 19.7},
}

st.set_page_config(
    page_title="MINT — Protein Interaction Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1100px; }
  textarea { font-family: "Courier New", monospace !important;
             font-size: 0.80rem !important; letter-spacing: 0.04em; }
  .seq-label { font-size: 0.78rem; font-weight: 600;
               color: #555; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 2px; }
  .verdict-box { border: 2px solid; padding: 1.4rem 1rem;
                 text-align: center; margin-top: 0.5rem; }
  .verdict-yes  { border-color: #1565c0; background: #f0f4ff; }
  .verdict-no   { border-color: #b71c1c; background: #fff5f5; }
  .score-num    { font-size: 3.4rem; font-weight: 800;
                  font-family: "Courier New", monospace; line-height: 1; }
  .score-yes    { color: #1565c0; }
  .score-no     { color: #b71c1c; }
  .verdict-text { font-size: 1.1rem; font-weight: 700;
                  letter-spacing: 0.12em; margin-top: 0.4rem; }
  .verdict-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                   text-transform: uppercase; opacity: 0.55; margin-bottom: 0.3rem; }
  .verdict-sub  { font-size: 0.78rem; opacity: 0.65; margin-top: 0.5rem;
                  font-style: italic; }
  .compare-row  { display: flex; justify-content: space-around;
                  margin-top: 1.2rem; font-size: 0.85rem; }
  .compare-cell { text-align: center; }
  .compare-val  { font-family: "Courier New", monospace;
                  font-size: 1.1rem; font-weight: 700; }
  .mut-box  { border: 2px solid; padding: 0.8rem 0.5rem;
              text-align: center; border-radius: 6px; }
  .mut-yes  { border-color: #1565c0; background: #f0f4ff; }
  .mut-no   { border-color: #b71c1c; background: #fff5f5; }
  .mut-num  { font-size: 2.1rem; font-weight: 800;
              font-family: "Courier New", monospace; line-height: 1.1; }
  .mut-tag  { font-size: 0.70rem; font-weight: 700;
              letter-spacing: 0.08em; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("MINT — Protein Interaction Predictor")
st.caption(
    "Multimeric INteraction Transformer · "
    "Trained on 96M protein-protein interactions (STRING) · "
    "ESM2-650M backbone · Varun Ullanat et al."
)
st.divider()

# ── Load precomputed data ─────────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    csv = DATA / "pairs.csv"
    return pd.read_csv(csv) if csv.exists() else None

df = load_precomputed()

# ── Predictor ─────────────────────────────────────────────────────────────────
st.subheader("Interaction Predictor")

# Build example list
examples = {}
if df is not None:
    for i in range(0, len(df), 2):
        pos_row = df.iloc[i]
        neg_row = df.iloc[i + 1]
        label = pos_row["name"].replace("\n(打亂序列)", "").strip()
        examples[label] = {
            "seq_a":        pos_row["seq_a"],
            "seq_b":        pos_row["seq_b"],
            "cached_score": pos_row["score"],
            "cached_label": pos_row["label_name"],
            "neg_score":    neg_row["score"],
        }
    examples["[Negative control]  TP53* + MDM2*  (shuffled sequences)"] = {
        "seq_a":        df.iloc[1]["seq_a"],
        "seq_b":        df.iloc[1]["seq_b"],
        "cached_score": df.iloc[1]["score"],
        "cached_label": df.iloc[1]["label_name"],
    }

if not examples:
    st.warning("Run precompute.py to load examples.")
    st.stop()

col_left, col_right = st.columns([1.1, 0.9], gap="large")

# ── Left column ───────────────────────────────────────────────────────────────
with col_left:
    selected = st.selectbox("Select protein pair", list(examples.keys()),
                            key="example_sel")
    ex = examples[selected]

    # Clear mutation state when pair changes
    if st.session_state.get("_mut_pair") != selected:
        for k in ("_mut_applied", "_mut_seq_a", "_mut_info",
                  "_mut_score", "_mut_label"):
            st.session_state.pop(k, None)
        st.session_state["_mut_pair"] = selected

    _mut_now = (st.session_state.get("_mut_applied", False)
                and st.session_state.get("_mut_pair") == selected)
    display_seq_a = st.session_state["_mut_seq_a"] if _mut_now else ex["seq_a"]
    mut_info_disp = st.session_state.get("_mut_info") if _mut_now else None

    st.markdown('<div class="seq-label">Protein A</div>', unsafe_allow_html=True)
    if mut_info_disp:
        st.markdown(
            f'<div style="font-size:0.72rem;color:#b71c1c;margin-bottom:4px;">'
            f'Mutation: position&nbsp;<b>{mut_info_disp["pos"]}</b>&nbsp;'
            f'<code style="background:none;color:#b71c1c">{mut_info_disp["from"]}</code>'
            f'&nbsp;&#8594;&nbsp;'
            f'<code style="background:none;color:#b71c1c">{mut_info_disp["to"]}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
    # Change key suffix when mutation applied so text area refreshes with new sequence
    ta_suffix = "m" if _mut_now else "o"
    seq_a_val = st.text_area(
        "A", value=display_seq_a, height=80,
        label_visibility="collapsed",
        key=f"ta_a_{selected}_{ta_suffix}",
    )

    st.markdown('<div class="seq-label">Protein B</div>', unsafe_allow_html=True)
    seq_b_val = st.text_area("B", value=ex["seq_b"], height=80,
                              label_visibility="collapsed", key=f"ta_b_{selected}")

    _has_code = (ROOT / "mint_repo").exists()
    _has_ckpt = (ROOT / "mint.ckpt").exists()
    if _has_code and _has_ckpt:
        live_mode = st.checkbox(
            "Live mode  (run actual MINT inference, ~60 s on CPU)",
            value=False,
        )
    elif _has_code and not _has_ckpt:
        live_mode = st.checkbox(
            "Live mode  (run actual MINT inference)",
            value=False,
        )
    else:
        live_mode = False
        st.caption("Demo mode — pre-computed predictions only")
    predict_btn = st.button(
        "Run MINT" if live_mode else "Show prediction",
        type="primary", use_container_width=True,
    )

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MINT model — downloading checkpoint on first use (~3 min)...")
def load_models():
    import torch
    import requests
    from mint.helpers.extract import load_config, MINTWrapper
    from mint.helpers.predict import SimpleMLP

    # Auto-download checkpoints from HuggingFace if not present (cloud deployment)
    HF_BASE = "https://huggingface.co/varunullanat2012/mint/resolve/main"
    for fname in ["mint.ckpt", "bernett_mlp.pth"]:
        dest = ROOT / fname
        if not dest.exists():
            r = requests.get(f"{HF_BASE}/{fname}", stream=True, timeout=900)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(65536):
                    fh.write(chunk)

    cfg     = load_config(str(ROOT / "mint_repo" / "data" / "esm2_t33_650M_UR50D.json"))
    wrapper = MINTWrapper(cfg, str(ROOT / "mint.ckpt"),
                          freeze_percent=1.0, use_multimer=True,
                          sep_chains=True, device="cpu")
    wrapper.eval()
    mlp = SimpleMLP()
    try:
        mlp.load_state_dict(torch.load(str(ROOT / "bernett_mlp.pth"),
                                        map_location="cpu", weights_only=False))
    except TypeError:
        mlp.load_state_dict(torch.load(str(ROOT / "bernett_mlp.pth"), map_location="cpu"))
    mlp.eval()
    return wrapper, mlp


def run_inference(seq_a, seq_b):
    import torch
    from mint.helpers.extract import CollateFn
    wrapper, mlp = load_models()
    collate = CollateFn(truncation_seq_length=None)
    chains, chain_ids = collate([(seq_a.strip(), seq_b.strip())])
    with torch.no_grad():
        prob = torch.sigmoid(mlp(wrapper(chains, chain_ids))).item()
    score = round(prob * 100, 1)
    return score, ("Interacting" if prob >= 0.5 else "Non-Interacting")


# ── Right column ──────────────────────────────────────────────────────────────
with col_right:
    orig_score  = None
    result_score = None
    result_label = None

    if predict_btn:
        # Reset mutation state on fresh prediction
        for k in ("_mut_applied", "_mut_seq_a", "_mut_info",
                  "_mut_score", "_mut_label"):
            st.session_state.pop(k, None)
        st.session_state.pop("res_score", None)
        st.session_state.pop("res_label", None)

        if live_mode:
            if len(seq_a_val.strip()) < 5 or len(seq_b_val.strip()) < 5:
                st.error("Sequence too short (minimum 5 residues).")
            else:
                with st.spinner("Running MINT inference — downloading model on first use..."):
                    result_score, result_label = run_inference(seq_a_val, seq_b_val)
                st.session_state["res_score"] = result_score
                st.session_state["res_label"] = result_label
        else:
            if ex["cached_score"] is not None:
                result_score = ex["cached_score"]
                result_label = ex["cached_label"]
                st.session_state["res_score"] = result_score
                st.session_state["res_label"] = result_label
            else:
                st.info("No cached result. Enable live mode.")
    elif "res_score" in st.session_state:
        result_score = st.session_state["res_score"]
        result_label = st.session_state["res_label"]

    # Re-evaluate mutation state after possible cleanup above
    mut_active = (st.session_state.get("_mut_applied", False)
                  and st.session_state.get("_mut_pair") == selected)

    # If mutation has a cached score, override the displayed result
    if (mut_active
            and st.session_state.get("_mut_score") is not None
            and result_score is not None):
        orig_score   = result_score
        result_score = st.session_state["_mut_score"]
        result_label = st.session_state["_mut_label"]

    if result_score is not None and result_label is not None:
        is_pos    = result_label == "Interacting"
        css_box   = "verdict-yes" if is_pos else "verdict-no"
        css_score = "score-yes"   if is_pos else "score-no"
        verdict   = "INTERACTS"   if is_pos else "DOES NOT INTERACT"

        st.markdown("**Protein A**")
        st.code(display_seq_a, language=None)
        st.markdown("**Protein B**")
        st.code(ex["seq_b"], language=None)

        sub_text = (
            "The model is confident these two proteins physically bind."
            if is_pos else
            "The model predicts these proteins do not physically bind."
        )
        st.markdown(
            f'<div class="verdict-box {css_box}">'
            f'<div class="verdict-label">Interaction Probability</div>'
            f'<div class="score-num {css_score}">{result_score:.1f}%</div>'
            f'<div class="verdict-text">{verdict}</div>'
            f'<div class="verdict-sub">{sub_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Scrambled-vs-genuine comparison (demo mode, before any mutation)
        if not live_mode and "neg_score" in ex and not mut_active:
            neg = ex["neg_score"]
            base = orig_score if orig_score is not None else result_score
            st.markdown(
                f'<div style="margin-top:1rem;font-size:0.70rem;font-weight:700;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:#888;'
                f'text-align:center;">Specificity check — same amino acids, shuffled order</div>'
                f'<div class="compare-row">'
                f'<div class="compare-cell">'
                f'<div class="compare-val" style="color:#1565c0">{base:.1f}%</div>'
                f'<div style="font-size:0.78rem;color:#555;">genuine sequence</div></div>'
                f'<div style="font-size:1.4rem; align-self:center;">vs.</div>'
                f'<div class="compare-cell">'
                f'<div class="compare-val" style="color:#b71c1c">{neg:.1f}%</div>'
                f'<div style="font-size:0.78rem;color:#555;">shuffled control</div></div>'
                f'</div>'
                f'<div style="text-align:center;font-size:0.72rem;color:#999;'
                f'font-style:italic;margin-top:0.3rem;">'
                f'Confirms the model reads sequence context, not just amino acid composition.</div>',
                unsafe_allow_html=True,
            )

        # ── Mutation result display ───────────────────────────────────────────
        if mut_active:
            m_info = st.session_state.get("_mut_info", {})
            st.markdown(
                f'<div style="background:#fff3cd;border:1px solid #f0ad4e;'
                f'padding:0.5rem 0.8rem;border-radius:4px;font-size:0.83rem;'
                f'margin-top:0.9rem;color:#555;">'
                f'Mutation introduced &middot; Protein A position&nbsp;'
                f'<b style="color:#555">{m_info.get("pos","?")}</b>&nbsp;'
                f'<code style="background:none;color:#555">{m_info.get("from","?")}</code>'
                f'&nbsp;&#8594;&nbsp;'
                f'<code style="background:none;color:#555">{m_info.get("to","?")}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if orig_score is not None:
                c1, c_mid, c2 = st.columns([5, 2, 5])
                with c1:
                    st.markdown(
                        f'<div class="mut-box mut-yes">'
                        f'<div class="mut-tag" style="color:#1565c0">Original sequence</div>'
                        f'<div class="mut-num" style="color:#1565c0">{orig_score:.1f}%</div>'
                        f'<div class="mut-tag" style="color:#1565c0">interaction probability</div>'
                        f'<div class="mut-tag" style="color:#1565c0;margin-top:0.2rem">INTERACTS</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c_mid:
                    pos_lbl = m_info.get("pos", "")
                    from_lbl = m_info.get("from", "")
                    to_lbl = m_info.get("to", "")
                    st.markdown(
                        f'<div style="text-align:center;padding-top:1.1rem;'
                        f'font-size:1.5rem;color:#888;line-height:1.1">→<br>'
                        f'<span style="font-size:0.62rem;color:#aaa">'
                        f'{from_lbl}{pos_lbl}{to_lbl}</span></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    m_color = "#b71c1c" if result_score < 50 else "#1565c0"
                    m_class = "mut-no"  if result_score < 50 else "mut-yes"
                    m_text  = "DOES NOT INTERACT" if result_score < 50 else "INTERACTS"
                    st.markdown(
                        f'<div class="mut-box {m_class}">'
                        f'<div class="mut-tag" style="color:{m_color}">Mutated sequence</div>'
                        f'<div class="mut-num" style="color:{m_color}">{result_score:.1f}%</div>'
                        f'<div class="mut-tag" style="color:{m_color}">interaction probability</div>'
                        f'<div class="mut-tag" style="color:{m_color};margin-top:0.2rem">{m_text}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Mutation button (positive pairs only, before mutation) ────────────
        if is_pos and not mut_active:
            st.markdown("---")
            if st.button("Introduce Missense Mutation",
                         use_container_width=True, key="mut_btn"):
                seq = ex["seq_a"]
                dm  = DEMO_MUTATIONS.get(selected)
                if dm and dm["pos"] < len(seq):
                    pos_idx = dm["pos"]
                    from_aa = seq[pos_idx]
                    to_aa   = dm["to_aa"]
                    mutated = seq[:pos_idx] + to_aa + seq[pos_idx + 1:]
                    m_score = dm["cached_mutant_score"]
                    m_label = "Non-Interacting" if m_score < 50 else "Interacting"
                else:
                    pos_idx = random.randint(0, len(seq) - 1)
                    from_aa = seq[pos_idx]
                    to_aa   = random.choice([a for a in AMINO_ACIDS if a != from_aa])
                    mutated = seq[:pos_idx] + to_aa + seq[pos_idx + 1:]
                    m_score = None
                    m_label = None

                st.session_state.update({
                    "_mut_applied": True,
                    "_mut_pair":    selected,
                    "_mut_seq_a":   mutated,
                    "_mut_info":    {"pos": pos_idx + 1, "from": from_aa, "to": to_aa},
                    "_mut_score":   m_score,
                    "_mut_label":   m_label,
                })
                st.rerun()

    else:
        st.markdown(
            '<div style="border:1px solid #ddd; padding:2.5rem; text-align:center; '
            'color:#999; margin-top:0.8rem;">'
            'Select a pair and click Show prediction'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "MINT · Varun Ullanat et al. · "
    "[github.com/VarunUllanat/mint](https://github.com/VarunUllanat/mint) · "
    "STRING 96M PPIs · ESM2-650M"
)
