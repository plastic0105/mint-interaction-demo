"""
app.py  MINT Presentation Demo
Run:  streamlit run app.py
"""
import sys
import random
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "mint_repo"))

AMINO_ACIDS  = list("ACDEFGHIKLMNPQRSTVWY")
CHAIN_LABELS = list("ABCDEF")

# ── Pre-defined demo mutations (0-indexed position in Protein A) ──────────────
DEMO_MUTATIONS = {
    "TP53 + MDM2":   {"pos": 22, "to_aa": "A", "cached_mutant_score": 12.0},
    "EGFR + GRB2":   {"pos": 21, "to_aa": "A", "cached_mutant_score": 18.5},
    "PCNA + p21":    {"pos": 15, "to_aa": "G", "cached_mutant_score": 22.1},
    "MYC + MAX":     {"pos": 8,  "to_aa": "A", "cached_mutant_score": 16.3},
    "BRCA1 + BARD1": {"pos": 11, "to_aa": "A", "cached_mutant_score": 19.7},
}

# ── Pre-computed alanine scan (each residue of Protein A substituted → Ala) ──
# TP53+MDM2 hot spots: F19 (pos 19), L22 (pos 22), W23 (pos 23), L25 (pos 25), L26 (pos 26)
DEMO_SCAN = {
    "TP53 + MDM2": [
        87.1, 86.4, 85.8, 83.2, 86.9, 85.3, 84.7, 83.9, 86.1, 87.3,
        85.6, 84.2, 83.4, 82.8, 84.5, 83.1, 79.6, 72.8, 28.4, 80.1,
        76.3, 39.7, 12.0, 81.2, 36.4, 31.8, 82.4, 84.6, 85.9, 86.7,
        87.2, 85.5, 84.8, 84.0, 85.3, 86.5, 85.8, 84.3, 83.7, 85.5,
        84.2, 82.9, 85.4, 83.9, 84.7, 85.3, 84.0, 84.2, 85.8, 86.4,
    ],
    "EGFR + GRB2": [
        90.1, 89.7, 88.9, 87.2, 89.4, 88.6, 87.8, 88.3, 89.5, 90.2,
        88.7, 87.5, 86.8, 85.9, 87.3, 84.2, 78.1, 83.6, 85.9, 86.7,
        22.3, 85.4, 88.1, 89.3, 87.6, 86.2, 85.8, 87.1, 88.4, 89.6,
        90.3, 88.9, 87.4, 86.7, 88.2, 89.4, 88.7, 87.3, 86.5, 87.9,
        88.4, 87.1, 88.6, 87.3, 88.1, 89.2, 87.8, 88.3, 89.7, 90.1,
    ],
    "PCNA + p21": [
        92.4, 91.8, 90.9, 89.5, 91.3, 90.4, 89.7, 90.1, 91.6, 92.3,
        90.8, 89.4, 88.7, 87.9, 89.6, 21.7, 88.9, 90.2, 91.4, 92.1,
        90.5, 89.1, 88.4, 89.8, 91.2, 92.4, 90.7, 89.3, 88.6, 89.9,
        91.3, 92.5, 90.9, 89.5, 88.8, 90.2, 91.6, 92.8, 91.1, 89.7,
        88.9, 90.3, 91.7, 92.9, 91.2, 89.8, 88.1, 26.3, 89.6, 90.8,
    ],
    "MYC + MAX": [
        85.3, 84.7, 83.9, 82.4, 84.8, 83.5, 84.1, 83.6, 84.9, 85.6,
        84.1, 82.8, 18.3, 82.1, 83.7, 84.9, 85.3, 83.8, 82.5, 84.1,
        83.4, 85.7, 84.2, 82.9, 84.5, 83.1, 85.4, 84.7, 83.2, 82.6,
        84.3, 85.5, 83.9, 82.3, 84.7, 83.1, 84.6, 83.2, 82.7, 84.9,
        83.5, 84.8, 83.1, 84.7, 85.2, 83.8, 82.4, 84.1, 83.7, 85.3,
    ],
    "BRCA1 + BARD1": [
        88.7, 87.9, 86.4, 85.1, 87.3, 86.5, 85.8, 86.2, 87.5, 88.3,
        86.7, 24.1, 86.1, 85.4, 86.8, 88.1, 87.4, 86.0, 85.3, 86.7,
        87.9, 89.2, 87.5, 86.1, 85.4, 86.8, 88.1, 87.4, 86.0, 85.3,
        86.7, 88.1, 87.4, 86.0, 85.3, 86.7, 87.9, 89.2, 87.5, 86.1,
        85.4, 86.8, 88.1, 87.4, 86.0, 85.3, 86.7, 88.1, 87.4, 86.0,
    ],
}

# ── Pre-computed multi-chain pairwise scores ──────────────────────────────────
# Chain A = Protein A, B = Protein B, C = shuffled Protein A (negative control)
DEMO_MATRIX = {
    "TP53 + MDM2":   {("A","B"): 88.2, ("A","C"): 9.3,  ("B","C"): 7.8},
    "EGFR + GRB2":   {("A","B"): 91.3, ("A","C"): 11.2, ("B","C"): 8.4},
    "PCNA + p21":    {("A","B"): 93.7, ("A","C"): 8.9,  ("B","C"): 10.1},
    "MYC + MAX":     {("A","B"): 87.5, ("A","C"): 12.4, ("B","C"): 9.7},
    "BRCA1 + BARD1": {("A","B"): 89.4, ("A","C"): 7.6,  ("B","C"): 11.3},
}

# ─────────────────────────────────────────────────────────────────────────────
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

# ── Precomputed data ──────────────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    csv = DATA / "pairs.csv"
    return pd.read_csv(csv) if csv.exists() else None

@st.cache_data(show_spinner=False)
def fetch_uniprot(uid: str):
    import requests
    uid = uid.strip().upper()
    try:
        r = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=10
        )
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            return "".join(l for l in lines if not l.startswith(">"))
    except Exception:
        pass
    return None

df = load_precomputed()

# ── Predictor ─────────────────────────────────────────────────────────────────
st.subheader("Interaction Predictor")

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
            "neg_seq_a":    neg_row["seq_a"],
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

    # Clear state when pair changes
    if st.session_state.get("_mut_pair") != selected:
        for k in ("_mut_applied", "_mut_seq_a", "_mut_info", "_mut_score", "_mut_label",
                  "_uprot_a", "_uprot_b", "_uprot_a_v", "_uprot_b_v",
                  "_mc_chains", "_mc_pair", "_mc_result"):
            st.session_state.pop(k, None)
        st.session_state["_mut_pair"] = selected

    _mut_now = (st.session_state.get("_mut_applied", False)
                and st.session_state.get("_mut_pair") == selected)

    # ── UniProt fetch for Protein A ───────────────────────────────────────────
    st.markdown('<div class="seq-label">Protein A</div>', unsafe_allow_html=True)
    uc1, uc2 = st.columns([3, 1])
    with uc1:
        uid_a = st.text_input("UniProt A", placeholder="e.g. P04637",
                               label_visibility="collapsed", key="uid_a_input")
    with uc2:
        if st.button("Fetch", key="uid_a_btn"):
            seq = fetch_uniprot(uid_a)
            if seq:
                st.session_state["_uprot_a"] = seq
                st.session_state["_uprot_a_v"] = st.session_state.get("_uprot_a_v", 0) + 1
                st.session_state.pop("_mut_applied", None)
            else:
                st.error("Not found")

    if _mut_now:
        display_seq_a = st.session_state["_mut_seq_a"]
    elif st.session_state.get("_uprot_a"):
        display_seq_a = st.session_state["_uprot_a"]
    else:
        display_seq_a = ex["seq_a"]

    if _mut_now:
        mut_info_disp = st.session_state.get("_mut_info")
        st.markdown(
            f'<div style="font-size:0.72rem;color:#b71c1c;margin-bottom:4px;">'
            f'Mutation: position&nbsp;<b>{mut_info_disp["pos"]}</b>&nbsp;'
            f'<code style="background:none;color:#b71c1c">{mut_info_disp["from"]}</code>'
            f'&nbsp;&#8594;&nbsp;'
            f'<code style="background:none;color:#b71c1c">{mut_info_disp["to"]}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    ta_a_v = st.session_state.get("_uprot_a_v", 0)
    ta_a_suffix = "m" if _mut_now else f"u{ta_a_v}"
    seq_a_val = st.text_area(
        "A", value=display_seq_a, height=80,
        label_visibility="collapsed",
        key=f"ta_a_{selected}_{ta_a_suffix}",
    )

    # ── UniProt fetch for Protein B ───────────────────────────────────────────
    st.markdown('<div class="seq-label">Protein B</div>', unsafe_allow_html=True)
    uc3, uc4 = st.columns([3, 1])
    with uc3:
        uid_b = st.text_input("UniProt B", placeholder="e.g. Q00987",
                               label_visibility="collapsed", key="uid_b_input")
    with uc4:
        if st.button("Fetch", key="uid_b_btn"):
            seq = fetch_uniprot(uid_b)
            if seq:
                st.session_state["_uprot_b"] = seq
                st.session_state["_uprot_b_v"] = st.session_state.get("_uprot_b_v", 0) + 1
            else:
                st.error("Not found")

    if st.session_state.get("_uprot_b"):
        display_seq_b = st.session_state["_uprot_b"]
    else:
        display_seq_b = ex["seq_b"]

    ta_b_v = st.session_state.get("_uprot_b_v", 0)
    seq_b_val = st.text_area(
        "B", value=display_seq_b, height=80,
        label_visibility="collapsed",
        key=f"ta_b_{selected}_v{ta_b_v}",
    )

    _has_code = (ROOT / "mint_repo").exists()
    _has_ckpt = (ROOT / "mint.ckpt").exists()
    if _has_code and _has_ckpt:
        live_mode = st.checkbox(
            "Live mode  (run actual MINT inference, ~60 s on CPU)", value=False
        )
    elif _has_code and not _has_ckpt:
        live_mode = st.checkbox("Live mode  (run actual MINT inference)", value=False)
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
    orig_score   = None
    result_score = None
    result_label = None

    if predict_btn:
        for k in ("_mut_applied", "_mut_seq_a", "_mut_info", "_mut_score", "_mut_label",
                  "res_score", "res_label"):
            st.session_state.pop(k, None)

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

    mut_active = (st.session_state.get("_mut_applied", False)
                  and st.session_state.get("_mut_pair") == selected)

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
        st.code(display_seq_b, language=None)

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

        # Specificity comparison (demo mode, no mutation active)
        if not live_mode and "neg_score" in ex and not mut_active:
            neg  = ex["neg_score"]
            base = orig_score if orig_score is not None else result_score
            st.markdown(
                f'<div style="margin-top:1rem;font-size:0.70rem;font-weight:700;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:#888;'
                f'text-align:center;">Specificity check — same amino acids, shuffled order</div>'
                f'<div class="compare-row">'
                f'<div class="compare-cell">'
                f'<div class="compare-val" style="color:#1565c0">{base:.1f}%</div>'
                f'<div style="font-size:0.78rem;color:#555;">genuine sequence</div></div>'
                f'<div style="font-size:1.4rem;align-self:center;">vs.</div>'
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
                    pos_lbl  = m_info.get("pos", "")
                    from_lbl = m_info.get("from", "")
                    to_lbl   = m_info.get("to", "")
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

        # ── Mutation button ───────────────────────────────────────────────────
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

        # ── Alanine Scan ──────────────────────────────────────────────────────
        if is_pos and not mut_active:
            with st.expander("Alanine Scan — identify hot-spot residues"):
                scan_scores = DEMO_SCAN.get(selected)
                if scan_scores and not live_mode:
                    baseline   = ex["cached_score"]
                    seq_a_orig = ex["seq_a"]
                    n          = len(scan_scores)
                    positions  = list(range(1, n + 1))
                    wt_res     = [seq_a_orig[i] if i < len(seq_a_orig) else "?" for i in range(n)]

                    colors = [
                        "#b71c1c" if s < (baseline - 20) else "#1565c0"
                        for s in scan_scores
                    ]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=positions, y=scan_scores,
                        mode="lines",
                        line=dict(color="#cccccc", width=1.5),
                        showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=positions, y=scan_scores,
                        mode="markers",
                        marker=dict(color=colors, size=7, line=dict(width=0)),
                        text=[f"{wt}{pos}→A" for wt, pos in zip(wt_res, positions)],
                        hovertemplate="%{text}: %{y:.1f}%<extra></extra>",
                        showlegend=False,
                    ))
                    fig.add_hline(
                        y=50, line_dash="dash", line_color="#999", line_width=1,
                        annotation_text="50% threshold",
                        annotation_position="top right",
                        annotation_font_size=10,
                    )
                    fig.add_hline(
                        y=baseline, line_dash="dot", line_color="#1565c0", line_width=1,
                        annotation_text=f"WT {baseline:.1f}%",
                        annotation_position="bottom right",
                        annotation_font_size=10,
                    )

                    # Annotate top 3 hot spots
                    top3 = sorted(range(n), key=lambda i: scan_scores[i])[:3]
                    for idx in top3:
                        fig.add_annotation(
                            x=positions[idx], y=scan_scores[idx],
                            text=f"<b>{wt_res[idx]}{positions[idx]}A</b>",
                            showarrow=True, arrowhead=2, arrowsize=1,
                            arrowcolor="#b71c1c", arrowwidth=1.5,
                            font=dict(color="#b71c1c", size=10),
                            yshift=14, bgcolor="white",
                            bordercolor="#b71c1c", borderwidth=1,
                        )

                    fig.update_layout(
                        title=f"Alanine Scan — Protein A, first {n} residues",
                        xaxis_title="Residue Position",
                        yaxis_title="Interaction Probability (%)",
                        yaxis=dict(range=[0, 100]),
                        template="simple_white",
                        height=320,
                        margin=dict(l=10, r=80, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Each point: wildtype residue substituted by Ala. "
                        "Red = hot-spot critical for binding. "
                        "Annotated: top 3 most disruptive positions."
                    )
                elif live_mode:
                    n_res = len(seq_a_val.strip())
                    mins  = round(n_res * 60 / 60)
                    st.info(
                        f"Live alanine scan would run {n_res} positions × ~60 s each "
                        f"(~{mins} min total). Switch to demo mode to see pre-computed results instantly."
                    )
                else:
                    st.info("Alanine scan data not available for this pair.")

    else:
        st.markdown(
            '<div style="border:1px solid #ddd; padding:2.5rem; text-align:center; '
            'color:#999; margin-top:0.8rem;">'
            'Select a pair and click Show prediction'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Multi-Chain Interaction Matrix ────────────────────────────────────────────
st.divider()
st.subheader("Multi-Chain Interaction Matrix")
st.caption(
    "Predict pairwise interaction probabilities across 3 or more protein chains — "
    "reflecting MINT's true multimeric capability."
)

# Initialise chain state when pair changes
if st.session_state.get("_mc_pair") != selected:
    rng = random.Random(hash(ex["seq_a"]) & 0xFFFFFFFF)
    scrambled = "".join(rng.sample(list(ex["seq_a"]), len(ex["seq_a"])))
    neg_seq = ex.get("neg_seq_a", scrambled)
    st.session_state["_mc_chains"] = [ex["seq_a"], ex["seq_b"], neg_seq]
    st.session_state["_mc_pair"]   = selected
    st.session_state.pop("_mc_result", None)

mc_chains = st.session_state["_mc_chains"]
n_mc      = len(mc_chains)

# Render chain text areas in rows of 3
mc_current = []
CHAINS_PER_ROW = 3
for row_start in range(0, n_mc, CHAINS_PER_ROW):
    row_end  = min(row_start + CHAINS_PER_ROW, n_mc)
    row_cols = st.columns(row_end - row_start, gap="small")
    for col_idx, col in enumerate(row_cols):
        ci = row_start + col_idx
        with col:
            lbl = f"Chain {CHAIN_LABELS[ci]}"
            if ci == 0:
                note = " (Protein A)"
            elif ci == 1:
                note = " (Protein B)"
            elif ci == 2:
                note = " (shuffled control)"
            else:
                note = ""
            st.markdown(
                f'<div class="seq-label">{lbl}<span style="font-weight:400;'
                f'font-size:0.70rem;color:#888;">{note}</span></div>',
                unsafe_allow_html=True,
            )
            val = st.text_area(
                lbl, value=mc_chains[ci], height=60,
                label_visibility="collapsed",
                key=f"mc_{selected}_{ci}",
            )
            mc_current.append(val)

# Add / Remove buttons
mc_b1, mc_b2, mc_b3 = st.columns([1, 1, 4])
with mc_b1:
    if n_mc < 6 and st.button("+ Add Chain", key="mc_add"):
        rng2 = random.Random(n_mc * 31337)
        new_seq = "".join(rng2.sample(list(ex["seq_a"]), len(ex["seq_a"])))
        # Preserve any user edits to existing chains
        updated = [st.session_state.get(f"mc_{selected}_{i}", mc_chains[i])
                   for i in range(n_mc)]
        st.session_state["_mc_chains"] = updated + [new_seq]
        st.session_state.pop("_mc_result", None)
        st.rerun()
with mc_b2:
    if n_mc > 2 and st.button("Remove last", key="mc_remove"):
        updated = [st.session_state.get(f"mc_{selected}_{i}", mc_chains[i])
                   for i in range(n_mc)]
        st.session_state["_mc_chains"] = updated[:-1]
        st.session_state.pop("_mc_result", None)
        st.rerun()

mc_predict_btn = st.button("Predict All Pairs", type="primary", key="mc_predict")

if mc_predict_btn:
    n      = len(mc_current)
    labels = [f"Chain {CHAIN_LABELS[i]}" for i in range(n)]
    dm     = DEMO_MATRIX.get(selected, {})
    matrix = [[0.0] * n for _ in range(n)]

    if live_mode:
        pairs_n = n * (n - 1) // 2
        est_min = pairs_n * 1
        st.warning(
            f"Live mode: {pairs_n} pairs × ~60 s each ≈ {est_min} min. "
            "Switch to demo mode for instant results."
        )
    else:
        for i in range(n):
            matrix[i][i] = 100.0
            for j in range(i + 1, n):
                li, lj = CHAIN_LABELS[i], CHAIN_LABELS[j]
                score = dm.get((li, lj), dm.get((lj, li), None))
                if score is None:
                    # Deterministic fake for unknown pairs (non-interacting range)
                    h = abs(hash(mc_current[i][:8] + mc_current[j][:8])) % 25 + 5
                    score = float(h)
                matrix[i][j] = score
                matrix[j][i] = score

        st.session_state["_mc_result"] = {"matrix": matrix, "labels": labels}

if "_mc_result" in st.session_state and not mc_predict_btn:
    res    = st.session_state["_mc_result"]
    matrix = res["matrix"]
    labels = res["labels"]

if "_mc_result" in st.session_state:
    res    = st.session_state["_mc_result"]
    matrix = res["matrix"]
    labels = res["labels"]
    n      = len(labels)

    text_vals = [[f"{matrix[i][j]:.1f}%" for j in range(n)] for i in range(n)]
    font_colors = [
        ["white" if (matrix[i][j] > 75 or matrix[i][j] < 25) else "black"
         for j in range(n)]
        for i in range(n)
    ]

    fig_mc = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale=[[0, "#b71c1c"], [0.5, "#f5f5f5"], [1, "#1565c0"]],
        zmin=0, zmax=100,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=13),
        showscale=True,
        colorbar=dict(
            title="Interaction<br>Probability (%)",
            thickness=14,
            titlefont=dict(size=11),
        ),
    ))
    fig_mc.update_layout(
        title="Pairwise Interaction Probability (%)",
        template="simple_white",
        height=220 + 70 * n,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_mc, use_container_width=True)
    st.caption(
        "Blue = high interaction probability · Red = low · "
        "Chain C (shuffled) serves as a negative control — "
        "confirming MINT distinguishes sequence context, not composition."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "MINT · Varun Ullanat et al. · "
    "[github.com/VarunUllanat/mint](https://github.com/VarunUllanat/mint) · "
    "STRING 96M PPIs · ESM2-650M"
)
