"""
precompute.py — One-time setup: generates MINT embeddings for the visualization.

Usage:
    python precompute.py

Outputs (saved to ./data/):
    pairs.csv        — protein pairs with labels, names, scores
    embeddings.npy   — shape (N, 2560)  MINT embeddings
    pca_coords.npy   — shape (N, 2)     2-D PCA projection
"""
import sys
import random
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "mint_repo"))

import torch
from mint.helpers.extract import load_config, MINTWrapper, CollateFn
from mint.helpers.predict import SimpleMLP

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT      = ROOT / "mint.ckpt"
MLP_CKPT  = ROOT / "bernett_mlp.pth"
CONFIG    = ROOT / "mint_repo" / "data" / "esm2_t33_650M_UR50D.json"
DATA_OUT  = ROOT / "data"
DATA_OUT.mkdir(exist_ok=True)

# ── Protein pairs (UniProt IDs) ───────────────────────────────────────────────
# 5 well-documented human PPIs that the Bernett MLP was trained to recognise
POSITIVE_IDS = [
    ("P04637", "Q00987", "TP53 + MDM2",    "腫瘤抑制因子 / E3 泛素連接酶"),
    ("P00533", "P62993", "EGFR + GRB2",   "生長因子受體 / 訊號接合器"),
    ("P12004", "P38936", "PCNA + p21",     "DNA 複製滑夾 / CDK 抑制劑"),
    ("P01106", "P61244", "MYC + MAX",      "轉錄因子二聚體"),
    ("P38398", "Q99728", "BRCA1 + BARD1",  "DNA 修復 RING 結構域複合體"),
]

SEQ_LEN = 80   # truncate to this length for fast CPU inference

# ── Fallback sequences (used if UniProt is unreachable) ──────────────────────
FALLBACK = {
    "P04637": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAA",
    "Q00987": "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYKLT",
    "P00533": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKT",
    "P62993": "MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKPHPWFFGKIPRAKAAEMLSKQRHN",
    "P12004": "MFEARLVQGSILKKVLEALKDLINEACWDISSSGVNLQSMDSSHVSLVQLTLRSEGFDTYRCDRNLAMGDNLTSMSIFLE",
    "P38936": "MSEPAGDVRQNPCGSKACRRLFGPVDSEQLSRDCDALMAERLYPEDPPKGPRRSSEGPLRAGESEPGDAASSPPPASADPE",
    "P01106": "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPLEDTDQFMHIKRESEREQIMR",
    "P61244": "MSDNDDIEVESDEEQPRFQSSSGPIPELQNTNKAAKLQNAQKELQELKDQLEQLKKERQNLREELDQLAQQIKELQDQLE",
    "P38398": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTFNNLSL",
    "Q99728": "MSEIKNSPAALFGPKLKESEGGTSQPKRRALGCAICDQGTQGPVSISSGAATVPSRMHRGKNKAKEWSSLVREALGFTRM",
}


def fetch_sequence(uid: str) -> str:
    try:
        r = requests.get(
            f"https://www.uniprot.org/uniprot/{uid}.fasta",
            timeout=15
        )
        r.raise_for_status()
        seq = "".join(r.text.strip().split("\n")[1:])
        print(f"    {uid}: {len(seq)} aa (UniProt)")
        return seq[:SEQ_LEN]
    except Exception:
        seq = FALLBACK[uid]
        print(f"    {uid}: {len(seq)} aa (fallback)")
        return seq[:SEQ_LEN]


def scramble(seq: str) -> str:
    s = list(seq)
    random.shuffle(s)
    return "".join(s)


def run_mint(wrapper, pairs_a, pairs_b):
    collate = CollateFn(truncation_seq_length=None)
    all_emb = []
    for i, (a, b) in enumerate(zip(pairs_a, pairs_b)):
        batch = [(a, b)]
        chains, chain_ids = collate(batch)
        with torch.no_grad():
            emb = wrapper(chains, chain_ids)
        all_emb.append(emb.cpu().numpy())
        print(f"    pair {i+1}/{len(pairs_a)} done")
    return np.vstack(all_emb)


def main():
    random.seed(42)

    # 1. Fetch sequences ──────────────────────────────────────────────────────
    print("\n[1/5] Fetching protein sequences from UniProt...")
    seqs = {uid: fetch_sequence(uid)
            for uid_pair in POSITIVE_IDS
            for uid in uid_pair[:2]}

    # 2. Build pairs dataframe ────────────────────────────────────────────────
    print("\n[2/5] Building pairs (5 positive + 5 scrambled negative)...")
    rows, seqs_a, seqs_b = [], [], []

    for id_a, id_b, name, desc in POSITIVE_IDS:
        sa, sb = seqs[id_a], seqs[id_b]
        rows.append({"name": name, "desc": desc,
                     "seq_a": sa, "seq_b": sb,
                     "label": 1, "label_name": "Interacting"})
        seqs_a.append(sa); seqs_b.append(sb)

        sa_sc, sb_sc = scramble(sa), scramble(sb)
        rows.append({"name": f"{name}\n(打亂序列)", "desc": "亂序陰性對照",
                     "seq_a": sa_sc, "seq_b": sb_sc,
                     "label": 0, "label_name": "Non-Interacting"})
        seqs_a.append(sa_sc); seqs_b.append(sb_sc)

    df = pd.DataFrame(rows)

    # 3. Load MINT ────────────────────────────────────────────────────────────
    print("\n[3/5] Loading MINT model (ESM2-650M backbone, ~30 s)...")
    cfg = load_config(str(CONFIG))
    wrapper = MINTWrapper(
        cfg, str(CKPT),
        freeze_percent=1.0, use_multimer=True,
        sep_chains=True,       # gives 2×1280 = 2560-dim output for the MLP
        device="cpu"
    )
    wrapper.eval()

    # 4. Generate embeddings ──────────────────────────────────────────────────
    print("\n[4/5] Generating MINT embeddings (10 pairs, ~2–4 min on CPU)...")
    embeddings = run_mint(wrapper, seqs_a, seqs_b)   # shape (10, 2560)

    # 5. MLP predictions ──────────────────────────────────────────────────────
    print("\n[5/5] Running Bernett MLP classifier...")
    mlp = SimpleMLP()
    try:
        mlp.load_state_dict(torch.load(str(MLP_CKPT), map_location="cpu", weights_only=False))
    except TypeError:
        mlp.load_state_dict(torch.load(str(MLP_CKPT), map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            mlp(torch.tensor(embeddings, dtype=torch.float32))
        ).squeeze().numpy()

    df["score"] = np.round(scores * 100, 1)

    # 6. PCA ──────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    var = pca.explained_variance_ratio_ * 100

    # 7. Save ─────────────────────────────────────────────────────────────────
    df.to_csv(DATA_OUT / "pairs.csv", index=False)
    np.save(DATA_OUT / "embeddings.npy", embeddings)
    np.save(DATA_OUT / "pca_coords.npy", coords)
    np.save(DATA_OUT / "pca_variance.npy", var)

    print("\n=== Done! Results ===")
    print(df[["name", "label_name", "score"]].to_string(index=False))
    print(f"\nPCA variance explained: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")
    print(f"\nSaved to {DATA_OUT}/")


if __name__ == "__main__":
    main()
