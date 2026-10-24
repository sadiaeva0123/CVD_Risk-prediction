#!/usr/bin/env python3
# =============================================================================
# Step-by-step recalculation of the entropy / impurity results
# Reads the CSV files produced by tree_entropy_impurity.py and re-does every
# key number by hand, printing each equation with the actual values.
# Kaggle: add the tree_impurity output as input (or set FOLDER).
# =============================================================================
import glob, os
import numpy as np
import pandas as pd

FOLDER = "/kaggle/input"          # folder that contains the tree_impurity CSV files


def load(name):
    hits = sorted(glob.glob(os.path.join(FOLDER, "**", f"*{name}"), recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return pd.read_csv(hits[0])


def gini(p):                      # Gini impurity of a node with CVD share p
    return 1 - p**2 - (1 - p)**2


def entropy_bits(p):              # Shannon entropy in bits
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def line(t=""):
    print("\n" + "=" * 78 + ("\n" + t if t else ""))


# -----------------------------------------------------------------------------
line("1. CLASS GINI AND ENTROPY OF THE DATASETS  (class_entropy_and_gini_of_datasets.csv)")
ds = load("class_entropy_and_gini_of_datasets.csv")
for _, r in ds.iterrows():
    p = r.P_CVD
    print(f"\n{r.dataset}: {int(r.patients)} patients, P(CVD) = p = {p:.5f}")
    print(f"  Gini    = 1 - p^2 - (1-p)^2 = 1 - {p:.5f}^2 - {1-p:.5f}^2 = {gini(p):.5f}   (file: {r.gini:.5f})")
    print(f"  Entropy = -p*log2(p) - (1-p)*log2(1-p) = {entropy_bits(p):.5f} bits   (file: {r.entropy_bits:.5f})")
    print(f"  Entropy in nats = bits * ln(2) = {entropy_bits(p):.5f} * 0.69315 = {entropy_bits(p)*np.log(2):.5f}")

# -----------------------------------------------------------------------------
line("2. ONE SPLIT, BY HAND: root node of Random Forest tree 0  (random_forest_tree0_all_nodes.csv)")
rf = load("random_forest_tree0_all_nodes.csv").set_index("node_id")
root = rf.loc[0]
L, R = rf.loc[int(root.left_child)], rf.loc[int(root.right_child)]
N, nL, nR = root.samples_weighted, L.samples_weighted, R.samples_weighted
print(f"Split: {root.split_feature} <= {root.split_threshold}")
print(f"  parent: n={N:.0f}  p={root.P_CVD:.4f}  Gini={root.gini:.4f}  H={root.entropy_bits:.4f}")
print(f"  left  : n={nL:.0f}  p={L.P_CVD:.4f}  Gini={L.gini:.4f}  H={L.entropy_bits:.4f}")
print(f"  right : n={nR:.0f}  p={R.P_CVD:.4f}  Gini={R.gini:.4f}  H={R.entropy_bits:.4f}")
gd = root.gini - nL / N * L.gini - nR / N * R.gini
ig = root.entropy_bits - nL / N * L.entropy_bits - nR / N * R.entropy_bits
print(f"  Gini decrease    = G(parent) - nL/N*G(left) - nR/N*G(right)")
print(f"                   = {root.gini:.4f} - {nL/N:.4f}*{L.gini:.4f} - {nR/N:.4f}*{R.gini:.4f} = {gd:.5f}"
      f"   (file: {root.gini_decrease_of_split:.5f})")
print(f"  Information gain = H(parent) - nL/N*H(left) - nR/N*H(right) = {ig:.5f} bits"
      f"   (file: {root.information_gain_bits_of_split:.5f})")

# -----------------------------------------------------------------------------
line("3. ROOT -> LEAVES: impurity left after the whole tree  (tree 0 of Random Forest)")
leaves = rf[rf.is_leaf.astype(str).str.lower() == "true"]
w = leaves.samples_weighted / N
leaf_g = (w * leaves.gini).sum()
leaf_h = (w * leaves.entropy_bits).sum()
print(f"  Leaf-weighted Gini    = sum over leaves (n_leaf/N) * Gini(leaf) = {leaf_g:.5f}")
print(f"  Leaf-weighted entropy = sum over leaves (n_leaf/N) * H(leaf)    = {leaf_h:.5f} bits")
print(f"  Gini reduction = 1 - {leaf_g:.4f}/{root.gini:.4f} = {100*(1-leaf_g/root.gini):.1f}%")
print(f"  Entropy reduction (information gained) = 1 - {leaf_h:.4f}/{root.entropy_bits:.4f} = "
      f"{100*(1-leaf_h/root.entropy_bits):.1f}%")
pt = load("extra_trees_random_forest_per_tree.csv")
t0 = pt[(pt.model == "Random Forest") & (pt.tree == 0)].iloc[0]
print(f"  file (per_tree, RF tree 0): leaf_weighted_gini={t0.leaf_weighted_gini:.5f}, "
      f"leaf_weighted_entropy_bits={t0.leaf_weighted_entropy_bits:.5f}")
print(f"  check: total_gini_decrease = root Gini - leaf Gini = {root.gini - leaf_g:.5f} "
      f"(file: {t0.total_gini_decrease:.5f})")

# -----------------------------------------------------------------------------
line("4. FEATURE IMPORTANCE FROM IMPURITY (tree 0 of Random Forest)")
inner = rf[rf.is_leaf.astype(str).str.lower() == "false"]
contrib_g = inner.samples_weighted / N * inner.gini_decrease_of_split
contrib_h = inner.samples_weighted / N * inner.information_gain_bits_of_split
imp = pd.DataFrame({"gini_share_%": 100 * contrib_g.groupby(inner.split_feature).sum() / contrib_g.sum(),
                    "entropy_share_%": 100 * contrib_h.groupby(inner.split_feature).sum() / contrib_h.sum()})
print("  importance(feature) = sum of (n_node/N * decrease) over nodes splitting on that feature,")
print("                        divided by the same sum over all nodes")
print(imp.sort_values("gini_share_%", ascending=False).round(1).to_string())
print("  (the file averages this over all 300 trees -> feature_impurity_importance_all_tree_models.csv)")

# -----------------------------------------------------------------------------
line("5. BOOSTED TREES: log loss, gradients, hessians  (lightgbm_tree0_all_nodes.csv)")
lg = load("lightgbm_tree0_all_nodes.csv")
r0 = lg.iloc[0]
p0 = 1 / (1 + np.exp(-r0.value))
print(f"  Root of LightGBM tree 0: rows used = {int(r0['count'])}, node value (log-odds) = {r0.value:.5f}")
print(f"  Probability = 1/(1+exp(-log-odds)) = {p0:.5f}  (= training CVD share)")
print(f"  Hessian per row = p(1-p) = {p0*(1-p0):.5f}")
print(f"  Hessian cover = rows * p(1-p) = {int(r0['count'])} * {p0*(1-p0):.5f} = {r0['count']*p0*(1-p0):.1f}"
      f"   (file: {r0.hessian_cover:.1f})")
p = ds.P_CVD.iloc[0]
print(f"  Starting log loss = -[p ln p + (1-p) ln(1-p)] = {entropy_bits(p)*np.log(2):.5f} nats "
      f"= the training entropy {entropy_bits(p):.4f} bits * ln 2")
bs = load("xgboost_lightgbm_impurity_summary.csv")
print(bs[["model", "trees", "total_splits", "mean_gain_per_split", "starting_log_loss_nats (= label entropy)"]]
      .round(4).to_string(index=False))

# -----------------------------------------------------------------------------
line("6. GAIN IMPORTANCE (XGBoost / LightGBM)")
fi = load("feature_impurity_importance_all_tree_models.csv")
b = fi[fi.model.isin(["XGBoost", "LightGBM"])]
for m, g in b.groupby("model"):
    tot = g.total_gain_loss_reduction.sum()
    print(f"\n{m}: share = gain on feature / total gain ({tot:,.0f})")
    for _, r in g.sort_values("total_gain_loss_reduction", ascending=False).iterrows():
        print(f"  {r.feature:<18} {r.total_gain_loss_reduction:>12,.0f} / {tot:,.0f} = "
              f"{100*r.total_gain_loss_reduction/tot:5.1f}%   ({int(r.number_of_splits)} splits)")
