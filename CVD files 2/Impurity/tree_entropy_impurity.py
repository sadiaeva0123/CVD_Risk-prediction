#!/usr/bin/env python3
# =============================================================================
# Entropy and node impurity of the trained tree-based models (Kaggle)
# Models: Extra Trees, Random Forest (class-probability trees)
#         XGBoost, LightGBM (gradient-boosted trees)
# Input : the training run output (models/*.joblib + cvd_components.py) - any
#         Kaggle input that contains these files (e.g. the BD prediction output).
#         Optional: BD file with actual labels -> class entropy of BD.
# Output: /kaggle/working/tree_impurity/
# Nothing is retrained; the values are read from the saved, fitted trees.
# =============================================================================
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

INPUT_ROOT = os.environ.get("TI_INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("TI_OUT", "/kaggle/working/tree_impurity")
os.makedirs(OUT, exist_ok=True)
FEATURES = ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol",
            "sex", "smoking", "diabetes", "physical_activity"]      # order the trees see
LABEL = {"extra_trees": "Extra Trees", "random_forest": "Random Forest",
         "xgboost": "XGBoost", "lightgbm": "LightGBM"}
GEN = []


def save(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    GEN.append(p)
    return p


def find(name):
    c = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", name), recursive=True))
    if not c:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    return c[0]


sys.path.insert(0, os.path.dirname(find("cvd_components.py")))
import cvd_components  # noqa: F401  (needed to unpickle the pipelines)


def entropy_bits(p1):
    p = np.clip(np.asarray(p1, float), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0) + np.where(p < 1, (1 - p) * np.log2(1 - p), 0))
    return h


def gini(p1):
    p = np.asarray(p1, float)
    return 1 - p ** 2 - (1 - p) ** 2


# =============================================================================
# 1. EXTRA TREES + RANDOM FOREST (every node stores its class proportions)
# =============================================================================
model_rows, tree_tables, feat_rows = [], [], []
dataset_rows = []
for key in ["extra_trees", "random_forest"]:
    clf = joblib.load(find(f"{key}.joblib")).named_steps["clf"]
    per_tree = []
    fi_gini = np.zeros(len(FEATURES))
    fi_ent = np.zeros(len(FEATURES))
    for t_i, est in enumerate(clf.estimators_):
        t = est.tree_
        v = t.value[:, 0, :]
        p1 = v[:, 1] / v.sum(axis=1)                      # P(CVD) in each node
        w = t.weighted_n_node_samples
        L, R = t.children_left, t.children_right
        leaf = L == -1
        g, h = gini(p1), entropy_bits(p1)
        inner = ~leaf
        gd = np.zeros(t.node_count)
        ig = np.zeros(t.node_count)
        gd[inner] = w[inner] * g[inner] - w[L[inner]] * g[L[inner]] - w[R[inner]] * g[R[inner]]
        ig[inner] = w[inner] * h[inner] - w[L[inner]] * h[L[inner]] - w[R[inner]] * h[R[inner]]
        f = t.feature[inner]
        tg = np.bincount(f, weights=gd[inner], minlength=len(FEATURES))
        te = np.bincount(f, weights=ig[inner], minlength=len(FEATURES))
        if tg.sum() > 0:
            fi_gini += tg / tg.sum()
        if te.sum() > 0:
            fi_ent += te / te.sum()
        W = w[0]
        per_tree.append({
            "model": LABEL[key], "tree": t_i, "n_nodes": t.node_count, "n_leaves": int(leaf.sum()),
            "max_depth": t.max_depth, "root_samples_weighted": W,
            "root_P_CVD": p1[0], "root_gini": g[0], "root_entropy_bits": h[0],
            "leaf_weighted_gini": float((w[leaf] * g[leaf]).sum() / W),
            "leaf_weighted_entropy_bits": float((w[leaf] * h[leaf]).sum() / W),
            "total_gini_decrease": float(gd.sum() / W),
            "total_information_gain_bits": float(ig.sum() / W),
            "gini_matches_stored_impurity": bool(np.allclose(g, t.impurity, atol=1e-9)),
        })
        if t_i == 0:   # full node-by-node table for the first tree
            depth = np.zeros(t.node_count, int)
            for n in range(t.node_count):
                if not leaf[n]:
                    depth[L[n]] = depth[R[n]] = depth[n] + 1
            node = pd.DataFrame({
                "node_id": np.arange(t.node_count), "depth": depth, "is_leaf": leaf,
                "split_feature": np.where(leaf, "", np.array(FEATURES)[np.maximum(t.feature, 0)]),
                "split_threshold": np.where(leaf, np.nan, t.threshold),
                "left_child": L, "right_child": R,
                "samples": t.n_node_samples, "samples_weighted": w,
                "P_CVD": p1, "gini": g, "entropy_bits": h,
                "gini_decrease_of_split": np.where(leaf, np.nan, gd / np.maximum(w, 1e-12)),
                "information_gain_bits_of_split": np.where(leaf, np.nan, ig / np.maximum(w, 1e-12))})
            save(node, f"{key}_tree0_all_nodes.csv")
    pt = pd.DataFrame(per_tree)
    tree_tables.append(pt)
    fi_gini /= len(clf.estimators_)
    fi_ent /= len(clf.estimators_)
    for j, fname in enumerate(FEATURES):
        feat_rows.append({"model": LABEL[key], "feature": fname,
                          "gini_importance_MDI": fi_gini[j], "entropy_importance_information_gain": fi_ent[j]})
    model_rows.append({
        "model": LABEL[key], "trees": len(pt), "split_criterion_used_in_training": clf.criterion,
        "total_nodes": int(pt.n_nodes.sum()), "total_leaves": int(pt.n_leaves.sum()),
        "mean_nodes_per_tree": pt.n_nodes.mean(), "mean_depth": pt.max_depth.mean(),
        "root_P_CVD (training)": pt.root_P_CVD.mean(),
        "root_gini": pt.root_gini.mean(), "root_entropy_bits": pt.root_entropy_bits.mean(),
        "leaf_weighted_gini": pt.leaf_weighted_gini.mean(),
        "leaf_weighted_entropy_bits": pt.leaf_weighted_entropy_bits.mean(),
        "gini_reduction_percent": 100 * (1 - pt.leaf_weighted_gini.mean() / pt.root_gini.mean()),
        "entropy_reduction_percent": 100 * (1 - pt.leaf_weighted_entropy_bits.mean() / pt.root_entropy_bits.mean()),
        "check_gini_equals_sklearn_impurity": "Yes" if pt.gini_matches_stored_impurity.all() else "NO"})
    if key == "extra_trees":   # no bootstrap -> root node = the whole training split
        r = pt.iloc[0]
        dataset_rows.append({"dataset": "training split (Extra Trees root node)",
                             "patients": int(r.root_samples_weighted), "P_CVD": r.root_P_CVD,
                             "gini": r.root_gini, "entropy_bits": r.root_entropy_bits,
                             "entropy_nats": r.root_entropy_bits * np.log(2)})
    print(f"{LABEL[key]}: {len(pt)} trees, {int(pt.n_nodes.sum()):,} nodes")

# =============================================================================
# 2. XGBOOST + LIGHTGBM (boosted trees: impurity = log loss / binary cross-entropy)
# =============================================================================
boost_model_rows, boost_trees = [], []
for key in ["xgboost", "lightgbm"]:
    clf = joblib.load(find(f"{key}.joblib")).named_steps["clf"]
    if key == "xgboost":
        b = clf.get_booster()
        df = b.trees_to_dataframe()
        base = float(json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
        df["is_leaf"] = df["Feature"] == "Leaf"
        df["feature_name"] = np.where(df.is_leaf, "",
                                      df["Feature"].str.replace("f", "", regex=False).apply(
                                          lambda s: FEATURES[int(s)] if isinstance(s, str) and s.isdigit() else ""))
        df["gain"] = np.where(df.is_leaf, np.nan, df["Gain"])
        df["hessian_cover"] = df["Cover"]
        df["depth"] = np.nan
        tree_col = "Tree"
        detail = df[df.Tree == 0][["Tree", "ID", "feature_name", "Split", "Yes", "No", "is_leaf", "gain",
                                    "hessian_cover", "Gain"]].rename(
            columns={"Gain": "gain_or_leaf_value", "Split": "split_threshold"})
    else:
        b = clf.booster_
        df = b.trees_to_dataframe()
        n_pos = None
        base = None
        df["is_leaf"] = df["split_feature"].isna()
        df["feature_name"] = np.where(df.is_leaf, "", df["split_feature"].astype(str).str.replace(
            "Column_", "", regex=False).apply(lambda s: FEATURES[int(s)] if isinstance(s, str) and s.isdigit() else ""))
        df["gain"] = np.where(df.is_leaf, np.nan, df["split_gain"])
        df["hessian_cover"] = df["weight"]
        tree_col = "tree_index"
        detail = df[df.tree_index == 0][["tree_index", "node_index", "node_depth", "feature_name", "threshold",
                                         "left_child", "right_child", "is_leaf", "gain", "hessian_cover",
                                         "count", "value"]]
        # boost_from_average: the first tree's root value holds the starting log-odds (training prevalence)
        base = float(1 / (1 + np.exp(-df[df.tree_index == 0]["value"].iloc[0])))
    save(detail, f"{key}_tree0_all_nodes.csv")
    splits = df[~df.is_leaf]
    per = df.groupby(tree_col).agg(n_nodes=("is_leaf", "size"), n_leaves=("is_leaf", "sum"))
    per["n_splits"] = per.n_nodes - per.n_leaves
    per["total_gain_loss_reduction"] = splits.groupby(tree_col)["gain"].sum()
    per = per.reset_index().rename(columns={tree_col: "tree"})
    per.insert(0, "model", LABEL[key])
    boost_trees.append(per)
    fg = splits.groupby("feature_name")["gain"].agg(["sum", "count"])
    fc = splits.groupby("feature_name")["hessian_cover"].sum()
    for fname in FEATURES:
        s = fg["sum"].get(fname, 0.0)
        feat_rows.append({"model": LABEL[key], "feature": fname,
                          "gain_importance (loss reduction share)": s / fg["sum"].sum(),
                          "total_gain_loss_reduction": s, "number_of_splits": int(fg["count"].get(fname, 0)),
                          "total_hessian_cover": float(fc.get(fname, 0.0))})
    boost_model_rows.append({
        "model": LABEL[key], "trees": int(df[tree_col].nunique()), "total_nodes": len(df),
        "total_splits": len(splits), "total_leaves": int(df.is_leaf.sum()),
        "node_impurity_measure": "binary log loss (cross-entropy), second-order approximation; "
                                 "split gain = reduction of the regularised loss",
        "total_gain_all_splits": float(splits["gain"].sum()),
        "mean_gain_per_split": float(splits["gain"].mean()),
        "starting_P_CVD (base score)": base if base is not None else np.nan})
    print(f"{LABEL[key]}: {df[tree_col].nunique()} trees, {len(df):,} nodes")

# XGBoost/LightGBM start from the training prevalence; their starting loss = label entropy (nats)
for r in boost_model_rows:
    p0 = r["starting_P_CVD (base score)"]
    if np.isnan(p0):
        p0 = dataset_rows[0]["P_CVD"] if dataset_rows else np.nan
        r["starting_P_CVD (base score)"] = p0
    r["starting_log_loss_nats (= label entropy)"] = float(entropy_bits(p0) * np.log(2))

# optional: class entropy of the BD dataset (actual labels)
try:
    bdp = [p for p in glob.glob(os.path.join(INPUT_ROOT, "**", "*.csv"), recursive=True)
           if "with actual" in os.path.basename(p).lower().replace("_", " ")]
    if bdp:
        yb = pd.read_csv(bdp[0])["CVD Prediction"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        pb = float(yb.mean())
        dataset_rows.append({"dataset": "BD dataset (actual labels)", "patients": int(yb.notna().sum()),
                             "P_CVD": pb, "gini": float(gini(pb)), "entropy_bits": float(entropy_bits(pb)),
                             "entropy_nats": float(entropy_bits(pb) * np.log(2))})
except Exception as e:
    print("BD class entropy skipped:", e)

# =============================================================================
# 3. SAVE
# =============================================================================
save(pd.DataFrame(dataset_rows), "class_entropy_and_gini_of_datasets.csv")
save(pd.DataFrame(model_rows), "extra_trees_random_forest_impurity_summary.csv")
save(pd.concat(tree_tables, ignore_index=True), "extra_trees_random_forest_per_tree.csv")
save(pd.DataFrame(boost_model_rows), "xgboost_lightgbm_impurity_summary.csv")
save(pd.concat(boost_trees, ignore_index=True), "xgboost_lightgbm_per_tree.csv")
save(pd.DataFrame(feat_rows), "feature_impurity_importance_all_tree_models.csv")

pd.set_option("display.width", 220)
print("\nDATASET CLASS ENTROPY / GINI\n", pd.DataFrame(dataset_rows).round(4).to_string(index=False))
print("\nEXTRA TREES / RANDOM FOREST\n", pd.DataFrame(model_rows).round(4).T.to_string())
print("\nXGBOOST / LIGHTGBM\n", pd.DataFrame(boost_model_rows).round(4).T.to_string())
print("\nGenerated files:")
for g in GEN:
    print("   ", g)
