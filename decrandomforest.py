import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Decision Tree & Random Forest", page_icon="🌳", layout="wide")

st.title("Interactive Decision Tree & Random Forest Classification")
st.write("Upload labeled training data, evaluate performance depth-by-depth, feature-by-feature and test set size.")

# ---------------------------------------------------------------------------
# Data loading: built-in, file upload, or URL
# ---------------------------------------------------------------------------
sklearn_options = {
    # custom embedded dataset — default
    "Radfahren":               "radfahren",
    # sklearn datasets
    "Iris (sklearn)":          "iris",
    "Wine (sklearn)":          "wine",
    "Breast Cancer (sklearn)": "breast_cancer",
    "Digits (sklearn)":        "digits",
    # seaborn datasets
    "Penguins":    "penguins",
    "Tips":        "tips",
    "Titanic":     "titanic",
    "Diamonds":    "diamonds",
    "MPG":         "mpg",
    "Exercise":    "exercise",
    "Attention":   "attention",
    "Planets":     "planets",
    "Taxis":       "taxis",
    "Car Crashes": "car_crashes",
    "Geyser":      "geyser",
    "Anscombe":    "anscombe",
}

# seaborn datasets loaded via URL to avoid caching issues on Streamlit Cloud
seaborn_url_base = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
seaborn_datasets = {
    "penguins", "tips", "titanic", "diamonds", "mpg",
    "exercise", "attention", "planets", "taxis",
    "car_crashes", "geyser", "anscombe",
}

load_method = st.radio("Data source", ["Built-in dataset", "Upload CSV", "Load from URL"], horizontal=True)

df = None

if load_method == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file with features and target", type=["csv"])
    if uploaded is not None:
        try:
            try:
                df = pd.read_csv(uploaded, sep=",", on_bad_lines="skip")
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=None, engine="python", on_bad_lines="skip")
            df = df.dropna(axis=1, how="all").dropna(how="all")
            st.success(f"CSV loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("Waiting for CSV upload...")
        st.stop()

elif load_method == "Load from URL":
    url_input = st.text_input(
        "Enter CSV URL",
        placeholder="https://raw.githubusercontent.com/.../file.csv"
    )
    if url_input:
        try:
            # Try comma separator first
            try:
                df = pd.read_csv(url_input, sep=",", on_bad_lines="skip")
            except Exception:
                # Fallback: auto-detect separator
                df = pd.read_csv(url_input, sep=None, engine="python", on_bad_lines="skip")

            # Drop columns that are entirely NaN (artifact of extra commas)
            df = df.dropna(axis=1, how="all")
            # Drop rows that are entirely NaN
            df = df.dropna(how="all")

            st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from URL")
            if df.shape[0] == 0:
                st.error("File loaded but contains no valid rows. Check the CSV format.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load URL: {e}")
            st.info("Tips: Make sure the URL points to a raw CSV file, not an HTML page. "
                    "For GitHub, use the 'Raw' button to get the direct file URL.")
            st.stop()
    else:
        st.info("Paste a public CSV URL above to load data.")
        st.stop()

else:  # Built-in dataset
    dataset_from_sklearn = st.selectbox("Pick a built-in dataset", list(sklearn_options.keys()), index=0)
    key = sklearn_options[dataset_from_sklearn]

    if key == "radfahren":
        # Embedded directly — no file needed in repo
        df = pd.DataFrame({
            "Sonnig":      ["J","N","N","N","J","N","J","J","J"],
            "Schnee":      ["N","N","N","N","N","N","N","J","J"],
            "Auto_kaputt": ["N","N","J","N","J","J","N","N","J"],
            "Radfahren":   ["J","N","J","N","J","J","N","N","N"],
        })
    elif key == "iris":
        ds = load_iris(as_frame=True)
        df = pd.concat([ds.data, ds.target.rename("target")], axis=1)
    elif key == "wine":
        ds = load_wine(as_frame=True)
        df = pd.concat([ds.data, ds.target.rename("target")], axis=1)
    elif key == "breast_cancer":
        ds = load_breast_cancer(as_frame=True)
        df = pd.concat([ds.data, ds.target.rename("target")], axis=1)
    elif key == "digits":
        ds = load_digits(as_frame=True)
        df = pd.concat([ds.data, ds.target.rename("target")], axis=1)
    elif key in seaborn_datasets:
        try:
            url = f"{seaborn_url_base}{key}.csv"
            df = pd.read_csv(url)
            df = df.dropna()
            st.info(
                f"**{dataset_from_sklearn}** loaded — "
                f"{df.shape[0]} rows, {df.shape[1]} columns. "
                f"Select your target column below."
            )
        except Exception as e:
            st.error(f"Failed to load {dataset_from_sklearn}: {e}")
            st.stop()

st.write("### Dataset preview (top 10 rows)")
st.dataframe(df.head(10))

all_columns = df.columns.tolist()
if len(all_columns) < 2:
    st.warning("Need at least one feature and one target column.")
    st.stop()

target_column = st.selectbox("Select target column", all_columns, index=len(all_columns) - 1)

default_features = [col for col in all_columns if col != target_column]
feature_columns = st.multiselect(
    "Select feature columns",
    all_columns,
    default=default_features,
)

if target_column in feature_columns:
    st.warning("Target column cannot also be a feature. Please adjust your selection.")
    st.stop()

if len(feature_columns) == 0:
    st.warning("At least one feature column is required.")
    st.stop()

# ---------------------------------------------------------------------------
# Scatter plots: each feature vs target
# ---------------------------------------------------------------------------
st.write("### Feature vs Target Scatter Plots")

# Encode features and target just for scatter plotting
X_scatter = df[feature_columns].copy()
for col in X_scatter.columns:
    if X_scatter[col].dtype == object or X_scatter[col].dtype.name == "category":
        X_scatter[col] = X_scatter[col].astype("category").cat.codes.astype(int)

y_raw            = df[target_column]
y_labels_scatter = y_raw.astype(str)
unique_targets   = sorted(y_labels_scatter.unique())
# Encode target numerically for y-axis when categorical
y_scatter = y_raw.copy()
if y_scatter.dtype == object or y_scatter.dtype.name == "category":
    y_scatter = y_scatter.astype("category").cat.codes.astype(int)
color_palette = plt.cm.tab10.colors

# Layout: up to 4 plots per row
n_cols  = 4
n_feats = len(feature_columns)
n_rows  = max(1, -(-n_feats // n_cols))  # ceiling division

fig_scatter, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 4 * n_rows),
    squeeze=False
)

for idx, feat in enumerate(feature_columns):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]

    for t_idx, target_val in enumerate(unique_targets):
        mask   = y_labels_scatter == target_val
        x_vals = X_scatter[feat][mask]   # encoded numeric values
        y_vals = y_scatter[mask]          # encoded numeric values
        ax.scatter(
            x_vals,
            y_vals,
            label=str(target_val),
            alpha=0.6,
            color=color_palette[t_idx % len(color_palette)],
            s=20,
        )

    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel(target_column, fontsize=9)
    ax.set_title(f"{feat} vs {target_column}", fontsize=10)
    ax.legend(fontsize=7, title=target_column, title_fontsize=7)

# Hide unused subplots
for idx in range(n_feats, n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes[row][col].set_visible(False)

plt.tight_layout()
st.pyplot(fig_scatter)
plt.close()

# ---------------------------------------------------------------------------
# Two-column layout: Model Parameters left, Pairplot right
# ---------------------------------------------------------------------------
col_params, col_pairplot = st.columns(2)

# --- LEFT: Model Parameters ---
with col_params:
    st.write("### Model Parameters")
    algorithm = st.radio("Algorithm", ["Decision Tree", "Random Forest"], horizontal=True)
    max_depth_input = st.slider("Max tree depth to evaluate", min_value=1, max_value=10, value=3)
    if algorithm == "Random Forest":
        n_estimators = st.slider("Number of trees", min_value=10, max_value=500, value=100, step=10)

    eval_mode = st.radio(
        "Evaluation mode",
        ["Train/Test Split", "Cross Validation", "Use All Data"],
        horizontal=True
    )
    if eval_mode == "Train/Test Split":
        test_size = st.slider("Test set size (%)", min_value=1, max_value=50, value=20)
    elif eval_mode == "Cross Validation":
        n_folds = st.slider("Number of folds", min_value=2, max_value=20, value=5)

    # keep backward compat for use_all_data flag used later
    use_all_data = (eval_mode == "Use All Data")

# ---------------------------------------------------------------------------
# Encode features and target (must happen before pairplot and training)
# ---------------------------------------------------------------------------
X = df[feature_columns].copy()
y = df[target_column].copy()

# Store original target labels before encoding (for pairplot legend)
y_labels = y.astype(str)

# Encode target if not numeric
if not pd.api.types.is_numeric_dtype(y):
    y = y.astype(str).astype("category").cat.codes

# Encode ALL non-numeric feature columns
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].astype(str).astype("category").cat.codes

# Convert to plain numpy-compatible types
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

# ---------------------------------------------------------------------------
# Train / test split — or cross validation — or all data
# ---------------------------------------------------------------------------
if eval_mode == "Use All Data":
    X_train = X
    X_test  = X
    y_train = y
    y_test  = y
    with col_params:
        st.write(f"Training samples: {X_train.shape[0]} (all data — training accuracy only)")

elif eval_mode == "Cross Validation":
    # Use all data for CV — train/test split done internally per fold
    X_train = X
    X_test  = X
    y_train = y
    y_test  = y
    with col_params:
        st.write(f"Cross validation: {n_folds} folds on {X.shape[0]} samples")

else:  # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100.0, random_state=42, stratify=y
    )
    with col_params:
        st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# --- RIGHT: Pairplot ---
with col_pairplot:
    st.write("### Pairplot — Feature Relationships colored by Target")
    max_pairplot_features = 6
    pairplot_features = feature_columns[:max_pairplot_features]
    if len(pairplot_features) < 2:
        st.info("Select at least 2 feature columns to show the pairplot.")
    else:
        if len(feature_columns) > max_pairplot_features:
            st.caption(f"Showing first {max_pairplot_features} features only to keep the plot readable.")
        try:
            pairplot_df = X[pairplot_features].copy()
            pairplot_df["target"] = y_labels.values
            fig_pair = sns.pairplot(pairplot_df, hue="target", diag_kind="kde", plot_kws={"alpha": 0.6})
            fig_pair.fig.suptitle(f"Pairplot colored by {target_column}", y=1.02)
            st.pyplot(fig_pair.fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render pairplot: {e}")

# ---------------------------------------------------------------------------
# Helper: build classifier
# ---------------------------------------------------------------------------
def make_clf(depth):
    if algorithm == "Decision Tree":
        return DecisionTreeClassifier(max_depth=depth, random_state=42)
    else:
        return RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, random_state=42)

# ---------------------------------------------------------------------------
# Depth accuracy loop
# ---------------------------------------------------------------------------
results = []
for depth in range(1, max_depth_input + 1):
    clf = make_clf(depth)
    if eval_mode == "Cross Validation":
        cv_scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
        acc = cv_scores.mean()
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    results.append((depth, acc))

results_df = pd.DataFrame(results, columns=["depth", "accuracy"])

# ---------------------------------------------------------------------------
# Information gain at root split for each feature
# ---------------------------------------------------------------------------
def calc_entropy(series):
    probs = series.value_counts(normalize=True)
    probs_nz = probs[probs > 0]
    return -np.sum(probs_nz * np.log2(probs_nz))

parent_entropy = calc_entropy(y_train)

def calc_gini(series):
    probs = series.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

parent_entropy = calc_entropy(y_train)
parent_gini    = calc_gini(y_train)

ig_results = []
for feat in feature_columns:
    groups = y_train.groupby(X_train[feat])

    weighted_entropy = sum(
        (len(g) / len(y_train)) * calc_entropy(g)
        for _, g in groups
    )
    weighted_gini = sum(
        (len(g) / len(y_train)) * calc_gini(g)
        for _, g in groups
    )

    ig_results.append({
        "Feature":          feat,
        "Info Gain":        round(parent_entropy - weighted_entropy, 4),
        "Weighted Entropy": round(weighted_entropy, 4),
        "Gini Reduction":   round(parent_gini - weighted_gini, 4),
        "Weighted Gini":    round(weighted_gini, 4),
    })

ig_df = pd.DataFrame(ig_results).sort_values("Info Gain", ascending=False)

with col_params:
    # Information gain table — above accuracy chart
    st.write("### Information Gain at Root Split")
    st.caption(f"Parent entropy: {parent_entropy:.4f}   |   Parent gini: {parent_gini:.4f}")
    st.dataframe(ig_df, hide_index=True)

    # Accuracy chart — below information gain
    if eval_mode == "Cross Validation":
        st.write(f"Accuracy per depth — {n_folds}-fold Cross Validation (mean accuracy)")
    else:
        st.write("Accuracy at different Depth. Vary test size using slider to see how it changes.")
    if results_df.empty:
        st.warning("No depth results available. Adjust max depth or check data.")
    else:
        st.line_chart(results_df.set_index("depth"))

    # Best depth summary — below accuracy chart
    st.write("### Best depth summary")
    best_depth = results_df.loc[results_df["accuracy"].idxmax(), "depth"]
    best_acc   = results_df["accuracy"].max()
    st.write(f"Algorithm: {algorithm}")
    st.write(f"Best depth: {best_depth} with accuracy {best_acc:.4f}")

    # Show per-fold scores for cross validation
    if eval_mode == "Cross Validation":
        best_clf = make_clf(best_depth)
        cv_scores = cross_val_score(best_clf, X, y, cv=n_folds, scoring="accuracy")
        st.write(f"CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        fold_df = pd.DataFrame({
            "Fold":     [f"Fold {i+1}" for i in range(n_folds)],
            "Accuracy": [f"{s:.4f}" for s in cv_scores]
        })
        st.dataframe(fold_df, hide_index=True)

# ---------------------------------------------------------------------------
# Tree inspection at selected depth
# ---------------------------------------------------------------------------
selected_depth = st.select_slider(
    "Select depth to inspect",
    options=list(range(1, max_depth_input + 1)),
    value=min(3, max_depth_input)
)

clf = make_clf(selected_depth)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### {algorithm} at depth {selected_depth}")
if eval_mode == "Cross Validation":
    cv_scores_sel = cross_val_score(make_clf(selected_depth), X, y, cv=n_folds, scoring="accuracy")
    st.write(f"CV accuracy: {cv_scores_sel.mean():.4f} ± {cv_scores_sel.std():.4f}  "
             f"(single-fold shown in tree below: {acc:.4f})")
else:
    st.write(f"Accuracy at this depth: {acc:.4f}")

# ---------------------------------------------------------------------------
# Tree visualization — Decision Tree only
# ---------------------------------------------------------------------------
if algorithm == "Decision Tree":
    fig, ax = plt.subplots(figsize=(12, 8))
    artists = plot_tree(
        clf,
        feature_names=feature_columns,
        class_names=[str(c) for c in sorted(y.unique())],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )

    # Annotate each node with entropy — matched by samples + gini
    tree_ = clf.tree_
    for i in range(tree_.node_count):
        artist = artists[i]
        current_text = artist.get_text()
        for node_id in range(tree_.node_count):
            node_samples = tree_.n_node_samples[node_id]
            node_gini    = tree_.impurity[node_id]
            if (f"samples = {node_samples}" in current_text and
                    f"{node_gini:.3f}" in current_text):
                values = tree_.value[node_id][0]
                total  = values.sum()
                if total > 0:
                    probs    = values / total
                    probs_nz = probs[probs > 0]
                    entropy  = -np.sum(probs_nz * np.log2(probs_nz))
                else:
                    entropy = 0.0
                artist.set_text(current_text + f"\nentropy = {entropy:.3f}")
                break

    st.pyplot(fig)

else:
    st.info(
        "Tree visualization is not shown for Random Forest — it builds 100+ trees internally. "
        "Use the Feature Importances section below to interpret the model."
    )

# ---------------------------------------------------------------------------
# Three-column layout: Classification report | Feature Importances | Manual Prediction
# ---------------------------------------------------------------------------
col_report, col_importance, col_predict = st.columns(3)

# --- LEFT 1/3: Classification report ---
with col_report:
    st.write("### Classification report")
    st.text(f"Target: {target_column}\n\n{classification_report(y_test, y_pred)}")

# --- MIDDLE 1/3: Feature importances ---
with col_importance:
    st.write("### Feature Importances")

    importance_df = pd.DataFrame({
        "Feature":    feature_columns,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    ax_imp.bar(importance_df["Feature"], importance_df["Importance"])
    ax_imp.set_xlabel("Feature")
    ax_imp.set_ylabel("Importance")
    ax_imp.set_title(f"Importances at depth {selected_depth}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_imp)

    importance_df["Importance"] = importance_df["Importance"].map("{:.3f}".format)
    st.dataframe(importance_df)

# --- RIGHT 1/3: Manual prediction ---
with col_predict:
    st.write("### Manual Prediction")
    st.write("Enter feature values below to predict the target class.")

    input_values = {}

    for col_name in feature_columns:
        original_col = df[col_name]
        if not pd.api.types.is_numeric_dtype(original_col):
            unique_vals  = sorted(original_col.astype(str).dropna().unique().tolist())
            selected_val = st.selectbox(f"{col_name}", unique_vals, key=f"input_{col_name}")
            input_values[col_name] = float(unique_vals.index(selected_val))
        else:
            col_min  = float(pd.to_numeric(original_col, errors="coerce").min())
            col_max  = float(pd.to_numeric(original_col, errors="coerce").max())
            col_mean = float(pd.to_numeric(original_col, errors="coerce").mean())
            input_values[col_name] = st.number_input(
                f"{col_name}",
                min_value=col_min,
                max_value=col_max,
                value=round(col_mean, 4),
                key=f"input_{col_name}"
            )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_values])

        prediction_encoded = clf.predict(input_df)[0]
        prediction_proba   = clf.predict_proba(input_df)[0]

        # Map encoded prediction back to original label
        unique_encoded   = sorted(y.unique())
        unique_labels    = sorted(y_labels.unique())
        label_map        = dict(zip(unique_encoded, unique_labels))
        prediction_label = label_map.get(prediction_encoded, str(prediction_encoded))

        st.success(f"Predicted {target_column}: **{prediction_label}**")

        proba_df = pd.DataFrame({
            "Class":       [label_map.get(c, str(c)) for c in unique_encoded],
            "Probability": [f"{p:.3f}" for p in prediction_proba]
        })
        st.write("Class probabilities:")
        st.dataframe(proba_df, hide_index=True)
