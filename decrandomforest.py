import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
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
    "Iris": "iris",
    "Wine": "wine",
    "Breast Cancer": "breast_cancer",
    "Digits": "digits",
    "Penguins": "penguins",
}

load_method = st.radio("Data source", ["Built-in dataset", "Upload CSV", "Load from URL"], horizontal=True)

df = None

if load_method == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file with features and target", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("CSV loaded successfully")
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
            df = pd.read_csv(url_input)
            st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from URL")
        except Exception as e:
            st.error(f"Failed to load URL: {e}")
            st.stop()
    else:
        st.info("Paste a public CSV URL above to load data.")
        st.stop()

else:  # Built-in dataset
    dataset_from_sklearn = st.selectbox("Pick a built-in dataset", list(sklearn_options.keys()), index=0)
    key = sklearn_options[dataset_from_sklearn]

    if key == "iris":
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
    elif key == "penguins":
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
        df = pd.read_csv(url).dropna()
        df = df.rename(columns={"species": "target"})
        df["island"] = df["island"].astype("category").cat.codes
        df["sex"] = df["sex"].astype("category").cat.codes

st.write("### Dataset preview")
st.dataframe(df.head())

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
# Model Parameters
# ---------------------------------------------------------------------------
st.write("### Model Parameters")

algorithm = st.radio("Algorithm", ["Decision Tree", "Random Forest"], horizontal=True)
max_depth_input = st.slider("Max tree depth to evaluate", min_value=1, max_value=10, value=3)

if algorithm == "Random Forest":
    n_estimators = st.slider("Number of trees", min_value=10, max_value=500, value=100, step=10)

use_all_data = st.checkbox("Use all data for tree building only (no train/test split)", value=False)

if not use_all_data:
    test_size = st.slider("Test set size (%)", min_value=1, max_value=50, value=20)

# ---------------------------------------------------------------------------
# Encode features and target
# ---------------------------------------------------------------------------
X = df[feature_columns].copy()
y = df[target_column].copy()

# Store original target labels before encoding (for pairplot legend)
y_labels = y.astype(str)

# Encode target if not numeric
if y.dtype == object or y.dtype.name == "category":
    y = y.astype("category").cat.codes

# Encode categorical feature columns
for col in X.columns:
    if X[col].dtype == object or X[col].dtype.name == "category":
        X[col] = X[col].astype("category").cat.codes

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
if use_all_data:
    X_train = X
    X_test  = X
    y_train = y
    y_test  = y
    st.write(f"Training samples: {X_train.shape[0]} (all data — training accuracy only)")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100.0, random_state=42, stratify=y
    )
    st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

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
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((depth, acc))

results_df = pd.DataFrame(results, columns=["depth", "accuracy"])

# ---------------------------------------------------------------------------
# Two columns: accuracy chart left, best depth right
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.write("Accuracy at different Depth. Vary test size using slider to see how it changes.")
    if results_df.empty:
        st.warning("No depth results available. Adjust max depth or check data.")
    else:
        st.line_chart(results_df.set_index("depth"))

with col_right:
    st.write("### Best depth summary")
    best_depth = results_df.loc[results_df["accuracy"].idxmax(), "depth"]
    best_acc   = results_df["accuracy"].max()
    st.write(f"Algorithm: {algorithm}")
    st.write(f"Best depth: {best_depth} with accuracy {best_acc:.4f}")

# ---------------------------------------------------------------------------
# Pairplot
# ---------------------------------------------------------------------------
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
# Classification report
# ---------------------------------------------------------------------------
st.write("### Classification report")
st.text(f"Target: {target_column}\n\n{classification_report(y_test, y_pred)}")

# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------
st.write("### Feature Importances")

importance_df = pd.DataFrame({
    "Feature":    feature_columns,
    "Importance": clf.feature_importances_
}).sort_values("Importance", ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
ax_imp.bar(importance_df["Feature"], importance_df["Importance"])
ax_imp.set_xlabel("Feature")
ax_imp.set_ylabel("Importance")
ax_imp.set_title(f"Feature Importances — {algorithm} at depth {selected_depth}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig_imp)

importance_df["Importance"] = importance_df["Importance"].map("{:.3f}".format)
st.dataframe(importance_df)

# ---------------------------------------------------------------------------
# Manual prediction input
# ---------------------------------------------------------------------------
st.write("### Manual Prediction")
st.write("Enter feature values below to predict the target class.")

input_values = {}
input_cols = st.columns(min(len(feature_columns), 4))

for idx, col_name in enumerate(feature_columns):
    col_widget = input_cols[idx % len(input_cols)]
    original_col = df[col_name]

    with col_widget:
        if original_col.dtype == object or original_col.dtype.name == "category":
            # Show original string options for categorical columns
            unique_vals = sorted(df[col_name].dropna().unique().tolist())
            selected_val = st.selectbox(f"{col_name}", unique_vals, key=f"input_{col_name}")
            # Encode to match training encoding
            encoded_val = unique_vals.index(selected_val)
            input_values[col_name] = encoded_val
        else:
            col_min  = float(original_col.min())
            col_max  = float(original_col.max())
            col_mean = float(original_col.mean())
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
    unique_encoded = sorted(y.unique())
    unique_labels  = sorted(y_labels.unique())
    label_map      = dict(zip(unique_encoded, unique_labels))
    prediction_label = label_map.get(prediction_encoded, str(prediction_encoded))

    st.success(f"Predicted {target_column}: **{prediction_label}**")

    # Show class probabilities
    proba_df = pd.DataFrame({
        "Class":       [label_map.get(c, str(c)) for c in unique_encoded],
        "Probability": [f"{p:.3f}" for p in prediction_proba]
    })
    st.write("Class probabilities:")
    st.dataframe(proba_df, hide_index=True)
