import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Decision Tree Classifier", page_icon="🌳", layout="wide")

st.title("Interactive Decision Tree Classification")
st.write("Upload labeled training data, evaluate performance depth-by-depth, feature-by-feature and test set size.")

uploaded = st.file_uploader("Optional: Upload a CSV file with features and target", type=["csv"])

sklearn_options = {
    "Iris": "iris",
    "Wine": "wine",
    "Breast Cancer": "breast_cancer",
}

dataset_from_sklearn = st.selectbox("Or pick a built-in sklearn dataset", list(sklearn_options.keys()), index=0)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("CSV loaded successfully")
else:
    st.info(f"Using built-in {dataset_from_sklearn} data. Upload a CSV to override.")
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer

    key = sklearn_options[dataset_from_sklearn]

    if key == "iris":
        ds = load_iris(as_frame=True)
    elif key == "wine":
        ds = load_wine(as_frame=True)
    else:
        ds = load_breast_cancer(as_frame=True)

    df = pd.concat([ds.data, ds.target.rename("target")], axis=1)

st.write("### Dataset preview (works with features value of type number)")
st.dataframe(df.head())

all_columns = df.columns.tolist()
if len(all_columns) < 2:
    st.warning("Need at least one feature and one target column.")
    st.stop()

target_column = st.selectbox("Select target column", all_columns, index=len(all_columns)-1)

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

col_left, col_right = st.columns(2)

# Right column: Sliders
with col_right:
    st.write("### Model Parameters")
    max_depth_input = st.slider("Max tree depth to evaluate", min_value=1, max_value=20, value=5)
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=30)

X = df[feature_columns]
y = df[target_column]

# If target is not numeric, encode it
if y.dtype == object or y.dtype.name == 'category':
    y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, random_state=42, stratify=y)

st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

results = []

for depth in range(1, max_depth_input + 1):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((depth, acc))

results_df = pd.DataFrame(results, columns=["depth", "accuracy"])

# Left column: Plot
with col_left:
    st.write("## Vary Test-set Size to Show Accuracy using different Depth")
    if results_df.empty:
        st.warning("No depth results available. Adjust max depth or check data.")
    else:
        st.line_chart(results_df.set_index("depth"))

selected_depth = st.select_slider("Select depth to inspect", options=list(range(1, max_depth_input + 1)), value=1)

clf = DecisionTreeClassifier(max_depth=selected_depth, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Tree at depth {selected_depth}")
st.write(f"Accuracy at this depth: {acc:.4f}")

fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=feature_columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
st.pyplot(fig)

st.write("### Classification report")
# st.text(classification_report(y_test, y_pred))
st.text(f"target {target_column} {classification_report(y_test, y_pred)}")

st.write("### Best depth summary")
best_depth = results_df.loc[results_df["accuracy"].idxmax(), "depth"]
best_acc = results_df["accuracy"].max()
st.write(f"Best depth: {best_depth} with accuracy {best_acc:.4f}")
