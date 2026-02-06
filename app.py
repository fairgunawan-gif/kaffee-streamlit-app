import streamlit as st
import pandas as pd

st.title("My First Streamlit App")

st.write("This is a simple example running in VS Code.")

# Simple DataFrame example
data = {
    "Team": ["A", "B", "C"],
    "Points": [8, 13, 18],
}
df = pd.DataFrame(data)

st.subheader("League Table")
st.dataframe(df)

# Simple widget example
points_filter = st.slider("Minimum points", 0, 30, 5)
filtered_df = df[df["Points"] >= points_filter]

st.subheader("Filtered Teams")
st.dataframe(filtered_df)