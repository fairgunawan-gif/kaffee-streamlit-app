import streamlit as st
import pandas as pd

st.title("League Table Viewer")

# Example: loading from Excel
df = pd.read_excel("premier_league_output.xlsx")

st.subheader("Full Table")
st.dataframe(df)

# Example filters
team = st.selectbox("Select a team", df["Team"].unique())
team_row = df[df["Team"] == team]

st.subheader("Selected Team Stats")
st.write(team_row)