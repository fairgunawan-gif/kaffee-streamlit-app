import streamlit as st
import pandas as pd
import altair as alt

st.title("Dashboard zur wöchentlichen Kontrolle")

# Example: loading from Excel
df = pd.read_excel("kaffeekette_logistik_daten1.xlsx")

# (optional) convert German decimal comma to float
df["Umsatzverlust_EUR"] = (
    df["Umsatzverlust_EUR"].astype(str).str.replace(",", ".").astype(float)
)

df["Verzoegerung_Min"] = (
    df["Verzoegerung_Min"].astype(str).str.replace(",", ".").astype(float)
)

# keep only the three routes you want (adapt names if yours are just "A", "B", "C")
routes = ["Route_A", "Route_B", "Route_C"]
df_plot = (
    df[df["Lieferroute"].isin(routes)]
    .groupby(["Kalenderwoche", "Lieferroute"], as_index=False)["Umsatzverlust_EUR"]
    .mean()
)

df_plot_verz = (
    df[df["Lieferroute"].isin(routes)]
    .groupby(["Kalenderwoche", "Lieferroute"], as_index=False)["Verzoegerung_Min"]
    .mean()
)

st.subheader("Mittelwert von Umsatzverlust pro Kalenderwoche")
pivot = df_plot.pivot(
    index="Kalenderwoche",
    columns="Lieferroute",
    values="Umsatzverlust_EUR",
).sort_index()

st.line_chart(pivot)

st.subheader("Mittelwert von Verzoegerung pro Kalenderwoche")
pivot_verz = df_plot_verz.pivot(
    index="Kalenderwoche",
    columns="Lieferroute",
    values="Verzoegerung_Min",
).sort_index()

st.line_chart(pivot_verz)

st.markdown("date 06.02.26", text_alignment="left")

# st.dataframe(df)

# # Example filters
# team = st.selectbox("Select a team", df["Team"].unique())
# team_row = df[df["Team"] == team]

# st.subheader("Selected Team Stats")
# st.write(team_row)