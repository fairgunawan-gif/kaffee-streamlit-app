import streamlit as st
import pandas as pd
import altair as alt

st.title("Dashboard zur wöchentlichen Kontrolle")

# Example: loading from Excel
df = pd.read_excel("kaffeekette_logistik_daten1.xlsx")

df["Datum"] = pd.to_datetime(df["Datum"])

# Monday-based week start (business calendar, not ISO week-year)
df["Week_Start"] = df["Datum"] - pd.to_timedelta(
    df["Datum"].dt.dayofweek, unit="D"
)

# Calendar-year week number, starting at KW01
df["KW_Num"] = df["Week_Start"].dt.strftime("%W").astype(int) + 1

df["KW_Label"] = (
    df["Week_Start"].dt.year.astype(str)
    + "-KW"
    + df["KW_Num"].astype(str).str.zfill(2)
)

latest_week = df["Week_Start"].max()
cutoff = latest_week - pd.Timedelta(weeks=4)

df_last5 = df[df["Week_Start"] >= cutoff]
# This always gives you exactly the most recent five weeks, even across years.
# No week numbers involved. Time flows forward like it should.

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
    df_last5[df_last5["Lieferroute"].isin(routes)]
    .groupby(["Week_Start", "KW_Label", "Lieferroute"], as_index=False)["Umsatzverlust_EUR"]
    .mean()
)

df_plot_verz = (
    df_last5[df_last5["Lieferroute"].isin(routes)]
    .groupby(["Week_Start", "KW_Label", "Lieferroute"], as_index=False)["Verzoegerung_Min"]
    .mean()
)


st.subheader("Mittelwert von Umsatzverlust pro Kalenderwoche")
base_umsatz = (
    alt.Chart(df_plot)
    .encode(
        x=alt.X(
            "Week_Start:T",
            title="Kalenderwoche",
            axis=alt.Axis(
                labelExpr="datum['KW_Label']"
            )
        ),
        y=alt.Y("Umsatzverlust_EUR:Q", title="Umsatzverlust (Euro)"),
        color=alt.Color("Lieferroute:N", title="Lieferroute")
    )
)

chart_umsatz = (
    base_umsatz.mark_line()
    + base_umsatz.mark_point(size=80)
)
st.altair_chart(chart_umsatz, use_container_width=True)


st.subheader("Mittelwert von Verzoegerung pro Kalenderwoche")
base_verz = (
    alt.Chart(df_plot)
    .encode(
        x=alt.X(
            "Week_Start:T",
            title="Kalenderwoche",
            axis=alt.Axis(
                labelExpr="datum['KW_Label']"
            )
        ),
        y=alt.Y("Verzoegerung_Min:Q", title="Verzögerung (Minuten)"),
        color=alt.Color("Lieferroute:N", title="Lieferroute")
    )
)

chart_verz = (
    base_verz.mark_line()
    + base_verz.mark_point(size=80)
)
st.altair_chart(chart_verz, use_container_width=True)


st.markdown("Last updated 07.02.26", text_alignment="left")

# Whenever you change the code:
# In VS Code:
# git add .
# git commit -m "Describe your change"
# git push
