import streamlit as st
import pandas as pd
import altair as alt

st.title("Dashboard zur wöchentlichen Kontrolle")

# Example: loading from Excel
df = pd.read_excel("kaffeekette_logistik_daten1.xlsx")

df["Datum"] = pd.to_datetime(df["Datum"])

df["Week_Start"] = df["Datum"] - pd.to_timedelta(df["Datum"].dt.weekday, unit="D")
# What this does:
# dt.weekday → Monday = 0, Sunday = 6
# subtracting that many days snaps every date to Monday of its week
# So:
# Datum	Week_Start
# 2025-12-29	2025-12-29
# 2026-01-02	2025-12-29
# Same week. No reset. No ambiguity.

latest_week = df["Week_Start"].max()
cutoff = latest_week - pd.Timedelta(weeks=3)

df_last4 = df[df["Week_Start"] >= cutoff]
# This always gives you exactly the most recent four weeks, even across years.
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
    df_last4[df_last4["Lieferroute"].isin(routes)]
    .groupby(["Week_Start", "Lieferroute"], as_index=False)["Umsatzverlust_EUR"]
    .mean()
)

df_plot_verz = (
    df_last4[df_last4["Lieferroute"].isin(routes)]
    .groupby(["Week_Start", "Lieferroute"], as_index=False)["Verzoegerung_Min"]
    .mean()
)


st.subheader("Mittelwert von Umsatzverlust pro Kalenderwoche")

chart_umsatz = (
    alt.Chart(df_plot)
    .mark_line()
    .encode(
        x=alt.X(
            "Week_Start:T",
            timeUnit= "yearweek",
            title="Kalenderwoche",
            axis=alt.Axis(format="%Y-KW%W")
        ),
        y=alt.Y(
            "Umsatzverlust_EUR:Q",
            title="Umsatzverlust (Euro)"
        ),
        color=alt.Color("Lieferroute:N", title="Lieferroute")
    )
)

st.altair_chart(chart_umsatz, use_container_width=True)



st.subheader("Mittelwert von Verzoegerung pro Kalenderwoche")

chart_verz = (
    alt.Chart(df_plot_verz)
    .mark_line()
    .encode(
        x=alt.X(
            "Week_Start:T",
            timeUnit= "yearweek",
            title="Kalenderwoche",
            axis=alt.Axis(format="%Y-KW%W")
        ),
        y=alt.Y(
            "Verzoegerung_Min:Q",
            title="Verzögerung (Minuten)"
        ),
        color=alt.Color("Lieferroute:N", title="Lieferroute")
    )
)

st.altair_chart(chart_verz, use_container_width=True)


st.markdown("Last updated 07.02.26", text_alignment="left")

# Whenever you change the code:
# In VS Code:
# git add .
# git commit -m "Describe your change"
# git push
