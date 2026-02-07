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
# pivot = df_plot.pivot(
#     index="Kalenderwoche",
#     columns="Lieferroute",
#     values="Umsatzverlust_EUR",
# ).sort_index()

# st.line_chart(pivot)


chart_umsatz = (
    alt.Chart(df_plot)
    .mark_line()
    .encode(
        x=alt.X(
            "Kalenderwoche:O",
            title="Kalenderwoche"
        ),
        y=alt.Y(
            "Umsatzverlust_EUR:Q",
            title="Umsatzverlust (Euro)"
        ),
        color=alt.Color(
            "Lieferroute:N",
            title="Lieferroute"
        )
    )
)

st.altair_chart(chart_umsatz, use_container_width=True)

st.subheader("Mittelwert von Verzoegerung pro Kalenderwoche")
# pivot_verz = df_plot_verz.pivot(
#     index="Kalenderwoche",
#     columns="Lieferroute",
#     values="Verzoegerung_Min",
# ).sort_index()

# st.line_chart(pivot_verz)


chart_verz = (
    alt.Chart(df_plot_verz)
    .mark_line()
    .encode(
        x=alt.X(
            "Kalenderwoche:O",
            title="Kalenderwoche"
        ),
        y=alt.Y(
            "Verzoegerung_Min:Q",
            title="Verzögerung (Minuten)"
        ),
        color=alt.Color(
            "Lieferroute:N",
            title="Lieferroute"
        )
    )
)

st.altair_chart(chart_verz, use_container_width=True)


st.markdown("Last updated 07.02.26", text_alignment="left")



# Whenever you change the code:
# In VS Code:
# git add .
# git commit -m "Describe your change"
# git push
