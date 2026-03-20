import streamlit as st

st.set_page_config(page_title="Temperature Converter", page_icon="🌡️", layout="centered")

st.title("Temperature Converter")
st.write("Convert temperatures between Celsius and Fahrenheit instantly.")


def celsius_to_fahrenheit(c):
    return (c * 9 / 5) + 32


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


conversion_mode = st.radio(
    "Choose conversion direction:",
    ("Celsius → Fahrenheit", "Fahrenheit → Celsius"),
)

input_temp = st.number_input(
    "Enter temperature:",
    value=0.0,
    format="%.2f",
    key="input_temp",
)

if conversion_mode == "Celsius → Fahrenheit":
    result = celsius_to_fahrenheit(input_temp)
    st.success(f"{input_temp:.2f} °C = {result:.2f} °F")
else:
    result = fahrenheit_to_celsius(input_temp)
    st.success(f"{input_temp:.2f} °F = {result:.2f} °C")

st.markdown("---")
st.write("**How it works**")
if conversion_mode == "Celsius → Fahrenheit":
    st.latex(r"T_{F} = T_{C} \times \frac{9}{5} + 32")
else:
    st.latex(r"T_{C} = (T_{F} - 32) \times \frac{5}{9}")

st.write("Powered by Python and Streamlit.")

# for pushing to GitHub:
# git add .
# git commit -m "Describe your change"
# git push