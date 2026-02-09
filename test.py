import pandas as pd
import pygwalker as pyg

df = pd.read_csv('kaffeekette_logistik_daten.csv')
walker = pyg.walk(df)