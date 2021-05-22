# latex_example.py
# Simple code to produce a latex table

import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

# Standard example, except here I use three models, not two
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data)
df.columns = ['Age', 'Sex', 'BMI', 'ABP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
df['target'] = diabetes.target

est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()
est3 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:8]])).fit()

stargazer = Stargazer([est, est2, est3])

# Produce a latex table
stargazer_latex = stargazer.render_latex()

# Write the latex to file
with open("latex_example.tex", "w") as f:
    f.write(stargazer_latex)