# Stargazer

This is a python port of the R stargazer package that can be found [on CRAN](https://CRAN.R-project.org/package=stargazer). I was disappointed that there wasn't equivalent functionality in any python packages I was aware of so I'm re-implementing it here.

There is an experimental function in the [statsmodels.regression.linear_model.OLSResults.summary2](http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.summary2.html) that can report single regression model results in HTML/CSV/LaTeX/etc, but it still didn't quite fulfill what I was looking for.

The python package is object oriented now with chained commands to make changes to the rendering parameters, which is hopefully more pythonic and the user doesn't have to put a bunch of arguments in a single function.

I'm a data scientist, not a software engineer so please don't crucify me over my bad code. Thanks in advance :D

## Installation

You can install this package with `pip install stargazer`. It depends on `statsmodels`, which in turn depends on several other libraries like `pandas`, `numpy`, etc. It's still in development so there may be bugs and only HTML output is supported currently. 

Future release should (hopefully) include LaTeX and Markdown support. 

## Example

Here is an examples of how to quickly get started with the library. More examples can be found in the `examples.ipynb` file in the github repo. The examples all use the scikit-learn diabetes dataset, but it is not a dependency for the package.

### OLS Models Preparation

```python
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data)
df.columns = ['Age', 'Sex', 'BMI', 'ABP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
df['target'] = diabetes.target

est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()


stargazer = Stargazer([est, est2])

stargazer.render_html()
```

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable:</em></td></tr><tr><td style="text-align:left"><tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr><td colspan="3" style="border-bottom: 1px solid black"><tr><td style="text-align:left">ABP</td><td>416.674<sup>***</sup></td><td>397.583<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(69.495)</td><td>(70.87)</td></tr><tr><td style="text-align:left">Age</td><td>37.241<sup></sup></td><td>24.704<sup></sup></td></tr><tr><td style="text-align:left"></td><td>(64.117)</td><td>(65.411)</td></tr><tr><td style="text-align:left">BMI</td><td>787.179<sup>***</sup></td><td>789.742<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(65.424)</td><td>(66.887)</td></tr><tr><td style="text-align:left">S1</td><td></td><td>197.852<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(143.812)</td></tr><tr><td style="text-align:left">S2</td><td></td><td>-169.251<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(142.744)</td></tr><tr><td style="text-align:left">Sex</td><td>-106.578<sup>*</sup></td><td>-82.862<sup></sup></td></tr><tr><td style="text-align:left"></td><td>(62.125)</td><td>(64.851)</td></tr><tr><td style="text-align:left">const</td><td>152.133<sup>***</sup></td><td>152.133<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(2.853)</td><td>(2.853)</td></tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Observations</td><td>442.0</td><td>442.0</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.4</td><td>0.403</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.395</td><td>0.395</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>59.976(df = 437.0)</td><td>59.982(df = 435.0)</td></tr><tr><td style="text-align: left">F Statistic</td><td>72.913<sup>***</sup>(df = 4.0; 437.0)</td><td>48.915<sup>***</sup>(df = 6.0; 435.0)</td></tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td><td colspan="2" style="text-align: right"><em>p<0.1</em>; <b>p<0.05</b>; p<0.01</td></tr></table>
