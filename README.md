# Stargazer

This is a python port of the R stargazer package that can be found [on CRAN](https://CRAN.R-project.org/package=stargazer). I was disappointed that there wasn't equivalent functionality in any python packages I was aware of so I'm re-implementing it here.

There is an experimental function in the [statsmodels.regression.linear_model.OLSResults.summary2](http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.summary2.html) that can report single regression model results in HTML/CSV/LaTeX/etc, but it still didn't quite fulfill what I was looking for.

The python package is object oriented now with chained commands to make changes to the rendering parameters, which is hopefully more pythonic and the user doesn't have to put a bunch of arguments in a single function.

## Installation

You can install this package through PyPi with `pip install stargazer` or just clone the repo and take the `stargazer.py` file since it's the only one in the package.

### Dependencies

It depends on `statsmodels`, which in turn depends on several other libraries like `pandas`, `numpy`, etc

## Editing Features

This library implements many of the customization features found in the original package. Examples of most can be found [in the examples jupyter notebook](https://github.com/StatsReporting/stargazer/blob/master/examples.ipynb) and a full list of the methods/features is here below:

* `title`: custom title
* `show_header`: display or hide model header data
* `show_model_numbers`: display or hide model numbers
* `custom_columns`: custom model names and model groupings
* `significance_levels`: change statistical significance thresholds
* `significant_digits`: change number of significant digits
* `show_confidence_intervals`: display confidence intervals instead of variance
* `dependent_variable_name`: rename dependent variable
* `rename_covariates`: rename covariates
* `covariate_order`: reorder covariates
* `reset_covariate_order`: reset covariate order to original ordering
* `show_degrees_of_freedom`: display or hide degrees of freedom
* `custom_note_label`: label notes section at bottom of table
* `add_custom_notes`: add custom notes to section at bottom of the table
* `add_line`: add a custom line to the table
* `append_notes`: display or hide statistical significance thresholds

These features are agnostic of the rendering type and will be applied whether the user outputs in HTML, LaTeX, etc

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
```

### HTML Example

```python
stargazer.render_html()
```

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable: target</em></td></tr><tr><td style="text-align:left"></td><tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr>

<tr><td style="text-align:left">ABP</td><td>416.674<sup>***</sup></td><td>397.583<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(69.495)</td><td>(70.870)</td></tr>
<tr><td style="text-align:left">Age</td><td>37.241<sup></sup></td><td>24.704<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td>(64.117)</td><td>(65.411)</td></tr>
<tr><td style="text-align:left">BMI</td><td>787.179<sup>***</sup></td><td>789.742<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(65.424)</td><td>(66.887)</td></tr>
<tr><td style="text-align:left">S1</td><td></td><td>197.852<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(143.812)</td></tr>
<tr><td style="text-align:left">S2</td><td></td><td>-169.251<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(142.744)</td></tr>
<tr><td style="text-align:left">Sex</td><td>-106.578<sup>*</sup></td><td>-82.862<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td>(62.125)</td><td>(64.851)</td></tr>
<tr><td style="text-align:left">const</td><td>152.133<sup>***</sup></td><td>152.133<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(2.853)</td><td>(2.853)</td></tr>

<td colspan="3" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align: left">Observations</td><td>442</td><td>442</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.400</td><td>0.403</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.395</td><td>0.395</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>59.976 (df=437)</td><td>59.982 (df=435)</td></tr><tr><td style="text-align: left">F Statistic</td><td>72.913<sup>***</sup> (df=4; 437)</td><td>48.915<sup>***</sup> (df=6; 435)</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td><td colspan="2" style="text-align: right"><sup>*</sup>p&lt;0.1; <sup>**</sup>p&lt;0.05; <sup>***</sup>p&lt;0.01</td></tr></table>

### LaTeX Example

```python
stargazer.render_latex()
```

![](https://raw.githubusercontent.com/StatsReporting/stargazer/master/latex_example.png)
