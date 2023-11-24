"""
Compatibility layer with results from statsmodels.
"""

from math import sqrt

from ..starlib import _extract_feature

from statsmodels.base.wrapper import ResultsWrapper
from statsmodels.regression.linear_model import RegressionResults

from . import register_class

# For features that are simple attributes of "model", establish the
# mapping with internal name:

statsmodels_map = {'p_values' : 'pvalues',
                   'cov_values' : 'params',
                   'cov_std_err' : 'bse',
                   'r2' : 'rsquared',
                   'r2_adj' : 'rsquared_adj',
                   'pseudo_r2' : 'prsquared',
                   'f_p_value' : 'f_pvalue',
                   'degree_freedom' : 'df_model',
                   'degree_freedom_resid' : 'df_resid',
                   'nobs' : 'nobs',
                   'f_statistic' : 'fvalue'
                   }

def extract_model_data(model):

    data = {}
    for key, val in statsmodels_map.items():
        data[key] = _extract_feature(model, val)

    data['dependent_variable'] = model.model.endog_names

    if isinstance(model, ResultsWrapper):
        data['cov_names'] = model.params.index.values
    else:
        # Simple RegressionResults, for instance as a result of
        # get_robustcov_results():
        data['cov_names'] = model.model.data.orig_exog.columns

        # These are simple arrays, not Series:
        for what in 'cov_values', 'p_values', 'cov_std_err':
            data[what] = pd.Series(data[what],
                                   index=data['cov_names'])

    data['conf_int_low_values'] = model.conf_int()[0]
    data['conf_int_high_values'] = model.conf_int()[1]
    data['resid_std_err'] = (sqrt(sum(model.resid**2) / model.df_resid)
                             if hasattr(model, 'resid') else None)

    # Workaround for
    # https://github.com/statsmodels/statsmodels/issues/6778:
    if 'f_statistic' in data:
        data['f_statistic'] = (lambda x : x[0, 0] if getattr(x, 'ndim', 0)
                               else x)(data['f_statistic'])

    return data


classes = [(ResultsWrapper, extract_model_data),
           (RegressionResults, extract_model_data),
           ]

for klass, translator in classes:
    register_class(klass, translator)

