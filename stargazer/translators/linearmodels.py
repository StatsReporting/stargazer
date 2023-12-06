"""
Compatibility layer with results from linearmodels.
"""

from math import sqrt

from ..starlib import _extract_feature

from linearmodels.panel.results import PanelEffectsResults, RandomEffectsResults, PanelResults

from . import register_class

# For features that are simple attributes of "model", establish the
# mapping with internal name:
# Mapping for linearmodels key parameters
# between-, within- and overall-R² values extracted for potential future stats display in Stargazer
# Note: We here use corr_... attributes for the R² as this matches the results obtained in Stata
linearmodels_map = {
    "p_values": "pvalues",
    "cov_values": "params",
    "cov_std_err": "std_errors",
    "r2": "rsquared",
    "degree_freedom": "df_model",
    "degree_freedom_resid": "df_resid",
    "nobs": "nobs",
    "between_r2": "corr_squared_between",
    "within_r2": "corr_squared_within",
    "overall_r2": "corr_squared_overall",
}


def extract_model_data(model):
    data = {}
    for key, val in linearmodels_map.items():
        data[key] = _extract_feature(model, val)

    data['dependent_variable'] = model.summary.tables[0].data[0][1]
    data['cov_names'] = model.params.index.values
    
    # Extract stats that are not attributes of PanelEffectsResults
    data['conf_int_low_values']  = model.conf_int().lower
    data['conf_int_high_values'] = model.conf_int().upper
    data['resid_std_err']   = sqrt(model.model_ss / model.df_resid)
    data['f_statistic'] =model.f_statistic.stat
    data['f_p_value']   =model.f_statistic.pval
    data['ngroups']     =str(int(model.entity_info.total))
    
    # Set remaining stats that stargazer.py requires 
    data['r2_adj'] = None
    data['pseudo_r2'] = None
    
    return data


classes = [
    (PanelEffectsResults, extract_model_data),
    (RandomEffectsResults, extract_model_data),
    (PanelResults, extract_model_data),
]

for klass, translator in classes:
    register_class(klass, translator)
