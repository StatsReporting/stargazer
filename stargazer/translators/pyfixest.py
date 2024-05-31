"""
Compatibility layer with results from pyfixest.
"""

from ..starlib import _extract_feature

from . import register_class
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.fepois_ import Fepois

# For features that are simple attributes of "model", establish the
# mapping with internal name:

pyfixest_map = {
                   'r2' : '_r2',
                   'cov_names' : '_coefnames',
                   'nobs' : '_N',
                   }

def extract_model_data(model):

    data = {}
    for key, val in pyfixest_map.items():
        data[key] = _extract_feature(model, val)

    data['dependent_variable'] = model._fml.split("~")[0].replace(" ", "")
    data['cov_values'] = model.coef()
    data['cov_std_err'] = model.se()
    data['p_values'] = model.pvalue()

    data['resid_std_err'] = None
    data['degree_freedom_resid'] = None
    data["f_statistic"] = None
    data["f_p_value"] = None
    data["r2_adj"] = None
    data["pseudo_r2"] = None


    #data['conf_int_low_values'] = model._conf_int[:,0]
    #data['conf_int_high_values'] = model._conf_int[:,1]

    return data


classes = [(Feols, extract_model_data),
           (Fepois, extract_model_data),
           (Feiv, extract_model_data)
           ]

for klass, translator in classes:
    register_class(klass, translator)

