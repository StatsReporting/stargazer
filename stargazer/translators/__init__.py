def register_class(klass, extractor):
    from ..stargazer import RESULTS_CLASSES

    RESULTS_CLASSES[klass] = extractor


### STATSMODELS ###
from .statsmodels import classes

### LINEARMODELS ###
try:
    from .linearmodels import classes
except ModuleNotFoundError:
    # linearmodels not found
    pass
