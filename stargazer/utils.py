from statsmodels.base.wrapper import ResultsWrapper
import numpy as np

class Wrapper(ResultsWrapper):
    """
    This class can be subclassed in order to easily alter (override) any aspect
    of any given model, before it is passed to a Stargazer table.
    It requires the original model as input and stores it as "_base" attribute.
    """

    def __init__(self, base):
        self._base = base

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        return getattr(self._base, name)

    def summary(self):
        raise NotImplementedError('WrapperOverrider is unable to produce a '
                                  'summary. Please override "summary()" if you'
                                  'think this subclass should be able to.')


class LogitOdds(Wrapper):
    """
    Just exponentiate parameters and confidence intervals.
    """

    def conf_int(self):
        return np.exp(self._base.conf_int())

    @property
    def params(self):
        return np.exp(self._base.params)
