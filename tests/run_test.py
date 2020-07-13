import unittest

import pandas as pd
import statsmodels.formula.api as smf

from stargazer.stargazer import Stargazer, LineLocation


class StargazerTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(list(zip(range(9), range(0, 18, 2))),  columns =['a', 'b'])
        self.est1 = smf.ols('a ~ 0 + b', self.df).fit()
        self.est2 = smf.ols('a ~ 1 + b', self.df).fit()
        self.stargazer = Stargazer([self.est1, self.est2])

    def test_add_line(self):
        # too few arguments
        self.assertRaises(AssertionError, self.stargazer.add_line, '', [0])

        # wrong location
        self.assertRaises(ValueError, self.stargazer.add_line, '', [0, 0], 'wrong')

        # correct usage
        for loc in LineLocation:
            self.stargazer.add_line(f'test {loc.value}', ['N/A', 'N/A'], loc)
        latex = self.stargazer.render_latex()
        for loc in LineLocation:
            self.assertIn(f' test {loc.value} & N/A & N/A \\', latex)

    def test_render_latex(self):
        # test escaping
        self.stargazer.rename_covariates({'b': 'b_'})
        self.assertIn(' b_ ', self.stargazer.render_latex())
        self.assertIn(r' b\_ ', self.stargazer.render_latex(escape=True))


if __name__ == '__main__':
    unittest.main()
