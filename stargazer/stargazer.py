"""
This is a python package to generate nicely formatted
regression results similar to the style of the R
package of the same name:
https://CRAN.R-project.org/package=stargazer

@authors:
    Pietro Battiston
        me@pietrobattiston.it
        https://pietrobattiston.it
    Matthew Burke:
        matthew.wesley.burke@gmail.com
        github.com/mwburke
"""

from statsmodels.base.wrapper import ResultsWrapper
from statsmodels.regression.linear_model import RegressionResults
from math import sqrt
from collections import defaultdict
from enum import Enum
import numbers
import pandas as pd

class LineLocation(Enum):
    HEADER_TOP = 'ht'
    HEADER_BOTTOM = 'hb'
    BODY_TOP = 'bt'
    BODY_BOTTOM = 'bb'
    FOOTER_TOP = 'ft'
    FOOTER_BOTTOM = 'fb'


class Stargazer:
    """
    Class that is constructed with one or more trained
    OLS models from the statsmodels package.

    The user then can change the rendering options by
    chaining different methods to the Stargazer object
    and then render the results in either HTML or LaTeX.
    """

    # This is a mapping from 'show_*' attribute to name of generating method
    # "_generate_{LABEL}" (if present) and to name of stat in data store
    # otherwise.
    # Stats will be automatically formatted. Order matters!
    _auto_stats = [('n', 'nobs'),
                   ('r2', 'r2'),
                   ('adj_r2', 'r2_adj'),
                   ('residual_std_err', 'resid_std_err'),
                   ('f_statistic', 'f_statistic')]

    def __init__(self, models):
        self.models = models
        self.num_models = len(models)
        self.reset_params()
        self.extract_data()

    def validate_input(self):
        """
        Check inputs to see if they are going to
        cause any problems further down the line.

        Any future checking will be added here.
        """
        targets = []

        for m in self.models:
            if not isinstance(m, (ResultsWrapper,
                                  RegressionResults)):
                raise ValueError('Please use trained OLS models as inputs')
            targets.append(m.model.endog_names)

        if targets.count(targets[0]) != len(targets):
            self.dependent_variable = ''
            self.dep_var_name = None
        else:
            self.dependent_variable = targets[0]

    def reset_params(self):
        """
        Set all of the rendering parameters to their default settings.
        Run upon initialization but also allows the user to reset
        if they have made several changes and want to start fresh.

        Does not effect any of the underlying model data.
        """
        self.title_text = None
        self.show_header = True
        self.dep_var_name = 'Dependent variable: '
        self.column_labels = None
        self.column_separators = None
        self.show_model_nums = True
        self.original_cov_names = None
        self.cov_map = None
        self.cov_spacing = None
        self.show_precision = True
        self.show_sig = True
        self.sig_levels = [0.1, 0.05, 0.01]
        self.sig_digits = 3
        self.confidence_intervals = False
        self.show_footer = True
        self.custom_lines = defaultdict(list)
        self.show_n = True
        self.show_r2 = True
        self.show_adj_r2 = True
        self.show_residual_std_err = True
        self.show_f_statistic = True
        self.show_dof = True
        self.show_notes = True
        self.notes_label = 'Note:'
        self.notes_append = True
        self.custom_notes = []
        self.show_stars = True
        self.table_label = None

    def extract_data(self):
        """
        Extract the values we need from the models and store
        for use or modification. They should not be able to
        be modified by any rendering parameters.
        """
        self.validate_input()
        self.model_data = []
        for m in self.models:
            self.model_data.append(self.extract_model_data(m))

        covs = []
        for md in self.model_data:
            covs = covs + list(md['cov_names'])
        self.cov_names = sorted(set(covs))

    def _extract_feature(self, obj, feature):
        """
        Just return obj.feature if present and None otherwise.
        """
        try:
            return getattr(obj, feature)
        except AttributeError:
            return None

    def extract_model_data(self, model):
        # For features that are simple attributes of "model", establish the
        # mapping with internal name (TODO: adopt same names?):
        statsmodels_map = {'p_values' : 'pvalues',
                           'cov_values' : 'params',
                           'cov_std_err' : 'bse',
                           'r2' : 'rsquared',
                           'r2_adj' : 'rsquared_adj',
                           'f_p_value' : 'f_pvalue',
                           'degree_freedom' : 'df_model',
                           'degree_freedom_resid' : 'df_resid',
                           'nobs' : 'nobs',
                           'f_statistic' : 'fvalue'
                           }

        data = {}
        for key, val in statsmodels_map.items():
            data[key] = self._extract_feature(model, val)

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

    # Begin render option functions
    def title(self, title):
        self.title_text = title

    def show_header(self, show):
        assert type(show) == bool, 'Please input True/False'
        self.header = show

    def show_model_numbers(self, show):
        assert type(show) == bool, 'Please input True/False'
        self.show_model_nums = show

    def custom_columns(self, labels, separators=None):
        """
        "labels": list of labels, or single label string.
        "separators" (optional): list of integers, of same length of "labels",
        indicating how many columns each header covers (default is 1).
        """
        if isinstance(labels, list):
            if separators is None:
                assert(len(labels) == self.num_models), ('If separators are '
                'not provided, custom headers must be the same number as '
                'models,')
                separators = [1] * self.num_models
            else:
                assert type(separators) == list, ('Please input a list of '
                                                  'column separators.')
                assert len(labels) == len(separators), ('Number of labels '
                                               'must match number of columns.')
                assert(all(isinstance(s, int) for s in separators)), ('Column '
                                                   'numbers must be integers.')
                assert sum(separators) == self.num_models, ('Please set total '
                                      'number of columns to number of models.')
        else:
            assert isinstance(labels, str), ('Please input a single string '
                                             'label, or a list of strings.')

        self.column_labels = labels
        self.column_separators = separators

    def significance_levels(self, levels):
        assert len(levels) == 3, 'Please input 3 significance levels'
        assert sum([int(type(l) != float) for l in levels]) == 0, 'Please input floating point values as significance levels'
        self.sig_levels = sorted(levels, reverse=True)

    def significant_digits(self, digits):
        assert type(digits) == int, 'The number of significant digits must be an int'
        assert digits < 10, 'Whoa hold on there bud, maybe use fewer digits'
        self.sig_digits = digits

    def show_confidence_intervals(self, show):
        assert type(show) == bool, 'Please input True/False'
        self.confidence_intervals = show

    def dependent_variable_name(self, name):
        assert type(name) == str, 'Please input a string to use as the depedent variable name'
        self.dependent_variable = name

    def covariate_order(self, cov_names):
        missing = set(cov_names).difference(set(self.cov_names))
        assert not missing, ('Covariate order must contain subset of existing '
                             'covariates: {} are not.'.format(missing))
        self.original_cov_names = self.cov_names
        self.cov_names = cov_names

    def rename_covariates(self, cov_map):
        assert isinstance(cov_map, dict), 'Please input a dictionary with covariate names as keys'
        self.cov_map = cov_map

    def reset_covariate_order(self):
        if self.original_cov_names is not None:
            self.cov_names = self.original_cov_names

    def add_line(self, label, values, location=LineLocation.BODY_BOTTOM):
        """
        Add a custom line to the table.

        At each location, lines are added in the order at which this method is called.
        To remove lines, modify the custom_lines[location] attribute.

        Parameters
        ----------
        label : str
            Name of the new line (left-most column).
        values : list of str
            List containing the custom content (one item per model).
        location : LineLocation or str
            Location at which to add the line. See list(LineLocation) for valid values.
        """
        assert len(values) == self.num_models, \
            'values has to be an iterables with {} elements (one for each model)'.format(self.num_models)
        if type(location) != LineLocation:
            location = LineLocation(location)
        self.custom_lines[location].append([label] + values)

    def show_degrees_of_freedom(self, show):
        assert type(show) == bool, 'Please input True/False'
        self.show_dof = show

    def custom_note_label(self, notes_label):
        assert type(notes_label) == str, 'Please input a string to use as the note label'
        self.notes_label = notes_label

    def add_custom_notes(self, notes):
        assert sum([int(type(n) != str) for n in notes]) == 0, 'Notes must be strings'
        self.custom_notes = notes

    def append_notes(self, append):
        assert type(append) == bool, 'Please input True/False'
        self.notes_append = append

    def render_html(self, *args, **kwargs):
        return HTMLRenderer(self).render(*args, **kwargs)

    def _repr_html_(self):
        return self.render_html()

    def render_latex(self, *args, escape=False, **kwargs):
        """
        Render as LaTeX code.

        Parameters
        ----------
        escape : bool
            Escape special characters.

        Returns
        -------
        str
            The LaTeX code.
        """
        return LaTeXRenderer(self, escape=escape).render(*args, **kwargs)


class Renderer:
    """
    Base class for renderers to specific formats. Only meant to be subclassed.
    """

    # Formatters for stats which are not formatted via Renderer._float_format()
    _formatters = {'nobs' : lambda x : str(int(x))}

    def __init__(self, table, **kwargs):
        """
        Initialize a new renderer.
        
        "table": Stargazer object to render
        """

        self.table = table
        self.kwargs = kwargs

    def __getattribute__(self, key):
        """
        Temporary fix while we better organize how a Stargazer table stores
        parameters: just retrieve them transparently as attributes of the
        Stargazer table object.
        """

        try:
            return object.__getattribute__(self, key)
        except AttributeError as exc:
            if hasattr(self.table, key):
                return getattr(self.table, key)
            else:
                raise exc

    def get_sig_icon(self, p_value, sig_char='*'):
        if p_value is None or not self.show_stars:
            return ''
        if p_value >= self.sig_levels[0]:
            return ''
        elif p_value >= self.sig_levels[1]:
            return sig_char
        elif p_value >= self.sig_levels[2]:
            return sig_char * 2
        else:
            return sig_char * 3

    def _generate_cov_spacing(self):
        if self.cov_spacing is None:
            return None
        if isinstance(self.cov_spacing, numbers.Number):
            # A number is interpreted in "em" by default:
            return f'{self.cov_spacing}em'
        else:
            return self.cov_spacing

    def _float_format(self, value):
        """
        Format value to string, using the precision set by the user.
        """
        if value is None:
            return ''

        return '{{:.{prec}f}}'.format(prec=self.sig_digits).format(value)

    def _generate_resid_std_err(self, md):
        rse = md['resid_std_err']
        if rse is None:
            return None

        rse_text = self._float_format(rse)
        if self.show_dof:
            rse_text += ' (df={degree_freedom_resid:.0f})'.format(**md)
        return rse_text

    def _generate_f_statistic(self, md):
        f_stat = md['f_statistic']
        if f_stat is None:
            return None

        f_stars = self._format_sig_icon(md['f_p_value'])
        f_text = f'{self._float_format(f_stat)}{f_stars}'
        if self.show_dof:
            f_text += (' (df={degree_freedom:.0f}; '
                       '{degree_freedom_resid:.0f})').format(**md)

        return f_text

    def _generate_stat_values(self, stat):
        if hasattr(self, f'_generate_{stat}'):
            generator = getattr(self, f'_generate_{stat}')
            return [generator(md) for md in self.model_data]
        else:
            return [md[stat] for md in self.model_data]

class HTMLRenderer(Renderer):
    fmt = 'html'

    # Labels for stats in Stargazer._auto_stats:
    _stats_labels = {'n' : 'Observations',
                     'r2' : 'R<sup>2</sup>',
                     'adj_r2' : 'Adjusted R<sup>2</sup>',
                     'residual_std_err' : 'Residual Std. Error',
                     'f_statistic' : 'F Statistic'}

    def render(self):
        html = self.generate_header()
        html += self.generate_body()
        html += self.generate_footer()
        return html

    def generate_header(self):
        header = ''
        if not self.show_header:
            return header

        if self.title_text is not None:
            header += self.title_text + '<br>'

        header += '<table style="text-align:center"><tr><td colspan="'
        header += str(self.num_models + 1) + '" style="border-bottom: 1px solid black"></td></tr>'
        header += self.generate_custom_lines(LineLocation.HEADER_TOP)

        if self.dep_var_name is not None:
            header += '<tr><td style="text-align:left"></td><td colspan="' + str(self.num_models)
            header += '"><em>' + self.dep_var_name + self.dependent_variable + '</em></td></tr>'

        header += '<tr><td style="text-align:left"></td>'
        if self.column_labels is not None:
            if type(self.column_labels) == str:
                header += '<td colspan="' + str(self.num_models) + '">'
                header += self.column_labels + "</td></tr>"
            else:
                # The first table column holds the covariates names:
                header += '<tr><td></td>'
                for i, label in enumerate(self.column_labels):
                    sep = self.column_separators[i]
                    header += '<td colspan="{}">{}</td>'.format(sep, label)
                header += '</tr>'

        if self.show_model_nums:
            header += '<tr><td style="text-align:left"></td>'
            for num in range(1, self.num_models + 1):
                header += '<td>(' + str(num) + ')</td>'
            header += '</tr>'

        header += self.generate_custom_lines(LineLocation.HEADER_BOTTOM)

        header += '<tr><td colspan="' + str(self.num_models + 1)
        header += '" style="border-bottom: 1px solid black"></td></tr>\n'

        return header

    def _generate_cov_style(self):
        if self.cov_spacing is None:
            return ''
        spacing = self._generate_cov_spacing()
        return f' style="padding-bottom:{spacing}"'

    def _format_sig_icon(self, pvalue):
        return '<sup>' + str(self.get_sig_icon(pvalue)) + '</sup>'

    def generate_body(self):
        """
        Generate the body of the results where the
        covariate reporting is.
        """

        spacing = self._generate_cov_style()

        body = ''
        body += self.generate_custom_lines(LineLocation.BODY_TOP)
        for cov_name in self.cov_names:
            body += self.generate_cov_rows(cov_name, spacing)
        body += self.generate_custom_lines(LineLocation.BODY_BOTTOM)

        return body

    def generate_cov_rows(self, cov_name, spacing):
        cov_text = ''
        main_spacing = spacing if not self.show_precision else ''
        cov_text += self.generate_cov_main(cov_name, spacing=main_spacing)
        if self.show_precision:
            cov_text += self.generate_cov_precision(cov_name, spacing=spacing)
        else:
            cov_text += '<tr></tr>'

        return cov_text

    def generate_cov_main(self, cov_name, spacing):
        cov_print_name = cov_name
        if self.cov_map is not None:
            cov_print_name = self.cov_map.get(cov_print_name, cov_name)
        cov_text = (f'<tr><td style="text-align:left">'
                    f'{cov_print_name}</td>')
        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += f'<td{spacing}>'
                cov_text += self._float_format(md['cov_values'][cov_name])
                if self.show_sig:
                    cov_text += self._format_sig_icon(md['p_values'][cov_name])
                cov_text += '</td>'
            else:
                cov_text += f'<td{spacing}></td>'
        cov_text += '</tr>\n'

        return cov_text

    def generate_cov_precision(self, cov_name, spacing):
        # This is the only place where we need to add spacing and there's a
        # "style" already:
        space_style = (f';padding-bottom:{self._generate_cov_spacing()}'
                       if self.cov_spacing else '')
        cov_text = f'<tr><td style="text-align:left{space_style}"></td>'
        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += f'<td{spacing}>('
                if self.confidence_intervals:
                    cov_text += self._float_format(md['conf_int_low_values'][cov_name]) + ' , '
                    cov_text += self._float_format(md['conf_int_high_values'][cov_name])
                else:
                    cov_text += self._float_format(md['cov_std_err'][cov_name])
                cov_text += ')</td>'
            else:
                cov_text += f'<td{spacing}></td>'
        cov_text += '</tr>\n'

        return cov_text

    def generate_footer(self):
        """
        Generate the footer of the table where
        model summary section is.
        """
        footer = '<td colspan="' + str(self.num_models + 1) + '" style="border-bottom: 1px solid black"></td></tr>'

        if not self.show_footer:
            return footer
        footer += self.generate_custom_lines(LineLocation.FOOTER_TOP)

        for attr, stat in Stargazer._auto_stats:
            if getattr(self, f'show_{attr}'):
                footer += self.generate_stat(stat, self._stats_labels[attr])

        footer += self.generate_custom_lines(LineLocation.FOOTER_BOTTOM)
        footer += '<tr><td colspan="' + str(self.num_models + 1) + '" style="border-bottom: 1px solid black"></td></tr>'
        if self.show_notes:
            footer += self.generate_notes()
        footer += '</table>'

        return footer

    def generate_custom_lines(self, location):
        custom_text = '\n'
        for custom_row in self.custom_lines[location]:
            custom_text += '<tr><td style="text-align: left">' + str(custom_row[0]) + '</td>'
            for custom_column in custom_row[1:]:
                custom_text += '<td>' + str(custom_column) + '</td>'
            custom_text += '</tr>'
        return custom_text

    def generate_stat(self, stat, label):
        values = self._generate_stat_values(stat)
        if not any(values):
            return ''

        formatter = self._formatters.get(stat, self._float_format)

        text = f'<tr><td style="text-align: left">{label}</td>'
        for value in values:
            if not isinstance(value, str):
                value = formatter(value)
            text += f'<td>{value}</td>'
        text += '</tr>'
        return text

    def generate_notes(self):
        notes_text = ''
        notes_text += '<tr><td style="text-align: left">' + self.notes_label + '</td>'
        if self.notes_append and self.show_stars:
            notes_text += self.generate_p_value_section()
        notes_text += '</tr>'
        notes_text += self.generate_additional_notes()
        return notes_text

    def generate_p_value_section(self):
        notes_text = f'<td colspan="{self.num_models}" style="text-align: right">'
        pval_cells = [self._format_sig_icon(self.sig_levels[idx] - 0.001)
                      + 'p&lt;' + str(self.sig_levels[idx]) for idx in range(3)]
        notes_text += '; '.join(pval_cells)
        notes_text += '</td>'
        return notes_text

    def generate_additional_notes(self):
        notes_text = ''
        if len(self.custom_notes) == 0:
            return notes_text
        i = 0
        for i, note in enumerate(self.custom_notes):
            if (i != 0) | (self.notes_append):
                notes_text += '<tr>'
            notes_text += '<td colspan="' + str(self.num_models+1) + '" style="text-align: right">' + note + '</td></tr>'

        return notes_text

class LaTeXRenderer(Renderer):
    fmt = 'LaTeX'

    # Labels for stats in Stargazer._auto_stats:
    _stats_labels = {'n' : 'Observations',
                     'r2' : '$R^2$',
                     'adj_r2' : 'Adjusted $R^2$',
                     'residual_std_err' : 'Residual Std. Error',
                     'f_statistic' : 'F Statistic'}

    # LaTeX escape characters, borrowed from pandas.io.formats.latex
    _ESCAPE_CHARS = [
        ('\\', r'\textbackslash '),
        ('_', r'\_'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde '),
        ('^', r'\textasciicircum '),
        ('&', r'\&')
    ]

    def _escape(self, text):
        """Escape LaTeX special characters"""
        if self.kwargs.get('escape', False):
            for orig_char, escape_char in LaTeXRenderer._ESCAPE_CHARS:
                text = text.replace(orig_char, escape_char)
        return text

    def render(self, only_tabular=False, insert_empty_rows=False):
        latex = self.generate_header(only_tabular=only_tabular)
        latex += self.generate_body(insert_empty_rows=insert_empty_rows)
        latex += self.generate_footer(only_tabular=only_tabular)

        return latex

    def generate_header(self, only_tabular=False):
        header = ''
        if not only_tabular:
            header += '\\begin{table}[!htbp] \\centering\n'
            if not self.show_header:
                return header

            if self.title_text is not None:
                header += '  \\caption{' + self.title_text + '}\n'

            if self.table_label is not None:
                header += '  \\label{' + self.table_label + '}\n'

        content_columns = 'c' * self.num_models
        header += '\\begin{tabular}{@{\\extracolsep{5pt}}l' + content_columns + '}\n'
        header += '\\\\[-1.8ex]\\hline\n'
        header += '\\hline \\\\[-1.8ex]\n'
        header += self.generate_custom_lines(LineLocation.HEADER_TOP)

        if self.dep_var_name is not None:
            header += '& \\multicolumn{' + str(self.num_models) + '}{c}'
            header += '{\\textit{' + self.dep_var_name + self.dependent_variable + '}} \\\n'
            header += '\\cr \\cline{2-' + str(self.num_models + 1) + '}\n'

        if self.column_labels is not None:
            if type(self.column_labels) == str:
                header += '\\\\[-1.8ex] & \\multicolumn{' + str(self.num_models) + '}{c}{' + self._escape(self.column_labels) + '} \\\\'
            else:
                header += '\\\\[-1.8ex] '
                for i, label in enumerate(self.column_labels):
                    header += '& \\multicolumn{' + str(self.column_separators[i])
                    header += '}{c}{' + self._escape(label) + '} '
                header += ' \\\\\n'

        if self.show_model_nums:
            header += '\\\\[-1.8ex] '
            for num in range(1, self.num_models + 1):
                header += '& (' + str(num) + ') '
            header += '\\\\\n'

        header += self.generate_custom_lines(LineLocation.HEADER_BOTTOM)

        header += '\\hline \\\\[-1.8ex]\n'

        return header

    def _generate_cov_end(self):
        if self.cov_spacing is None:
            return '\\\\\n'
        spacing = self._generate_cov_spacing()
        return f'\\\\[{spacing}]\n'

    def _format_sig_icon(self, pvalue):
        return '$^{' + str(self.get_sig_icon(pvalue)) + '}$'

    def generate_body(self, insert_empty_rows=False):
        """
        Generate the body of the results where the
        covariate reporting is.
        """
        body = ''
        body += self.generate_custom_lines(LineLocation.BODY_TOP)

        cov_end = self._generate_cov_end()

        for cov_name in self.cov_names:
            body += self.generate_cov_rows(cov_name)
            if insert_empty_rows:
                body += '\\\\\n  ' + '& '*len(self.num_models)
            body += cov_end
        body += self.generate_custom_lines(LineLocation.BODY_BOTTOM)

        return body

    def generate_cov_rows(self, cov_name):
        cov_text = ''
        cov_text += self.generate_cov_main(cov_name)
        if self.show_precision:
            cov_text += self.generate_cov_precision(cov_name)
        else:
            cov_text += '& '

        return cov_text

    def generate_cov_main(self, cov_name):
        cov_print_name = cov_name

        if self.cov_map is not None:
            if cov_name in self.cov_map:
                cov_print_name = self.cov_map[cov_name]

        cov_text = ' ' + self._escape(cov_print_name) + ' '
        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += '& ' + self._float_format(md['cov_values'][cov_name])
                if self.show_sig:
                    cov_text += self._format_sig_icon(md['p_values'][cov_name])
                cov_text += ' '
            else:
                cov_text += '& '

        return cov_text

    def generate_cov_precision(self, cov_name):
        cov_text = '\\\\\n'

        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += '& ('
                if self.confidence_intervals:
                    cov_text += self._float_format(md['conf_int_low_values'][cov_name]) + ' , '
                    cov_text += self._float_format(md['conf_int_high_values'][cov_name])
                else:
                    cov_text += self._float_format(md['cov_std_err'][cov_name])
                cov_text += ') '
            else:
                cov_text += '& '

        return cov_text

    def generate_footer(self, only_tabular=False):
        """
        Generate the footer of the table where
        model summary section is.
        """

        footer = '\\hline \\\\[-1.8ex]\n'

        if not self.show_footer:
            return footer
        footer += self.generate_custom_lines(LineLocation.FOOTER_TOP)

        for attr, stat in Stargazer._auto_stats:
            if getattr(self, f'show_{attr}'):
                footer += self.generate_stat(stat, self._stats_labels[attr])

        footer += self.generate_custom_lines(LineLocation.FOOTER_BOTTOM)
        footer += '\\hline\n\\hline \\\\[-1.8ex]\n'
        if self.show_notes:
            footer += self.generate_notes()
        footer += '\\end{tabular}'

        if not only_tabular:
            footer += '\n\\end{table}'

        return footer

    def generate_custom_lines(self, location):
        custom_text = ''
        for custom_row in self.custom_lines[location]:
            custom_text += ' ' + str(custom_row[0]) + ' '
            for custom_column in custom_row[1:]:
                custom_text += '& ' + str(custom_column) + ' '
            custom_text += '\\\\\n'
        return custom_text

    def generate_stat(self, stat, label):
        values = self._generate_stat_values(stat)
        if not any(values):
            return ''

        formatter = self._formatters.get(stat, self._float_format)

        text = f' {label} '
        for value in values:
            if not isinstance(value, str):
                value = formatter(value)
            text += f'& {value} '
        text += '\\\\\n'
        return text

    def generate_notes(self):
        notes_text = ''
        notes_text += '\\textit{' + self.notes_label + '}'
        if self.notes_append and self.show_stars:
            notes_text += self.generate_p_value_section()
        notes_text += self.generate_additional_notes()
        return notes_text

    def generate_p_value_section(self):
        notes_text = ' & \\multicolumn{' + str(self.num_models) + '}{r}{'
        pval_cells = [self._format_sig_icon(self.sig_levels[idx] - 0.001)
                      + 'p$<$' + str(self.sig_levels[idx]) for idx in range(3)]
        notes_text += '; '.join(pval_cells)
        notes_text += '} \\\\\n'
        return notes_text

    def generate_additional_notes(self):
        notes_text = ''
        # if len(self.custom_notes) == 0:
        #     return notes_text
        for note in self.custom_notes:
            # if (i != 0) | (self.notes_append):
            #     notes_text += '\\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'
            # else:
            #     notes_text += ' & \\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'
            notes_text += '\\multicolumn{' + str(self.num_models+1) + '}{r}\\textit{' + self._escape(note) + '} \\\\\n'

        return notes_text
