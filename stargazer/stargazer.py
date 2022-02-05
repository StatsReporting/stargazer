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
import xlsxwriter

class LineLocation(Enum):
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
        if separators is not None:
            assert type(labels) == list, 'Please input a list of labels or a single label string'
            assert type(separators) == list, 'Please input a list of column separators'
            assert len(labels) == len(separators), 'Number of labels must match number of columns'
            assert sum([int(type(s) != int) for s in separators]) == 0, 'Columns must be ints'
            assert sum(separators) == self.num_models, 'Please set number of columns to number of models'
        else:
            assert type(labels) == str, 'Please input a single string label if no columns specified'

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
        self.dep_var_name = name

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
    
    def render_excel(self, *args, **kwargs):
        return ExcelRenderer(self).render(*args, **kwargs)


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
        if self.dep_var_name is not None:
            header += '& \\multicolumn{' + str(self.num_models) + '}{c}'
            header += '{\\textit{' + self.dep_var_name + self.dependent_variable + '}} \\\n'
            header += '\\cr \\cline{2-' + str(self.num_models + 1) + '}\n'

        if self.column_labels is not None:
            if type(self.column_labels) == str:
                header += '\\\\[-1.8ex] & \\multicolumn{' + str(self.num_models) + '}{c}{' + self.column_labels + '} \\\\'
            else:
                header += '\\\\[-1.8ex] '
                for i, label in enumerate(self.column_labels):
                    header += '& \\multicolumn{' + str(self.column_separators[i])
                    header += '}{c}{' + label + '} '
                header += ' \\\\\n'

        if self.show_model_nums:
            header += '\\\\[-1.8ex] '
            for num in range(1, self.num_models + 1):
                header += '& (' + str(num) + ') '
            header += '\\\\\n'

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

class ExcelRenderer(Renderer):
    fmt = 'Excel'

    # Labels for stats in Stargazer._auto_stats:
    _stats_labels = {'n' : 'Observations',
                     'r2' : 'R²',
                     'adj_r2' : 'Adjusted R²',
                     'residual_std_err' : 'Residual Std. Error',
                     'f_statistic' : 'F Statistic'}
    
    def render(self, filename='workbook.xlsx', ignore_errors=True, fit_to_width=True, cell_height=20, start_row=1, start_col=1, insert_empty_rows=False):
        with xlsxwriter.Workbook(filename) as wb:
            ws = wb.add_worksheet()
            row = self.generate_header(wb, ws, start_row, start_col)
            row = self.generate_body(wb, ws, row, start_col, insert_empty_rows=insert_empty_rows)
            row = self.generate_footer(wb, ws, row, start_col)

            if ignore_errors:
                ws.ignore_errors(
                    {'number_stored_as_text': f'A1:{self._excel_col(start_col+1+self.num_models)}{row+1}'}
                )
            
            if fit_to_width:
                self._fit_to_width(ws, start_col)

            if cell_height > 0:
                for i in range(start_row, row):
                    ws.set_row(i, cell_height)

    def generate_header(self, wb, ws, row, col):

        if not self.show_header:
            return

        if self.title_text is not None:
            ws.write(row, col, self.title_text)
            row += 1

        if self.dep_var_name is not None:
            ws.write(row, col, '', wb.add_format({'top': 6, 'bottom': 1}))
            ws.merge_range(row, col+1, 
                row, col+self.num_models, 
                self.dep_var_name + self.dependent_variable, 
                wb.add_format({'top' : 6, 'bottom': 1, 'align' : 'center', 'valign': 'vcenter', 'italic' : True})
            )
            row += 1
        
        if self.show_model_nums:

            if self.dep_var_name is not None:
                format1 = {'top': 0, 'bottom': 1}
                format2 = {'bottom': 1, 'align': 'center', 'valign': 'vcenter'}
            else:
                format1 = {'top': 6, 'bottom': 1}
                format2 = {'top': 6, 'bottom': 1, 'align': 'center', 'valign': 'vcenter'}
            
            ws.write(row, col, '', wb.add_format(format1))
            ws.write_row(row,col+1,
                [f'({i})' for i in range(1,self.num_models+1)], 
                wb.add_format(format2)
            )
            row += 1

        return row

    def generate_body(self, wb, ws, row, col, insert_empty_rows=False):

        row = self.generate_custom_lines(LineLocation.BODY_TOP, wb, ws, row, col)

        for cov_name in self.cov_names:
            row = self.generate_cov_rows(wb, ws, row, col, cov_name)
            if insert_empty_rows:
                row += 1

        row = self.generate_custom_lines(LineLocation.BODY_BOTTOM, wb, ws, row, col)

        return row

    def generate_cov_rows(self, wb, ws, row, col, cov_name):

        cov_print_name = cov_name
        if self.cov_map is not None:
            cov_print_name = self.cov_map.get(cov_print_name, cov_name)

        ws.write(row, col, cov_print_name, wb.add_format({'align': 'left', 'valign': 'vcenter'}))
        ws.write(row+1, col, '', wb.add_format({'align': 'left', 'valign': 'vcenter'}))

        for (i, md) in enumerate(self.model_data):  # refactor to generate_cov_rows

            if cov_name in md['cov_names']:
                cov_text = self._float_format(md['cov_values'][cov_name])

                if self.show_sig:
                    cov_text += self._format_sig_icon(md['p_values'][cov_name])

                cov_prec_text = '('

                if self.confidence_intervals:
                    cov_prec_text += self._float_format(md['conf_int_low_values'][cov_name]) + ' , '
                    cov_prec_text += self._float_format(md['conf_int_high_values'][cov_name])
                else:
                    cov_prec_text += self._float_format(md['cov_std_err'][cov_name])

                cov_prec_text += ')'

            else:
                cov_text = ''
                cov_prec_text = ''

            ws.write(row, col+1+i, cov_text, wb.add_format({'align': 'center', 'valign': 'vcenter'}))
            ws.write(row+1, col+1+i, cov_prec_text, wb.add_format({'align': 'center', 'valign': 'vcenter'}))
        
        row += 2
        return row

    def generate_footer(self, wb, ws, row, col):

        if not self.show_footer:
            return

        for (i,(attr, stat)) in enumerate(Stargazer._auto_stats):
            if getattr(self, f'show_{attr}'):
                self._generate_stat(wb, ws, stat, self._stats_labels[attr], row, col, i, len(Stargazer._auto_stats))
                row +=1

        if self.show_notes:
            ws.write(row, 1, self.notes_label, wb.add_format({'align': 'left', 'italic' : True, 'top': 6, 'valign': 'vcenter'}))
            if self.notes_append and self.show_stars:
                ws.merge_range(row, col+1, row, col+self.num_models, self._generate_p_value_section(), wb.add_format({'align': 'right', 'top': 6, 'valign': 'vcenter'}))
            row += 1

        return row

    def _format_sig_icon(self, pvalue):
        return '*' * len(str(self.get_sig_icon(pvalue)))

    def _generate_p_value_section(self):
        return '; '.join([self._format_sig_icon(sig_level - 0.001)
                      + '<' + str(sig_level) for sig_level in self.sig_levels])

    def _excel_col(self, col):
        quot, rem = divmod(col-1,26)
        return self._excel_col(quot) + chr(rem+ord('A')) if col!=0 else ''

    def generate_custom_lines(self, location, wb, ws, row, col):
        for custom_row in self.custom_lines[location]:
            format1 = {'align': 'left', 'valign': 'vcenter'}
            format2 = {'align': 'center', 'valign': 'vcenter'}

            if 'BOTTOM' in location.name:
                format1 = {'align': 'left', 'valign': 'vcenter', 'bottom': 1}
                format2 = {'align': 'center', 'valign': 'vcenter', 'bottom': 1}

                if 'FOOTER' in location.name:
                    format1['bottom'] = 6
                    format1['bottom'] = 6

            ws.write(row, col, str(custom_row[0]), wb.add_format(format1))
            ws.write_row(row, col+1, [str(custom_column) for custom_column in custom_row[1:]], wb.add_format(format2))
            row += 1

        return row

    def _generate_stat(self, wb, ws, stat, label, row, col, i, n):
        values = self._generate_stat_values(stat)
        if not any(values):
            return ''

        format = {'align': 'left', 'valign': 'vcenter'}
        if n == 1:
            format.update({'top': 1, 'bottom': 6})
        elif i == 0:
            format.update({'top': 1})
        elif i == n - 1:
            format.update({'bottom': 6})

        ws.write(row, col, label, wb.add_format(format))

        format['align'] = 'center'
        formatter = self._formatters.get(stat, self._float_format)
        for (j, value) in enumerate(values):
            if not isinstance(value, str):
                value = formatter(value)
            ws.write(row, col+1+j, value, wb.add_format(format))

    def _fit_to_width(self, ws, col, min_width=15, max_width=75):

        data = ws.table
        str_data = ws.str_table

        string_idxs = [[d[i].string if hasattr(d.get(i), 'string') else None for d in data.values()] for i in range(1,2+self.num_models)]
        string_idx_mapper = {v:k  for k,v in str_data.string_table.items()}

        df = pd.DataFrame(string_idxs).transpose().stack().reset_index(level=0, drop=True)
        df = df.map(string_idx_mapper).apply(len)

        width_index = df[df.index==0].max()
        width_values = df[df.index!=0].max()

        width_index = max(min(width_index*0.9, max_width), min_width)
        width_values = max(min(width_values*0.9, max_width), min_width)

        ws.set_column(col, col, width_index)
        ws.set_column(col+1, col+self.num_models, width_values)
