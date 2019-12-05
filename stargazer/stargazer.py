"""
This is a python package to generate nicely formatted
regression results similar to the style of the R
package of the same name:
https://CRAN.R-project.org/package=stargazer

@authors:
    Matthew Burke:
        matthew.wesley.burke@gmail.com
        github.com/mwburke
"""

from __future__ import print_function
from statsmodels.regression.linear_model import RegressionResultsWrapper
from numpy import round, sqrt


class Stargazer:
    """
    Class that is constructed with one or more trained
    OLS models from the statsmodels package.

    The user then can change the rendering options by
    chaining different methods to the Stargazer object
    and then render the results in either HTML or LaTeX.
    """

    def __init__(self, models):
        self.models = models
        self.num_models = len(models)
        self.extract_data()
        self.reset_params()

    def validate_input(self):
        """
        Check inputs to see if they are going to
        cause any problems further down the line.

        Any future checking will be added here.
        """
        targets = []

        for m in self.models:
            if not isinstance(m, RegressionResultsWrapper):
                raise ValueError('Please use trained OLS models as inputs')
            targets.append(m.model.endog_names)

        if targets.count(targets[0]) != len(targets):
            raise ValueError('Please make sure OLS targets are identical')

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
        self.dep_var_name = 'Dependent variable:'
        self.column_labels = None
        self.column_separators = None
        self.show_model_nums = True
        self.original_cov_names = None
        self.cov_map = None
        self.show_precision = True
        self.show_sig = True
        self.sig_levels = [0.1, 0.05, 0.01]
        self.sig_digits = 3
        self.confidence_intervals = False
        self.show_footer = True
        self.custom_footer_text = []
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

    def extract_model_data(self, model):
        data = {}
        data['cov_names'] = model.params.index.values
        data['cov_values'] = model.params
        data['p_values'] = model.pvalues
        data['cov_std_err'] = model.bse
        data['conf_int_low_values'] = model.conf_int()[0]
        data['conf_int_high_values'] = model.conf_int()[1]
        data['r2'] = model.rsquared
        data['r2_adj'] = model.rsquared_adj
        data['resid_std_err'] = sqrt(model.scale)
        data['f_statistic'] = model.fvalue
        data['f_p_value'] = model.f_pvalue
        data['degree_freedom'] = model.df_model
        data['degree_freedom_resid'] = model.df_resid

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

    # Begin HTML render functions
    def render_html(self):
        html = ''
        html += self.generate_header_html()
        html += self.generate_body_html()
        html += self.generate_footer_html()

        return html

    def generate_header_html(self):
        header = ''
        if not self.show_header:
            return header

        if self.title_text is not None:
            header += self.title_text + '<br>'

        header += '<table style="text-align:center"><tr><td colspan="'
        header += str(self.num_models + 1) + '" style="border-bottom: 1px solid black"></td></tr>'
        if self.dep_var_name is not None:
            header += '<tr><td style="text-align:left"></td><td colspan="' + str(self.num_models)
            header += '"><em>' + self.dep_var_name + '</em></td></tr>'

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
        header += '" style="border-bottom: 1px solid black"></td></tr>'

        return header

    def generate_body_html(self):
        """
        Generate the body of the results where the
        covariate reporting is.
        """
        body = ''
        for cov_name in self.cov_names:
            body += self.generate_cov_rows_html(cov_name)

        return body

    def generate_cov_rows_html(self, cov_name):
        cov_text = ''
        cov_text += self.generate_cov_main_html(cov_name)
        if self.show_precision:
            cov_text += self.generate_cov_precision_html(cov_name)
        else:
            cov_text += '<tr></tr>'

        return cov_text

    def generate_cov_main_html(self, cov_name):
        cov_print_name = cov_name
        if self.cov_map is not None:
            cov_print_name = self.cov_map.get(cov_print_name, cov_name)
        cov_text = '<tr><td style="text-align:left">' + cov_print_name + '</td>'
        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += '<td>'
                cov_text += str(round(md['cov_values'][cov_name], self.sig_digits))
                if self.show_sig:
                    cov_text += '<sup>' + str(self.get_sig_icon(md['p_values'][cov_name])) + '</sup>'
                cov_text += '</td>'
            else:
                cov_text += '<td></td>'
        cov_text += '</tr>'

        return cov_text

    def generate_cov_precision_html(self, cov_name):
        cov_text = '<tr><td style="text-align:left"></td>'
        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += '<td>('
                if self.confidence_intervals:
                    cov_text += str(round(md['conf_int_low_values'][cov_name], self.sig_digits)) + ' , '
                    cov_text += str(round(md['conf_int_high_values'][cov_name], self.sig_digits))
                else:
                    cov_text += str(round(md['cov_std_err'][cov_name], self.sig_digits))
                cov_text += ')</td>'
            else:
                cov_text += '<td></td>'
        cov_text += '</tr>'

        return cov_text

    def get_sig_icon(self, p_value, sig_char='*'):
        if p_value >= self.sig_levels[0]:
            return ''
        elif p_value >= self.sig_levels[1]:
            return sig_char
        elif p_value >= self.sig_levels[2]:
            return sig_char * 2
        else:
            return sig_char * 3

    def generate_footer_html(self):
        """
        Generate the footer of the table where
        model summary section is.
        """
        footer = '<td colspan="' + str(self.num_models + 1) + '" style="border-bottom: 1px solid black"></td></tr>'

        if not self.show_footer:
            return footer
        footer += self.generate_observations_html()
        footer += self.generate_r2_html()
        footer += self.generate_r2_adj_html()
        if self.show_residual_std_err:
            footer += self.generate_resid_std_err_html()
        if self.show_f_statistic:
            footer += self.generate_f_statistic_html()
        footer += '<tr><td colspan="' + str(self.num_models + 1) + '" style="border-bottom: 1px solid black"></td></tr>'
        footer += self.generate_notes_html()
        footer += '</table>'

        return footer

    def generate_observations_html(self):
        obs_text = ''
        if not self.show_n:
            return obs_text
        obs_text += '<tr><td style="text-align: left">Observations</td>'
        for md in self.model_data:
            obs_text += '<td>' + str(md['degree_freedom'] + md['degree_freedom_resid'] + 1) + '</td>'
        obs_text += '</tr>'
        return obs_text

    def generate_r2_html(self):
        r2_text = ''
        if not self.show_r2:
            return r2_text
        r2_text += '<tr><td style="text-align: left">R<sup>2</sup></td>'
        for md in self.model_data:
            r2_text += '<td>' + str(round(md['r2'], self.sig_digits)) + '</td>'
        r2_text += '</tr>'
        return r2_text

    def generate_r2_adj_html(self):
        r2_text = ''
        if not self.show_r2:
            return r2_text
        r2_text += '<tr><td style="text-align: left">Adjusted R<sup>2</sup></td>'
        for md in self.model_data:
            r2_text += '<td>' + str(round(md['r2_adj'], self.sig_digits)) + '</td>'
        r2_text += '</tr>'
        return r2_text

    def generate_resid_std_err_html(self):
        rse_text = ''
        if not self.show_r2:
            return rse_text
        rse_text += '<tr><td style="text-align: left">Residual Std. Error</td>'
        for md in self.model_data:
            rse_text += '<td>' + str(round(md['resid_std_err'], self.sig_digits))
            if self.show_dof:
                rse_text += '(df = ' + str(round(md['degree_freedom_resid'])) + ')'
            rse_text += '</td>'
        rse_text += '</tr>'
        return rse_text

    def generate_f_statistic_html(self):
        f_text = ''
        if not self.show_r2:
            return f_text
        f_text += '<tr><td style="text-align: left">F Statistic</td>'
        for md in self.model_data:
            f_text += '<td>' + str(round(md['f_statistic'], self.sig_digits))
            f_text += '<sup>' + self.get_sig_icon(md['f_p_value']) + '</sup>'
            if self.show_dof:
                f_text += '(df = ' + str(md['degree_freedom']) + '; ' + str(md['degree_freedom_resid']) + ')'
            f_text += '</td>'
        f_text += '</tr>'
        return f_text

    def generate_notes_html(self):
        notes_text = ''
        if not self.show_notes:
            return notes_text

        notes_text += '<tr><td style="text-align: left">' + self.notes_label + '</td>'

        if self.notes_append:
            notes_text += self.generate_p_value_section_html()

        notes_text += '</tr>'

        notes_text += self.generate_additional_notes_html()

        return notes_text

    def generate_p_value_section_html(self):
        notes_text = """
 <td colspan="{}" style="text-align: right">
  <sup>*</sup>p&lt;{};
  <sup>**</sup>p&lt;{};
  <sup>***</sup>p&lt;{}
 </td>""".format(self.num_models, *self.sig_levels)
        return notes_text

    def generate_additional_notes_html(self):
        notes_text = ''
        if len(self.custom_notes) == 0:
            return notes_text
        i = 0
        for i, note in enumerate(self.custom_notes):
            if (i != 0) | (self.notes_append):
                notes_text += '<tr>'
            notes_text += '<td></td><td colspan="' + str(self.num_models) + '" style="text-align: right">' + note + '</td></tr>'

        return notes_text

    # Begin LaTeX render functions
    def render_latex(self, only_tabular=False):
        latex = ''
        latex += self.generate_header_latex(only_tabular=only_tabular)
        latex += self.generate_body_latex()
        latex += self.generate_footer_latex(only_tabular=only_tabular)

        return latex

    def generate_header_latex(self, only_tabular=False):
        header = ''
        if not only_tabular:
            header += '\\begin{table}[!htbp] \\centering\n'
            if not self.show_header:
                return header

            if self.title_text is not None:
                header += '  \\caption{' + self.title_text + '}\n'

            header += '  \\label{}\n'

        header += '\\begin{tabular}{@{\\extracolsep{5pt}}lcc}\n'
        header += '\\\\[-1.8ex]\\hline\n'
        header += '\\hline \\\\[-1.8ex]\n'
        if self.dep_var_name is not None:
            header += '& \\multicolumn{' + str(self.num_models) + '}{c}'
            header += '{\\textit{' + self.dep_var_name + '}} \\\n'
            header += '\\cr \\cline{' + str(self.num_models) + '-' + str(self.num_models + 1) + '}\n'

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

    def generate_body_latex(self):
        """
        Generate the body of the results where the
        covariate reporting is.
        """
        body = ''
        for cov_name in self.cov_names:
            body += self.generate_cov_rows_latex(cov_name)
            body += '  '
            for _ in range(self.num_models):
                body += '& '
            body += '\\\\\n'

        return body

    def generate_cov_rows_latex(self, cov_name):
        cov_text = ''
        cov_text += self.generate_cov_main_latex(cov_name)
        if self.show_precision:
            cov_text += self.generate_cov_precision_latex(cov_name)
        else:
            cov_text += '& '

        return cov_text

    def generate_cov_main_latex(self, cov_name):
        cov_print_name = cov_name

        if self.cov_map is not None:
            if cov_name in self.cov_map:
                cov_print_name = self.cov_map[cov_name]

        cov_text = ' ' + cov_print_name + ' '
        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += '& ' + str(round(md['cov_values'][cov_name], self.sig_digits))
                if self.show_sig:
                    cov_text += '$^{' + str(self.get_sig_icon(md['p_values'][cov_name])) + '}$'
                cov_text += ' '
            else:
                cov_text += '& '
        cov_text += '\\\\\n'

        return cov_text

    def generate_cov_precision_latex(self, cov_name):
        cov_text = '  '

        for md in self.model_data:
            if cov_name in md['cov_names']:
                cov_text += '& ('
                if self.confidence_intervals:
                    cov_text += str(round(md['conf_int_low_values'][cov_name], self.sig_digits)) + ' , '
                    cov_text += str(round(md['conf_int_high_values'][cov_name], self.sig_digits))
                else:
                    cov_text += str(round(md['cov_std_err'][cov_name], self.sig_digits))
                cov_text += ') '
            else:
                cov_text += '& '
        cov_text += '\\\\\n'

        return cov_text

    def generate_footer_latex(self, only_tabular=False):
        """
        Generate the footer of the table where
        model summary section is.
        """

        footer = '\\hline \\\\[-1.8ex]\n'

        if not self.show_footer:
            return footer
        footer += self.generate_observations_latex()
        footer += self.generate_r2_latex()
        footer += self.generate_r2_adj_latex()
        if self.show_residual_std_err:
            footer += self.generate_resid_std_err_latex()
        if self.show_f_statistic:
            footer += self.generate_f_statistic_latex()
        footer += '\\hline\n\\hline \\\\[-1.8ex]\n'
        footer += self.generate_notes_latex()
        footer += '\\end{tabular}'

        if not only_tabular:
            footer += '\n\\end{table}'

        return footer

    def generate_observations_latex(self):
        obs_text = ''
        if not self.show_n:
            return obs_text
        obs_text += ' Observations '
        for md in self.model_data:
            obs_text += '& ' + str(md['degree_freedom'] + md['degree_freedom_resid'] + 1) + ' '
        obs_text += '\\\\\n'
        return obs_text

    def generate_r2_latex(self):
        r2_text = ''
        if not self.show_r2:
            return r2_text
        r2_text += ' R${2}$ '
        for md in self.model_data:
            r2_text += '& ' + str(round(md['r2'], self.sig_digits)) + ' '
        r2_text += '\\\\\n'
        return r2_text

    def generate_r2_adj_latex(self):
        r2_text = ''
        if not self.show_r2:
            return r2_text
        r2_text += ' Adjusted R${2}$ '
        for md in self.model_data:
            r2_text += '& ' + str(round(md['r2_adj'], self.sig_digits)) + ' '
        r2_text += '\\\\\n'
        return r2_text

    def generate_resid_std_err_latex(self):
        rse_text = ''
        if not self.show_r2:
            return rse_text
        rse_text += ' Residual Std. Error '
        for md in self.model_data:
            rse_text += '& ' + str(round(md['resid_std_err'], self.sig_digits))
            if self.show_dof:
                rse_text += '(df = ' + str(round(md['degree_freedom_resid'])) + ')'
            rse_text += ' '
        rse_text += ' \\\\\n'
        return rse_text

    def generate_f_statistic_latex(self):
        f_text = ''
        if not self.show_r2:
            return f_text

        f_text += ' F Statistic '

        for md in self.model_data:
            f_text += '& ' + str(round(md['f_statistic'], self.sig_digits))
            f_text += '$^{' + self.get_sig_icon(md['f_p_value']) + '}$ '
            if self.show_dof:
                f_text += '(df = ' + str(md['degree_freedom']) + '; ' + str(md['degree_freedom_resid']) + ')'
            f_text += ' '
        f_text += '\\\\\n'
        return f_text

    def generate_notes_latex(self):
        notes_text = ''
        if not self.show_notes:
            return notes_text

        notes_text += '\\textit{' + self.notes_label + '}'

        if self.notes_append:
            notes_text += self.generate_p_value_section_latex()
        notes_text += self.generate_additional_notes_latex()

        return notes_text

    def generate_p_value_section_latex(self):
        notes_text = ''
        notes_text += ' & \\multicolumn{' + str(self.num_models) + '}{r}{$^{' + self.get_sig_icon(self.sig_levels[0] - 0.001) + '}$p$<$' + str(self.sig_levels[0]) + '; '
        notes_text += '$^{' + self.get_sig_icon(self.sig_levels[1] - 0.001) + '}$p$<$' + str(self.sig_levels[1]) + '; '
        notes_text += '$^{' + self.get_sig_icon(self.sig_levels[2] - 0.001) + '}$p$<$' + str(self.sig_levels[2]) + '} \\\\\n'
        return notes_text

    def generate_additional_notes_latex(self):
        notes_text = ''
        # if len(self.custom_notes) == 0:
        #     return notes_text
        for note in self.custom_notes:
            # if (i != 0) | (self.notes_append):
            #     notes_text += '\\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'
            # else:
            #     notes_text += ' & \\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'
            notes_text += ' & \\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'

        return notes_text

    # Begin Markdown render functions
    # def render_markdown(self):
    #     print("sorry haven't made this yet :/")

    # Begin ASCII render functions
    # def render_ascii(self):
    #     print("sorry haven't made this yet :/")
