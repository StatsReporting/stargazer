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

from collections import defaultdict
from enum import Enum
import numbers

from .label import Label
from .starlib import _find_duplicates
from itertools import chain
import math as m

RESULTS_CLASSES = {}

from .translators import *


class LineLocation(Enum):
    HEADER_TOP = "ht"
    HEADER_BOTTOM = "hb"
    BODY_TOP = "bt"
    BODY_BOTTOM = "bb"
    FOOTER_TOP = "ft"
    FOOTER_BOTTOM = "fb"


class Stargazer:
    """
    Class that is constructed with one or more trained statistical models, for
    instance from the statsmodels package.

    The user then can change the rendering options by
    chaining different methods to the Stargazer object
    and then render the results in either HTML or LaTeX.
    """

    # The following tuple represents a mapping from 'show_*' attribute to
    # - name of generating method "_generate_{LABEL}" (if present) and
    #   name of stat in data store otherwise.
    # - label, in the form of a str or a Label
    # Stats will be automatically formatted. Order matters!
    # Note that a stat is automatically hidden if none of the models displayed
    # has it.

    _auto_stats = [
        ("n", "nobs", "Observations"),
        ("ngroups", "ngroups", "N. of groups"),
        ("r2", "r2", Label({"LaTeX": "$R^2$", "html": "R<sup>2</sup>"})),
        (
            "within_r2",
            "within_r2",
            Label({"LaTeX": "Within $R^2$", "html": "Within R<sup>2</sup>"}),
        ),
        (
            "between_r2",
            "between_r2",
            Label({"LaTeX": "Between $R^2$", "html": "Between R<sup>2</sup>"}),
        ),
        (
            "adj_r2",
            "r2_adj",
            Label({"LaTeX": "Adjusted $R^2$", "html": "Adjusted R<sup>2</sup>"}),
        ),
        (
            "pseudo_r2",
            "pseudo_r2",
            Label({"LaTeX": "Pseudo $R^2$", "html": "Pseudo R<sup>2</sup>"}),
        ),
        ("residual_std_err", "resid_std_err", "Residual Std. Error"),
        ("f_statistic", "f_statistic", "F Statistic"),
    ]

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

        targets = [mod["dependent_variable"] for mod in self.model_data]

        if targets.count(targets[0]) != len(targets):
            self.dependent_variable = ""
            self.dep_var_name = None
        else:
            self.dependent_variable = targets[0]

        self.dependent_variables = targets

    def reset_params(self):
        """
        Set all of the rendering parameters to their default settings.
        Run upon initialization but also allows the user to reset
        if they have made several changes and want to start fresh.

        Does not effect any of the underlying model data.
        """
        self.title_text = None
        self.show_header = True
        self.dep_var_name = "Dependent variable: "
        self.column_labels = None
        self.column_separators = None
        self.show_model_nums = True
        self.original_cov_names = None
        self.cov_map = lambda x: x
        self.cov_spacing = None
        self.show_precision = True
        self.show_sig = True
        self.sig_levels = [0.1, 0.05, 0.01]
        self.sig_digits = 3
        self.confidence_intervals = False
        self.show_footer = True
        self.custom_lines = defaultdict(list)
        self.show_n = True
        self.show_ngroups = True
        self.show_r2 = True
        # Off by default, as discussed in GH #98:
        self.show_within_r2 = False
        self.show_between_r2 = False
        self.show_adj_r2 = True
        self.show_pseudo_r2 = True
        self.show_residual_std_err = True
        self.show_f_statistic = True
        self.show_dof = True
        self.show_notes = True
        self.notes_label = "Note:"
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
        self.model_data = []
        for m in self.models:
            self.model_data.append(self.extract_model_data(m))

        covs = []
        for md in self.model_data:
            covs = covs + list(md["cov_names"])
        self.cov_names = sorted(set(covs))

        self.validate_input()

    def extract_model_data(self, model):
        """
        Call appropriate (library-specific) data extractor.
        """

        for klass in model.__class__.__mro__:
            if klass in RESULTS_CLASSES:
                extractor = RESULTS_CLASSES[klass]

                return extractor(model)

        raise NotImplementedError(model.__class__)

    # Begin render option functions
    def title(self, title):
        self.title_text = title

    def show_header(self, show):
        assert type(show) == bool, "Please input True/False"
        self.header = show

    def show_model_numbers(self, show):
        assert type(show) == bool, "Please input True/False"
        self.show_model_nums = show

    def custom_columns(self, labels, separators=None):
        """
        "labels": list of labels, or single label string.
        "separators" (optional): list of integers, of same length of "labels",
        indicating how many columns each header covers (default is 1).
        """
        if isinstance(labels, list):
            if separators is None:
                assert len(labels) == self.num_models, (
                    "If separators are "
                    "not provided, custom headers must be the same number as "
                    "models,"
                )
                separators = [1] * self.num_models
            else:
                assert type(separators) == list, (
                    "Please input a list of " "column separators."
                )
                assert len(labels) == len(separators), (
                    "Number of labels " "must match number of columns."
                )
                assert all(isinstance(s, int) for s in separators), (
                    "Column " "numbers must be integers."
                )
                assert sum(separators) == self.num_models, (
                    "Please set total " "number of columns to number of models."
                )
        else:
            assert isinstance(labels, str), (
                "Please input a single string " "label, or a list of strings."
            )

        self.column_labels = labels
        self.column_separators = separators

    def significance_levels(self, levels):
        assert len(levels) == 3, "Please input 3 significance levels"
        assert (
            sum([int(type(l) != float) for l in levels]) == 0
        ), "Please input floating point values as significance levels"
        self.sig_levels = sorted(levels, reverse=True)

    def significant_digits(self, digits):
        assert type(digits) == int, "The number of significant digits must be an int"
        assert digits < 10, "Whoa hold on there bud, maybe use fewer digits"
        self.sig_digits = digits

    def show_confidence_intervals(self, show):
        assert type(show) == bool, "Please input True/False"
        self.confidence_intervals = show

    def dependent_variable_name(self, name):
        assert (
            type(name) == str
        ), "Please input a string to use as the depedent variable name"
        self.dependent_variable = name

    def covariate_order(self, cov_names, restrict=True):
        """
        Specify order and subset of covariates.

        Paramters
        ---------
        cov_names : list of str
            Covariate names, or a subset of them, in the desired order.
        restrict : bool, default True
            Whether covariates not included in "cov_names" should be dropped;
            otherwise, they are just appended, in their default order.
        """

        duplicates = _find_duplicates(cov_names)
        assert not duplicates, (
            "Covariates must not be repeated, the " f"following are: {duplicates}"
        )

        missing = set(cov_names).difference(set(self.cov_names))
        assert not missing, (
            "Covariate order must contain subset of existing "
            "covariates: {} are not.".format(missing)
        )

        if not restrict:
            others = [name for name in self.cov_names if name not in cov_names]
            cov_names += others

        self.original_cov_names = self.cov_names
        self.cov_names = cov_names

    def rename_covariates(self, cov_map):
        if hasattr(cov_map, "get"):
            self.cov_map = lambda k: cov_map.get(k, k)
        elif callable(cov_map):
            self.cov_map = cov_map
        else:
            msg = (
                '"rename_covariates" must receive either a dictionary with '
                "covariate names as keys, or a function accepting any "
                "covariate name as input and returning the new name."
            )
            raise ValueError(msg)

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
        assert (
            len(values) == self.num_models
        ), "values has to be an iterables with {} elements (one for each model)".format(
            self.num_models
        )
        if type(location) != LineLocation:
            location = LineLocation(location)
        self.custom_lines[location].append([label] + values)

    def show_degrees_of_freedom(self, show):
        assert type(show) == bool, "Please input True/False"
        self.show_dof = show

    def custom_note_label(self, notes_label):
        assert (
            type(notes_label) == str
        ), "Please input a string to use as the note label"
        self.notes_label = notes_label

    def add_custom_notes(self, notes):
        assert sum([int(type(n) != str) for n in notes]) == 0, "Notes must be strings"
        self.custom_notes = notes

    def append_notes(self, append):
        assert type(append) == bool, "Please input True/False"
        self.notes_append = append

    def render_html(self, *args, **kwargs):
        with Label.context("html"):
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
            To prevent a specific string from being escaped, just pass it as

                Label({'LaTeX' : 'the_string'})

            ... it will be left unchanged.

        Returns
        -------
        str
            The LaTeX code.
        """
        with Label.context("LaTeX"):
            return LaTeXRenderer(self, escape=escape).render(*args, **kwargs)

    def render_threeparttable(self, *args, escape=False, **kwargs):
        """
        Render as LaTeX code.

        Parameters
        ----------
        escape : bool
            Escape special characters.
            To prevent a specific string from being escaped, just pass it as

                Label({'LaTeX' : 'the_string'})

            ... it will be left unchanged.

        Returns
        -------
        str
            The LaTeX code.
        """
        with Label.context("LaTeX"):
            return ThreePartRenderer(self, escape=escape).render(*args, **kwargs)


class Renderer:
    """
    Base class for renderers to specific formats. Only meant to be subclassed.
    """

    # Formatters for stats which are not formatted via Renderer._float_format()
    _formatters = {"nobs": lambda x: str(int(x))}

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

    def get_sig_icon(self, p_value, sig_char="*"):
        if p_value is None or not self.show_stars:
            return ""
        if p_value >= self.sig_levels[0]:
            return ""
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
            return f"{self.cov_spacing}em"
        else:
            return self.cov_spacing

    def _float_format(self, value):
        """
        Format value to string, using the precision set by the user.
        """
        if value is None:
            return ""

        return "{{:.{prec}f}}".format(prec=self.sig_digits).format(value)

    def _generate_resid_std_err(self, md):
        rse = md["resid_std_err"]
        if rse is None:
            return None

        rse_text = self._float_format(rse)
        if self.show_dof:
            rse_text += " (df={degree_freedom_resid:.0f})".format(**md)
        return rse_text

    def _generate_f_statistic(self, md):
        f_stat = md["f_statistic"]
        if f_stat is None:
            return None

        f_stars = self._format_sig_icon(md["f_p_value"])
        f_text = f"{self._float_format(f_stat)}{f_stars}"
        if self.show_dof:
            f_text += (
                " (df={degree_freedom:.0f}; " "{degree_freedom_resid:.0f})"
            ).format(**md)

        return f_text

    def _generate_stat_values(self, stat):
        if hasattr(self, f"_generate_{stat}"):
            generator = getattr(self, f"_generate_{stat}")
            return [generator(md) for md in self.model_data]
        else:
            return [md.get(stat, None) for md in self.model_data]


class HTMLRenderer(Renderer):
    fmt = "html"

    def render(self):
        html = self.generate_header()
        html += self.generate_body()
        html += self.generate_footer()
        return html

    def generate_header(self):
        header = ""
        if not self.show_header:
            return header

        if self.title_text is not None:
            header += self.title_text + "<br>"

        header += '<table style="text-align:center"><tr><td colspan="'
        header += (
            str(self.num_models + 1)
            + '" style="border-bottom: 1px solid black"></td></tr>'
        )
        header += self.generate_custom_lines(LineLocation.HEADER_TOP)

        if self.dep_var_name is not None:
            header += '<tr><td style="text-align:left"></td><td colspan="' + str(
                self.num_models
            )
            header += (
                '"><em>'
                + self.dep_var_name
                + self.dependent_variable
                + "</em></td></tr>"
            )

        header += '<tr><td style="text-align:left"></td>'
        if self.column_labels is not None:
            if type(self.column_labels) == str:
                header += '<td colspan="' + str(self.num_models) + '">'
                header += self.column_labels + "</td></tr>"
            else:
                # The first table column holds the covariates names:
                header += "<tr><td></td>"
                for i, label in enumerate(self.column_labels):
                    sep = self.column_separators[i]
                    header += '<td colspan="{}">{}</td>'.format(sep, label)
                header += "</tr>"

        if self.show_model_nums:
            header += '<tr><td style="text-align:left"></td>'
            for num in range(1, self.num_models + 1):
                header += "<td>(" + str(num) + ")</td>"
            header += "</tr>"

        header += self.generate_custom_lines(LineLocation.HEADER_BOTTOM)

        header += '<tr><td colspan="' + str(self.num_models + 1)
        header += '" style="border-bottom: 1px solid black"></td></tr>\n'

        return header

    def _generate_cov_style(self):
        if self.cov_spacing is None:
            return ""
        spacing = self._generate_cov_spacing()
        return f' style="padding-bottom:{spacing}"'

    def _format_sig_icon(self, pvalue):
        return "<sup>" + str(self.get_sig_icon(pvalue)) + "</sup>"

    def generate_body(self):
        """
        Generate the body of the results where the
        covariate reporting is.
        """

        spacing = self._generate_cov_style()

        body = ""
        body += self.generate_custom_lines(LineLocation.BODY_TOP)
        for cov_name in self.cov_names:
            body += self.generate_cov_rows(cov_name, spacing)
        body += self.generate_custom_lines(LineLocation.BODY_BOTTOM)

        return body

    def generate_cov_rows(self, cov_name, spacing):
        cov_text = ""
        main_spacing = spacing if not self.show_precision else ""
        cov_text += self.generate_cov_main(cov_name, spacing=main_spacing)
        if self.show_precision:
            cov_text += self.generate_cov_precision(cov_name, spacing=spacing)
        else:
            cov_text += "<tr></tr>"

        return cov_text

    def generate_cov_main(self, cov_name, spacing):
        cov_print_name = self.cov_map(cov_name)

        cov_text = f'<tr><td style="text-align:left">' f"{cov_print_name}</td>"
        for md in self.model_data:
            if cov_name in md["cov_names"]:
                cov_text += f"<td{spacing}>"
                cov_text += self._float_format(md["cov_values"][cov_name])
                if self.show_sig:
                    cov_text += self._format_sig_icon(md["p_values"][cov_name])
                cov_text += "</td>"
            else:
                cov_text += f"<td{spacing}></td>"
        cov_text += "</tr>\n"

        return cov_text

    def generate_cov_precision(self, cov_name, spacing):
        # This is the only place where we need to add spacing and there's a
        # "style" already:
        space_style = (
            f";padding-bottom:{self._generate_cov_spacing()}"
            if self.cov_spacing
            else ""
        )
        cov_text = f'<tr><td style="text-align:left{space_style}"></td>'
        for md in self.model_data:
            if cov_name in md["cov_names"]:
                cov_text += f"<td{spacing}>("
                if self.confidence_intervals:
                    cov_text += (
                        self._float_format(md["conf_int_low_values"][cov_name]) + " , "
                    )
                    cov_text += self._float_format(md["conf_int_high_values"][cov_name])
                else:
                    cov_text += self._float_format(md["cov_std_err"][cov_name])
                cov_text += ")</td>"
            else:
                cov_text += f"<td{spacing}></td>"
        cov_text += "</tr>\n"

        return cov_text

    def generate_footer(self):
        """
        Generate the footer of the table where
        model summary section is.
        """
        footer = (
            '<td colspan="'
            + str(self.num_models + 1)
            + '" style="border-bottom: 1px solid black"></td></tr>'
        )

        if not self.show_footer:
            return footer
        footer += self.generate_custom_lines(LineLocation.FOOTER_TOP)

        for attr, stat, label in Stargazer._auto_stats:
            if getattr(self, f"show_{attr}"):
                footer += self.generate_stat(stat, label)

        footer += self.generate_custom_lines(LineLocation.FOOTER_BOTTOM)
        footer += (
            '<tr><td colspan="'
            + str(self.num_models + 1)
            + '" style="border-bottom: 1px solid black"></td></tr>'
        )
        if self.show_notes:
            footer += self.generate_notes()
        footer += "</table>"

        return footer

    def generate_custom_lines(self, location):
        custom_text = "\n"
        for custom_row in self.custom_lines[location]:
            custom_text += (
                '<tr><td style="text-align: left">' + str(custom_row[0]) + "</td>"
            )
            for custom_column in custom_row[1:]:
                custom_text += "<td>" + str(custom_column) + "</td>"
            custom_text += "</tr>"
        return custom_text

    def generate_stat(self, stat, label):
        values = self._generate_stat_values(stat)
        if not any(values):
            return ""

        formatter = self._formatters.get(stat, self._float_format)

        text = f'<tr><td style="text-align: left">{label}</td>'
        for value in values:
            if not isinstance(value, str):
                value = formatter(value)
            text += f"<td>{value}</td>"
        text += "</tr>"
        return text

    def generate_notes(self):
        notes_text = ""
        notes_text += '<tr><td style="text-align: left">' + self.notes_label + "</td>"
        if self.notes_append and self.show_stars:
            notes_text += self.generate_p_value_section()
        notes_text += "</tr>"
        notes_text += self.generate_additional_notes()
        return notes_text

    def generate_p_value_section(self):
        notes_text = f'<td colspan="{self.num_models}" style="text-align: right">'
        pval_cells = [
            self._format_sig_icon(self.sig_levels[idx] - 0.001)
            + "p&lt;"
            + str(self.sig_levels[idx])
            for idx in range(3)
        ]
        notes_text += "; ".join(pval_cells)
        notes_text += "</td>"
        return notes_text

    def generate_additional_notes(self):
        notes_text = ""
        if len(self.custom_notes) == 0:
            return notes_text
        i = 0
        for i, note in enumerate(self.custom_notes):
            if (i != 0) | (self.notes_append):
                notes_text += "<tr>"
            notes_text += (
                '<td colspan="'
                + str(self.num_models + 1)
                + '" style="text-align: right">'
                + note
                + "</td></tr>"
            )

        return notes_text


class LaTeXRenderer(Renderer):
    fmt = "LaTeX"

    # LaTeX escape characters, borrowed from pandas.io.formats.latex
    _ESCAPE_CHARS = [
        ("\\", r"\textbackslash "),
        ("_", r"\_"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde "),
        ("^", r"\textasciicircum "),
        ("&", r"\&"),
    ]

    def _escape(self, text):
        """
        Escape LaTeX special characters
        """

        if isinstance(text, Label) and text.specific():
            # Do _not_ escape characters of a LaTeX-specific string:
            return text

        if self.kwargs.get("escape", False):
            for orig_char, escape_char in LaTeXRenderer._ESCAPE_CHARS:
                text = text.replace(orig_char, escape_char)
        return text

    def render(self, only_tabular=False, insert_empty_rows=False):
        latex = self.generate_header(only_tabular=only_tabular)
        latex += self.generate_body(insert_empty_rows=insert_empty_rows)
        latex += self.generate_footer(only_tabular=only_tabular)

        return latex

    def generate_header(self, only_tabular=False):
        header = ""
        if not only_tabular:
            header += "\\begin{table}[!htbp] \\centering\n"
            if not self.show_header:
                return header

            if self.title_text is not None:
                header += "  \\caption{" + self.title_text + "}\n"

            if self.table_label is not None:
                header += "  \\label{" + self.table_label + "}\n"

        content_columns = "c" * self.num_models
        header += "\\begin{tabular}{@{\\extracolsep{5pt}}l" + content_columns + "}\n"
        header += "\\\\[-1.8ex]\\hline\n"
        header += "\\hline \\\\[-1.8ex]\n"
        header += self.generate_custom_lines(LineLocation.HEADER_TOP)

        if self.dep_var_name is not None:
            header += (
                f"& \\multicolumn{{{self.num_models}}}{{c}}"
                f"{{\\textit{{{self.dep_var_name}"
                f"{self._escape(self.dependent_variable)}}}}} \\\n"
                f"\\cr \\cline{{2-{self.num_models + 1}}}\n"
            )

        if self.column_labels is not None:
            if type(self.column_labels) == str:
                header += (
                    "\\\\[-1.8ex] & \\multicolumn{"
                    + str(self.num_models)
                    + "}{c}{"
                    + self._escape(self.column_labels)
                    + "} \\\\"
                )
            else:
                header += "\\\\[-1.8ex] "
                for i, label in enumerate(self.column_labels):
                    header += "& \\multicolumn{" + str(self.column_separators[i])
                    header += "}{c}{" + self._escape(label) + "} "
                header += " \\\\\n"

        if self.show_model_nums:
            header += "\\\\[-1.8ex] "
            for num in range(1, self.num_models + 1):
                header += "& (" + str(num) + ") "
            header += "\\\\\n"

        header += self.generate_custom_lines(LineLocation.HEADER_BOTTOM)

        header += "\\hline \\\\[-1.8ex]\n"

        return header

    def _generate_cov_end(self):
        if self.cov_spacing is None:
            return "\\\\\n"
        spacing = self._generate_cov_spacing()
        return f"\\\\[{spacing}]\n"

    def _format_sig_icon(self, pvalue):
        return "$^{" + str(self.get_sig_icon(pvalue)) + "}$"

    def generate_body(self, insert_empty_rows=False):
        """
        Generate the body of the results where the
        covariate reporting is.
        """
        body = ""
        body += self.generate_custom_lines(LineLocation.BODY_TOP)

        cov_end = self._generate_cov_end()

        for cov_name in self.cov_names:
            body += self.generate_cov_rows(cov_name)
            if insert_empty_rows:
                body += "\\\\\n  " + "& " * len(self.num_models)
            body += cov_end
        body += self.generate_custom_lines(LineLocation.BODY_BOTTOM)

        return body

    def generate_cov_rows(self, cov_name):
        cov_text = ""
        cov_text += self.generate_cov_main(cov_name)
        if self.show_precision:
            cov_text += self.generate_cov_precision(cov_name)
        else:
            cov_text += "& "

        return cov_text

    def generate_cov_main(self, cov_name):
        cov_print_name = self.cov_map(cov_name)

        cov_text = f" {self._escape(cov_print_name)} "
        for md in self.model_data:
            if cov_name in md["cov_names"]:
                cov_text += "& " + self._float_format(md["cov_values"][cov_name])
                if self.show_sig:
                    cov_text += self._format_sig_icon(md["p_values"][cov_name])
                cov_text += " "
            else:
                cov_text += "& "

        return cov_text

    def generate_cov_precision(self, cov_name):
        cov_text = "\\\\\n"

        for md in self.model_data:
            if cov_name in md["cov_names"]:
                cov_text += "& ("
                if self.confidence_intervals:
                    cov_text += (
                        self._float_format(md["conf_int_low_values"][cov_name]) + " , "
                    )
                    cov_text += self._float_format(md["conf_int_high_values"][cov_name])
                else:
                    cov_text += self._float_format(md["cov_std_err"][cov_name])
                cov_text += ") "
            else:
                cov_text += "& "

        return cov_text

    def generate_footer(self, only_tabular=False):
        """
        Generate the footer of the table where
        model summary section is.
        """

        footer = "\\hline \\\\[-1.8ex]\n"

        if not self.show_footer:
            return footer
        footer += self.generate_custom_lines(LineLocation.FOOTER_TOP)

        for attr, stat, label in Stargazer._auto_stats:
            if getattr(self, f"show_{attr}"):
                footer += self.generate_stat(stat, label)

        footer += self.generate_custom_lines(LineLocation.FOOTER_BOTTOM)
        footer += "\\hline\n\\hline \\\\[-1.8ex]\n"
        if self.show_notes:
            footer += self.generate_notes()
        footer += "\\end{tabular}"

        if not only_tabular:
            footer += "\n\\end{table}"

        return footer

    def generate_custom_lines(self, location):
        custom_text = ""
        for custom_row in self.custom_lines[location]:
            custom_text += " " + str(custom_row[0]) + " "
            for custom_column in custom_row[1:]:
                custom_text += "& " + str(custom_column) + " "
            custom_text += "\\\\\n"
        return custom_text

    def generate_stat(self, stat, label):
        values = self._generate_stat_values(stat)
        if not any(values):
            return ""

        formatter = self._formatters.get(stat, self._float_format)

        text = f" {label} "
        for value in values:
            if not isinstance(value, str):
                value = formatter(value)
            text += f"& {value} "
        text += "\\\\\n"
        return text

    def generate_notes(self):
        notes_text = ""
        notes_text += "\\textit{" + self.notes_label + "}"
        if self.notes_append and self.show_stars:
            notes_text += self.generate_p_value_section()
        notes_text += self.generate_additional_notes()
        return notes_text

    def generate_p_value_section(self):
        notes_text = " & \\multicolumn{" + str(self.num_models) + "}{r}{"
        pval_cells = [
            self._format_sig_icon(self.sig_levels[idx] - 0.001)
            + "p$<$"
            + str(self.sig_levels[idx])
            for idx in range(3)
        ]
        notes_text += "; ".join(pval_cells)
        notes_text += "} \\\\\n"
        return notes_text

    def generate_additional_notes(self):
        notes_text = ""
        # if len(self.custom_notes) == 0:
        #     return notes_text
        for note in self.custom_notes:
            # if (i != 0) | (self.notes_append):
            #     notes_text += '\\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'
            # else:
            #     notes_text += ' & \\multicolumn{' + str(self.num_models) + '}{r}\\textit{' + note + '} \\\\\n'
            notes_text += (
                "\\multicolumn{"
                + str(self.num_models + 1)
                + "}{r}\\textit{"
                + self._escape(note)
                + "} \\\\\n"
            )

        return notes_text


class ThreePartRenderer(LaTeXRenderer):
    """
    A LaTeX renderer that produces tables in threeparttable format with siunitx alignment.
    Outputs properly formatted LaTeX code that can be further processed by latexindent.
    """

    def render(
        self,
        standalone=False,
        extra_packages=None,
        extra_siunitx_options=None,
        margin="20pt",
    ):
        """Render the table, optionally as a standalone LaTeX document.

        Parameters
        ----------
        standalone : bool, default False
            If True, wraps the table in a complete LaTeX document with preamble
        extra_packages : list of str, optional
            Additional LaTeX packages to include in the preamble
        extra_siunitx_options : list of str, optional
            Additional siunitx package options to configure table formatting
        margin : str, default "20pt"
            Document margin setting when standalone=True

        Returns
        -------
        str
            LaTeX code for the table. If standalone=True, includes full document structure

        See Also
        --------
        _render_table : Generates just the threeparttable LaTeX content
        _render_standalone : Wraps table in standalone document with preamble
        """

        table = self._render_table()
        if standalone:
            return self._render_standalone(
                table,
                extra_packages=extra_packages,
                extra_siunitx_options=extra_siunitx_options,
                margin=margin,
            )
        return table

    def _render_table(self):
        """Generate the threeparttable LaTeX table."""

        table = r"\begin{threeparttable}" + "\n"
        table += self.generate_header()
        table += self.generate_body()
        table += self.generate_footer()
        table += self.generate_notes_section()
        table += r"\end{threeparttable}"
        return table

    def _get_siunitx_precision(self):
        """Calculate the precision settings for siunitx table-format option.

        This function determines the number of digits before and after the decimal point:
        - Integer digits are calculated by finding the maximum number of digits
        before the decimal point across all coefficient values
        - Decimal places are controlled by self.sig_digits, which can be set using
        the significant_digits() method

        Returns
        -------
        tuple of int
            integer_digits : Maximum number of digits before decimal point
            decimal_places : Number of decimal places from self.sig_digits

        See Also
        --------
        significant_digits : Method to set the number of decimal places
        """
        cov_values = list(
            chain.from_iterable([md["cov_values"].values for md in self.model_data])
        )
        split_numbers = [str(num).split(".") for num in cov_values]

        integer_digits = max(len(parts[0]) for parts in split_numbers)

        return integer_digits, self.sig_digits

    def _render_standalone(
        self,
        table,
        extra_packages=None,
        extra_siunitx_options=None,
        margin="20pt",
        integer_digits=None,
    ):
        """Wrap table in standalone LaTeX document with configured siunitx.

        Parameters
        ----------
        table : str
            LaTeX table content to wrap in document
        extra_packages : list of str, optional
            Additional LaTeX packages to include in preamble beyond defaults:
            booktabs, threeparttable, amsmath, siunitx
        extra_siunitx_options : list of str, optional
            Additional siunitx package options beyond defaults:
            input-symbols, table-format, table-auto-round, table-space-text
        margin : str, default "20pt"
            Document margin setting
        integer_digits : int, optional
            Number of digits before decimal point for siunitx table-format.
            If not provided, automatically calculated from coefficient values.

        Returns
        -------
        str
            Complete LaTeX document with preamble and table
        """

        auto_integer_digits, decimal_places = self._get_siunitx_precision()
        integer_digits = (
            integer_digits if integer_digits is not None else auto_integer_digits
        )

        packages = [
            "booktabs",
            "threeparttable",
            "amsmath",
            "siunitx",
        ]
        siunitx_options = [
            "input-symbols=()",
            f"table-format={integer_digits}.{decimal_places}***",
            "table-auto-round=false",
            "table-space-text-pre=()",
            "table-space-text-post=()",
        ]

        if extra_packages is not None:
            packages = extra_packages + packages

        if extra_siunitx_options is not None:
            siunitx_options = siunitx_options + extra_siunitx_options

        packages = "\n".join([rf"\usepackage{{{package}}}" for package in packages])
        siunitx_options = "\n".join([f"{option}," for option in siunitx_options])

        standalone = (
            rf"\documentclass[margin={margin}]{{standalone}}"
            "\n\n"
            f"{packages}"
            "\n\n"
            r"\sisetup{"
            "\n"
            f"{siunitx_options}"
            "\n"
            r"}"
            "\n"
            r"\begin{document}"
            "\n\n"
            f"{table}"
            "\n\n"
            r"\end{document}"
        )

        return standalone

    def save_to_file(
        self,
        filepath,
        standalone=True,
        extra_packages=None,
        extra_siunitx_options=None,
        margin="20pt",
    ):
        """Save rendered table to a LaTeX file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path where the LaTeX file should be saved
        standalone : bool, default True
            If True, saves as complete LaTeX document
        extra_packages : list of str, optional
            Additional LaTeX packages to include in preamble beyond defaults:
            booktabs, threeparttable, amsmath, siunitx
        extra_siunitx_options : list of str, optional
            Additional siunitx package options beyond defaults:
            input-symbols, table-format, table-auto-round, table-space-text
        margin : str, default "20pt"
            Document margin setting

        Returns
        -------
        pathlib.Path
            Path to the saved file

        Examples
        --------
        >>> renderer.save_to_file("table1.tex")
        >>> renderer.save_to_file("table2.tex", standalone=False)
        """

        latex = self.render(
            standalone=standalone,
            extra_packages=extra_packages,
            extra_siunitx_options=extra_siunitx_options,
            margin=margin,
        )

        # Save to file
        Path(filepath).write_text(latex)

    def generate_body(self):
        """Generate the body section of the LaTeX table.

        The body contains:
        1. Pre-estimation custom lines
        2. Coefficient estimates with their significance
        3. Standard errors or confidence intervals
        4. Post-estimation custom lines

        Returns
        -------
        str
            LaTeX formatted table bodyproper spacing and formatting between coefficients and custom lines.
        """
        body = ""

        # Add any custom pre-estimation lines
        body += self.generate_custom_lines(LineLocation.BODY_TOP)

        # Add coefficient estimates
        for cov_name in self.cov_names:
            # Main coefficient row
            body += self.generate_cov_main(cov_name)
            # Standard error row
            if self.show_precision:
                body += self.generate_cov_precision(cov_name)

        # Add midrule after coefficients
        body += r"\midrule" + "\n"

        # Add any custom post-estimation lines including controls and fixed effects
        body += self.generate_custom_lines(LineLocation.BODY_BOTTOM)

        return body

    def generate_header(self):
        """
        Generate header with S columns and booktabs rules.
        Uses column_labels to display dependent variables above model numbers.
        """
        if not isinstance(self.num_models, int) or self.num_models < 1:
            raise ValueError("Number of models must be a positive integer")

        content_columns = "S" * self.num_models
        header = r"\begin{tabular}{l" + content_columns + "}\n"
        header += r"\toprule" + "\n"

        # Handle dependent variables using column_labels
        if self.column_labels is not None:
            header += "& "
            dep_vars = []
            for label in self.column_labels:
                dep_vars.append(f"{{{self._escape(label)}}}")
        else:
            dep_vars = [
                f"& {{{self._escape(dep)}}}" for dep in self.dependent_variables
            ]

        header += " ".join(dep_vars) + r" \\" + "\n"
        header += rf"\cmidrule{{2-{1 + len(self.dependent_variables)}}}" + "\n"

        # Model numbers below dependent variables, now with curly braces
        if self.show_model_nums:
            numbers = [f"& {{({num})}}" for num in range(1, self.num_models + 1)]
            header += " ".join(numbers) + r" \\" + "\n"

        header += r"\midrule" + "\n"
        return header

    def generate_cov_main(self, cov_name):
        """Generate main coefficient row for a covariate.

        Parameters
        ----------
        cov_name : str
            Name of the covariate to generate row for

        Returns
        -------
        str
            LaTeX formatted row with coefficient values and significance stars
        """
        cov_print_name = self.cov_map(cov_name)
        cov_text = f"{self._escape(cov_print_name)} "

        for md in self.model_data:
            if cov_name in md["cov_names"]:
                value = self._float_format(md["cov_values"][cov_name])
                stars = (
                    self._format_sig_icon(md["p_values"][cov_name])
                    if self.show_sig
                    else ""
                )
                cov_text += f"& {value}{stars} "
            else:
                cov_text += "& "

        return cov_text

    def generate_cov_precision(self, cov_name):
        """Generate precision statistics row for a covariate.

        Parameters
        ----------
        cov_name : str
            Name of the covariate to generate precision stats for

        Returns
        -------
        str
            LaTeX formatted row with standard errors or confidence intervals

        Notes
        -----
        Displays either standard errors or confidence intervals based on
        self.confidence_intervals setting
        """
        cov_text = r"\\" + "\n"

        for md in self.model_data:
            if cov_name in md["cov_names"]:
                if self.confidence_intervals:
                    low = self._float_format(md["conf_int_low_values"][cov_name])
                    high = self._float_format(md["conf_int_high_values"][cov_name])
                    cov_text += f"& ({low}, {high}) "
                else:
                    se = self._float_format(md["cov_std_err"][cov_name])
                    cov_text += f"& ({se}) "
            else:
                cov_text += "& "

        cov_text += r"\\" + "\n"

        return cov_text

    def generate_footer(self):
        """Generate the footer section of the LaTeX table.

        Returns
        -------
        str
            LaTeX formatted footer with statistics and bottom rules

        Notes
        -----
        Includes:
        - Custom lines at footer top
        - Automated statistics if show_footer is True
        - Custom lines at footer bottom
        - Bottom rule and table end
        """
        footer = ""

        if self.show_footer:
            footer += self.generate_custom_lines(LineLocation.FOOTER_TOP)

            for attr, stat, label in self._auto_stats:
                if getattr(self, f"show_{attr}"):
                    footer += self.generate_stat(stat, label)

            footer += self.generate_custom_lines(LineLocation.FOOTER_BOTTOM)

        footer += r"\bottomrule" + "\n"
        footer += r"\end{tabular}" + "\n"
        return footer

    def generate_notes_section(self):
        """Generate the notes section explaining table notation.

        Returns
        -------
        str
            LaTeX formatted table notes with standard error and significance explanations
        """
        notes = r"\begin{tablenotes}" + "\n"

        if self.confidence_intervals:
            statistic = "Confidence intervals"
        else:
            statistic = "Standard errors"

        notes += rf"\item \raggedleft \small {statistic} in parentheses" + "\n"
        notes += (
            r"\item \raggedleft $^{*}p < 0.10$, \ $^{**}p < 0.05$, \ $^{***}p < 0.01$"
            + "\n"
        )
        notes += r"\end{tablenotes}" + "\n"
        return notes

    def generate_stat(self, stat, label):
        """Generate a statistics row for the table footer.

        Parameters
        ----------
        stat : str
            Name of the statistic to generate
        label : str
            Label to display for the statistic

        Returns
        -------
        str
            LaTeX formatted row with the statistic values across models

        Notes
        -----
        - Handles both numeric and text values
        - Uses custom formatters if defined for the statistic
        - Returns empty string if no values are present
        """
        values = self._generate_stat_values(stat)
        if not any(values):
            return ""

        formatter = self._formatters.get(stat, self._float_format)

        text = f"{label} "
        for value in values:
            if not isinstance(value, str):
                value = formatter(value)
            if value == "Yes" or not value.replace(".", "").isdigit():
                # Protect any non-numeric entries
                text += f"& {{{value}}} "
            else:
                # Format numeric values with \num
                text += f"& {{\\num{{{value}}}}} "
        text += r"\\" + "\n"
        return text

    def _format_sig_icon(self, pvalue):
        """Format significance level indicator for coefficient values.

        Parameters
        ----------
        pvalue : float
            P-value to determine significance level

        Returns
        -------
        str
            LaTeX formatted significance stars/symbols wrapped in math mode

        Notes
        -----
        Uses get_sig_icon() to determine which symbol to show based on p-value
        Returns empty string if p-value doesn't meet significance threshold
        """
        return "$^{" + str(self.get_sig_icon(pvalue)) + "}$"

    def generate_custom_lines(self, location):
        """Generate custom rows for a specific location in the table.

        Parameters
        ----------
        location : LineLocation
            Enum indicating where in the table to place the custom lines:
            - BODY_TOP: Above coefficient estimates
            - BODY_BOTTOM: Below coefficient estimates
            - FOOTER_TOP: Above footer statistics
            - FOOTER_BOTTOM: Below footer statistics

        Returns
        -------
        str
            LaTeX formatted rows of custom content

        Notes
        -----
        Each custom row is a list where:
        - First element is the row label
        - Subsequent elements are column values
        Empty values are rendered as blank cells
        All non-empty values are wrapped in curly braces for LaTeX
        """
        custom_text = ""
        for custom_row in self.custom_lines[location]:
            custom_text += str(custom_row[0]) + " "
            for custom_column in custom_row[1:]:
                if custom_column:  # If not empty
                    custom_text += f"& {{{custom_column}}} "
                else:
                    custom_text += "& "
            custom_text += r"\\" + "\n"
        return custom_text
