What's new in 0.0.6 (July 28, 2023)
-----------------------------------

This release brings a large refactoring, deduplicating much code between HTML
and LateX output and potentially simplifying the future support for other
output formats.

In addition, it introduces a large number of enhancements and bug fixes.

Note that the project URL on GitHub has moved to
`https://github.com/StatsReporting/stargazer <https://github.com/StatsReporting/stargazer>`_

Enhancements
^^^^^^^^^^^^

- Support escaping LaTeX special characters in ``render_latex()`` (:issue:`60`)
- Allow for custom spacing between coefficient rows (:issue:`66`)
- Support the result of ``get_robustcov_results()`` from ``Statsmodels``
- Implement more general computation of residual standard error (:issue:`76`)
- Support for statistical results of other models than just OLS
- Introduction of a ``utils.Wrapper`` class allowing easy customization of any given model (:issue:`71`)
- Make ``separators`` argument to ``custom_columns()`` optional
- Allow changing the name of the dependent variable (:issue:`93`)
- Support custom lines (also) in header

Bug fixes
^^^^^^^^^

- Fix width of line above mode numbers (:issue:`75`)
- Fix display of dependent variable in LaTeX
- Fix rendering of ``generate_additional_notes()`` (:issue:`80`)
- Escape special characters (also) in columns labels (:issue:`90`)
- Fix typo in ``LineLocation`` value

Contributors
~~~~~~~~~~~~

The following people contributed commits to this release:

* Pietro Battiston
* Christoph Semken
* Isaac Liu
* Leo Goldman
* Mathias Ruoss
* Alex Dong


What's new in 0.0.5 (July 13, 2020)
-----------------------------------

This release introduced a large number of enhancements and bug fixes.

Enhancements
^^^^^^^^^^^^

- Support for multiple different target variables
- Empty rows for spacing are now optional
- Enabled rich display for HTML tables inside Jupyter notebooks
- Support models with missing features, hence beyond OLS
- Displayed thousands separator in number of observations
- Added ``table_label`` attribute and hidden the label if empty
- Added the ``show_stars`` attribute to not features significance signs
- Added the ``add_line()`` method to insert custom content
- Added first unit tests


Bug fixes
^^^^^^^^^

- Honored behavior of several ``show_*`` attributes
- Adapted creation of multiple columns in LaTeX
- Show dependent variable name in HTML
- Fixed representation of floating numbers, keeping constant the number of decimal digits
- Fixed the number of observations in models without constant (:issue:`32`)
- Fixed output of R^2 in LaTeX
- Workaround for statsmodels bug #6778 affecting output of f statistic (:issue:`33`)


Contributors
~~~~~~~~~~~~

The following people contributed patches to this release:

* Pietro Battiston
* Christoph Semken
* Brandon Zborowski
* Max Ghenis
