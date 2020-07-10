What's new in 0.0.5 (July 10, 2020)
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
