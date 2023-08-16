class _LabelContext:
    """
    A context manager determining what format any Label must be rendered in.
    """

    def __init__(self, fmt):
        """
        Store format for context execution.
        """

        self._fmt = fmt

    def __enter__(self):
        """
        Overwrite Label methods as to use the chosen format.
        """

        self._orig = {param : getattr(Label, param)
                      for param in ['__repr__', 'specific']}

        Label.__repr__ = lambda l : l._versions.get(self._fmt,
                                                    l._versions[None])

        def specific(label):
            """
            Return a bool indicating whether the passed "label" has a specific
            version for the currently used format.
            """

            return self._fmt in label._versions

        Label.specific = specific

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Restore the original Label methods.
        """

        Label.__repr__, Label.specific = [self._orig[param] for param in
                                          ['__repr__', 'specific']]


class Label:
    """
    A string (including possible placeholders) that can be represented in
    different formats.
    Formats can be set by using Label.context, as in

    with Label.context('html'):
        ...
    """
    context = _LabelContext

    def __init__(self, label):
        if isinstance(label, dict):
            self._versions = label

            # A default is required:
            if not None in label:
                # If not provided, pick the first:
                label[None] = label[list(label)[0]]
        else:
            self._versions = {None : label}

    def specific(self):
        """
        To be used only within a Label.context.
        """

        raise ValueError("specific() can only be called within a "
                         "Label.context")

    def __getattr__(self, attr):
        return getattr(str(self), attr)

    def __repr__(self):
        return self._versions[None]
