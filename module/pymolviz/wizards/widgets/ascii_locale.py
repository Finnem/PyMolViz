"""Force '.' as the decimal separator so 1.000 is not read as one thousand."""


def apply_ascii_float_locale(spin, QtCore) -> None:
    """Use the C locale and hide grouping separators on a numeric spinbox."""
    if spin is None or QtCore is None:
        return
    locale = QtCore.QLocale(QtCore.QLocale.C)
    omit = getattr(QtCore.QLocale, "OmitGroupSeparator", None)
    if omit is not None:
        options = locale.numberOptions() if hasattr(locale, "numberOptions") else omit
        try:
            locale.setNumberOptions(options | omit if options is not omit else omit)
        except TypeError:
            locale.setNumberOptions(omit)
    spin.setLocale(locale)
    setter = getattr(spin, "setGroupSeparatorShown", None)
    if callable(setter):
        setter(False)
    configure_committed_spin(spin)


def configure_committed_spin(spin) -> None:
    """Emit valueChanged only after Enter, focus loss, or a step click."""
    setter = getattr(spin, "setKeyboardTracking", None)
    if callable(setter):
        setter(False)
