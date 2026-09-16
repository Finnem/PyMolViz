from pymolviz.wizards.widgets.spin_step import (
    apply_magnitude_step,
    apply_peer_steps,
    apply_spin_step,
    decimals_for_step,
    nice_step,
    radius_step,
    step_for_magnitude,
    step_for_spin_value,
    step_for_values,
)


class _Spin:
    def __init__(self, value, *, step=1.0, decimals=2):
        self._value = float(value)
        self._step = float(step)
        self._decimals = int(decimals)
        self.changed = []

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = float(value)
        for cb in self.changed:
            cb(self._value)

    def singleStep(self):
        return self._step

    def setSingleStep(self, step):
        self._step = float(step)

    def decimals(self):
        return self._decimals

    def setDecimals(self, decimals):
        self._decimals = int(decimals)

    class _Signal:
        def __init__(self, owner):
            self._owner = owner

        def connect(self, callback):
            self._owner.changed.append(callback)

    @property
    def valueChanged(self):
        return _Spin._Signal(self)


def test_nice_step_uses_1_2_5():
    assert nice_step(0.00012) == 0.0001
    assert nice_step(0.0004) == 0.0005
    assert nice_step(0.12) == 0.1
    assert nice_step(0.25) == 0.2
    assert nice_step(0.6) == 0.5
    assert nice_step(1.2) == 1.0
    assert nice_step(3.0) == 2.0
    assert nice_step(7.0) == 5.0


def test_density_scale_steps():
    assert step_for_spin_value(0.005) == 0.0005
    assert step_for_values([0.0, 0.005]) == 0.0001
    lo = step_for_spin_value(0.0, companions=(0.0, 0.005))
    hi = step_for_spin_value(0.005, companions=(0.0, 0.005))
    assert lo == 0.0001
    assert hi == 0.0001


def test_wide_range_keeps_fine_step_on_small_value():
    step = step_for_spin_value(0.005, companions=(0.0, 9.4))
    assert step == 0.0005


def test_zero_uses_span_or_fallback():
    assert step_for_spin_value(0.0, companions=(0.0, 9.4)) == 0.1
    assert step_for_spin_value(0.0) == 0.1


def test_decimals_cover_the_step():
    assert decimals_for_step(0.0001, min_decimals=2) == 4
    assert decimals_for_step(0.1, min_decimals=2) == 2
    assert decimals_for_step(1.0, min_decimals=2) == 2


def test_apply_peer_steps_on_density_limits():
    vmin = _Spin(0.0, step=0.1, decimals=4)
    vmax = _Spin(0.005, step=0.1, decimals=4)
    apply_peer_steps(vmin, vmax, min_decimals=4)
    assert vmin.singleStep() == 0.0001
    assert vmax.singleStep() == 0.0001
    assert vmin.decimals() >= 4
    assert vmax.decimals() >= 4


def test_apply_magnitude_step_tracks_iso():
    spin = _Spin(0.005, step=0.1, decimals=4)
    apply_magnitude_step(spin, min_decimals=4)
    assert spin.singleStep() == 0.0005
    assert spin.decimals() >= 4


def test_fixed_step_helper():
    spin = _Spin(1.0, step=1.0, decimals=2)
    apply_spin_step(spin, 0.05, decimals=3)
    assert spin.singleStep() == 0.05
    assert spin.decimals() == 3


def test_radius_step_is_finer_for_small_radii():
    assert radius_step(0.3) == 0.01
    assert radius_step(1.7) == 0.1
    assert radius_step(25.0) == 1.0


def test_from_selection_iso_steps():
    from pymolviz.wizards.builders.field_params import iso_spin_range
    from pymolviz.fields.identity import (
        GEN_DISTANCE,
        GEN_GAUSSIAN,
        GEN_NEAREST_PROP,
        GEN_SIGNED_VDW,
    )

    assert iso_spin_range(GEN_GAUSSIAN)[2] == 0.05
    assert iso_spin_range(GEN_DISTANCE)[2] == 0.1
    assert iso_spin_range(GEN_SIGNED_VDW)[2] == 0.1
    assert iso_spin_range(GEN_NEAREST_PROP)[2] == 0.1


def test_unbounded_dummy_range_uses_magnitude():
    assert step_for_values([-1e8, 1e8]) == step_for_magnitude(1e8)
    assert step_for_magnitude(1.2) == 0.1
