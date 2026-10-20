"""Factor-level designation syntax, algebra, and safety tests."""

import copy
import pickle
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pint
import pytest

from xpint import DesignationTerm, SemanticQuantity, SemanticRegistry
from xpint.units import J, L, N, degC, m, mL, mM, mmol, mol, ms, s, ureg


@pytest.mark.parametrize("automatic", [False, True])
def test_automatic_reduction_keeps_factor_scale_consistent(automatic: bool) -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=automatic)
    original_units = registry.meter["a"] / registry.centimeter["b"]
    result = registry.Quantity(1, registry.meter["a"]) / registry.Quantity(
        1, registry.centimeter["b"]
    )
    assert str(result.units) == ("meter[a] / meter[b]" if automatic else "meter[a] / centimeter[b]")
    assert result.magnitude == pytest.approx(100 if automatic else 1, rel=1e-15, abs=0.0)
    assert result.to(original_units).magnitude == pytest.approx(1, rel=1e-15, abs=0.0)
    assert result.as_undesignated().to_root_units().magnitude == pytest.approx(
        100, rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("kind", [None, "Length"])
def test_automatic_in_place_reduction_preserves_designations_and_alias(kind: str | None) -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    quantity = registry.Quantity(np.array([1.0, 2.0]), registry.meter["a"], kind=kind)
    alias = quantity.magnitude
    _ = quantity.dimensionality
    quantity /= registry.Quantity(1, registry.centimeter["b"])
    assert str(quantity.units) == "meter[a] / meter[b]"
    assert quantity.dimensionless
    assert quantity.kind is None
    assert quantity.magnitude is alias
    np.testing.assert_allclose(alias, [100.0, 200.0], rtol=1e-15, atol=0.0)


def test_automatic_reduction_preserves_derived_unit_residual_scale() -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    ratio = registry.Quantity(1, registry.centimeter["a"] / registry.meter["b"])
    result = registry.Quantity(1, registry.newton) * ratio
    assert str(result.units) == "meter[a] * newton / meter[b]"
    assert result.designation_signature == ratio.designation_signature
    assert result.magnitude == pytest.approx(0.01, rel=1e-15, abs=0.0)
    assert result.as_undesignated().to("newton").magnitude == pytest.approx(
        0.01, rel=1e-15, abs=0.0
    )


def test_automatic_scalar_scaling_reconciles_existing_compound_factors() -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    quantity = registry.Quantity(1, registry.meter["a"] / registry.centimeter["b"])
    result = quantity * 2
    assert str(result.units) == "meter[a] / meter[b]"
    assert result.magnitude == pytest.approx(200, rel=1e-15, abs=0.0)


def test_automatic_reduction_cancels_matching_designations() -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    result = registry.Quantity(1, registry.meter["a"]) / registry.Quantity(
        1, registry.centimeter["a"]
    )
    assert result.is_semantic_unity
    assert str(result.units) == "dimensionless"
    assert result.magnitude == pytest.approx(100, rel=1e-15, abs=0.0)


def test_automatic_in_place_reduction_cast_failure_preserves_state() -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    quantity = registry.Quantity(
        np.array([2, 4], dtype=np.int64), registry.meter["a"], kind="Length"
    )
    alias = quantity.magnitude
    with pytest.raises(TypeError):
        quantity /= registry.Quantity(2, registry.centimeter["b"])
    assert quantity.magnitude is alias
    assert quantity.kind == "Length"
    assert str(quantity.units) == "meter[a]"
    np.testing.assert_array_equal(alias, [2, 4])


@pytest.mark.parametrize("inplace", [False, True])
def test_explicit_reduction_preserves_designations(inplace: bool) -> None:
    registry = SemanticRegistry()
    quantity = registry.Quantity(
        np.array([1.0, 2.0]), registry.meter["a"] / registry.centimeter["b"]
    )
    if inplace:
        quantity.ito_reduced_units()
        result = quantity
    else:
        result = quantity.to_reduced_units()
    assert str(result.units) == "meter[a] / meter[b]"
    np.testing.assert_allclose(result.magnitude, [100.0, 200.0], rtol=1e-15, atol=0.0)


def test_reflected_division_retains_pint_scale_until_explicit_reduction() -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    quantity = registry.Quantity(1, registry.centimeter["a"] / registry.meter["b"])
    result = 1 / quantity
    assert str(result.units) == "meter[b] / centimeter[a]"
    assert result.magnitude == pytest.approx(1, rel=1e-15, abs=0.0)
    reduced = result.to_reduced_units()
    assert str(reduced.units) == "meter[b] / meter[a]"
    assert reduced.magnitude == pytest.approx(100, rel=1e-15, abs=0.0)


def test_reflected_offset_division_reconciles_factors() -> None:
    registry = SemanticRegistry(autoconvert_offset_to_baseunit=True)
    registry.define("shifted_length = meter; offset: 10")
    result = 1 / registry.Quantity(2, registry.shifted_length["a"])
    assert str(result.units) == "1 / meter[a]"
    assert result.magnitude == pytest.approx(1 / 12, rel=1e-15, abs=0.0)


def test_non_reducible_compound_returns_new_wrapper_like_pint() -> None:
    quantity = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"] * ureg.second)
    result = quantity.to_reduced_units()
    assert result is not quantity
    assert result.magnitude is quantity.magnitude
    assert result.units == quantity.units


def test_explicit_reduction_no_op_preserves_identity() -> None:
    quantity = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"], kind="Length")
    alias = quantity.magnitude
    assert quantity.to_reduced_units() is quantity
    quantity.ito_reduced_units()
    assert quantity.magnitude is alias
    assert quantity.kind == "Length"


def test_automatic_reduction_preserves_masked_array_shape_and_dtype() -> None:
    registry = SemanticRegistry(auto_reduce_dimensions=True)
    magnitude = np.ma.array([1.0, 2.0], mask=[False, True], dtype=np.float32)
    quantity = registry.Quantity(magnitude, registry.meter["a"])
    quantity /= registry.Quantity(1, registry.centimeter["b"])
    assert quantity.magnitude is magnitude
    assert quantity.shape == (2,)
    assert quantity.magnitude.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(quantity.magnitude.mask, [False, True])
    assert quantity[0].magnitude == pytest.approx(100, rel=1e-7, abs=0.0)
    assert str(quantity.units) == "meter[a] / meter[b]"


def test_factor_subscript_builds_canonical_signature() -> None:
    oxygen = mmol["O2"]

    assert oxygen.has_designation
    assert not oxygen.is_semantic_unity
    assert oxygen.designation_signature == (DesignationTerm("[substance]", "O2", Fraction(1)),)
    assert str(oxygen) == "millimole[O2]"
    assert format(oxygen, "~") == "mmol[O2]"


@pytest.mark.parametrize("magnitude", [1, np.array([1, 2], dtype=np.int64)])
def test_quantity_unit_alias_preserves_designation(magnitude: Any) -> None:
    assert ureg.Quantity(magnitude, mmol).u == mmol
    quantity = ureg.Quantity(magnitude, mmol["O2"])

    assert quantity.u.designation_signature == quantity.units.designation_signature
    assert quantity.u == mmol["O2"]
    converted = ureg.Quantity(magnitude, quantity.u).to(mmol)
    assert converted.designation_signature == quantity.designation_signature


def test_designations_are_opaque_and_case_sensitive() -> None:
    assert mmol["O2"] != mmol["o2"]
    assert mmol["O2"].designation_signature != mmol["oxygen"].designation_signature
    assert "blood plasma" in str(mL["blood plasma"])


def test_millimolar_declares_substance_designation_slot() -> None:
    alias = mM["O2"]
    decomposed = mmol["O2"] / L

    assert alias.designation_signature == decomposed.designation_signature
    assert alias.is_compatible_with(decomposed)
    assert str(alias) == "millimolar[O2]"
    assert format(alias, "~") == "mM[O2]"


@pytest.mark.parametrize(
    ("text", "prefix", "symbol"), [("uM", "micro", "µM"), ("nM", "nano", "nM"), ("M", "", "M")]
)
def test_prefixed_molar_units_inherit_substance_designation_slot(
    text: str, prefix: str, symbol: str
) -> None:
    alias = ureg.Unit(f"{text}[Fe]")
    decomposed = ureg.Unit(f"{prefix}mol[Fe]") / L

    assert alias.designation_signature == decomposed.designation_signature
    assert str(alias) == f"{prefix}molar[Fe]"
    assert format(alias, "~") == f"{symbol}[Fe]"
    assert ureg.Unit(str(alias)) == alias
    assert pickle.loads(pickle.dumps(alias)) == alias
    # One prefix rescaling plus one conversion; allow floating roundoff.
    expected = ureg.Quantity(1, f"{text}").m_as("mM")
    converted = ureg.Quantity(1, alias).to("mM[Fe]")
    assert converted.units == mM["Fe"]
    assert converted.magnitude == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_exact_designation_slot_takes_precedence_over_base_unit_slot() -> None:
    registry = SemanticRegistry()
    registry.define_designation_slot("kilometer", "[length]", exponent=2)

    assert registry.km["x"].designation_signature != registry.m["x"].designation_signature
    assert registry.cm["x"].designation_signature == registry.m["x"].designation_signature


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    [
        (lambda: mmol.with_designation(""), ValueError, "must not be empty"),
        (lambda: mmol.with_designation(cast(Any, 3)), TypeError, "must be a string"),
        (lambda: (mmol / L).with_designation("O2"), ValueError, "one unit factor"),
        (lambda: N.with_designation("force"), ValueError, "physical dimensions"),
        (lambda: ureg.dimensionless.with_designation("x"), ValueError, "one unit factor"),
        (lambda: degC.with_designation("body"), ValueError, "absolute temperature"),
    ],
)
def test_invalid_designation_targets_are_rejected(
    operation: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        operation()


def test_same_designation_cancels_to_semantic_unity() -> None:
    ratio = mL["blood"] / mL["blood"]

    assert ratio.dimensionless
    assert not ratio.has_designation
    assert ratio.is_semantic_unity
    assert ratio.designation_signature == ()


def test_different_designations_remain_physically_dimensionless() -> None:
    ratio = mL["blood"] / mL["tissue"]

    assert ratio.dimensionless
    assert ratio.has_designation
    assert not ratio.is_semantic_unity
    assert ratio.designation_signature == (
        DesignationTerm("[length]", "blood", Fraction(3)),
        DesignationTerm("[length]", "tissue", Fraction(-3)),
    )
    assert str(ratio) == "milliliter[blood] / milliliter[tissue]"


def test_designated_and_undesignated_factors_do_not_cancel_semantically() -> None:
    ratio = mL["blood"] / mL

    assert ratio.dimensionless
    assert ratio.has_designation
    assert not ratio.is_semantic_unity
    assert "milliliter[blood]" in str(ratio)
    assert " / milliliter" in str(ratio)


def test_same_designation_cancels_across_prefixes() -> None:
    ratio = mmol["O2"] / mol["O2"]

    assert ratio.dimensionless
    assert not ratio.has_designation
    assert ratio.is_semantic_unity
    assert "[" not in str(ratio)


def test_designations_propagate_through_unit_power() -> None:
    area = m["tissue"] ** 2
    inverse = 1 / m["tissue"]

    assert area.designation_signature == (DesignationTerm("[length]", "tissue", Fraction(2)),)
    assert inverse.designation_signature == (DesignationTerm("[length]", "tissue", Fraction(-1)),)
    assert not (m["tissue"] ** 0).has_designation


def test_fractional_designation_power_formats_unambiguously() -> None:
    rooted = m["tissue"] ** Fraction(1, 2)

    assert str(rooted) == "meter[tissue] ** (1/2)"


def test_quantity_construction_and_formatting_preserve_designation() -> None:
    oxygen = 5 * mmol["O2"] / L

    assert oxygen.units.has_designation
    assert oxygen.designation_signature == oxygen.units.designation_signature
    assert str(oxygen) == "5.0 millimole[O2] / liter"
    assert format(oxygen, "~") == "5.0 mmol[O2] / l"
    assert repr(oxygen) == "<Quantity(5.0, 'millimole[O2] / liter')>"
    assert "millimole[O2]" in oxygen._repr_html_()
    assert "\\mathtt" in oxygen._repr_latex_()
    assert "millimole[O2]" in oxygen.units._repr_html_()
    assert "\\mathtt" in oxygen.units._repr_latex_()


def test_same_designation_adds_across_equivalent_units() -> None:
    explicit = 5 * mmol["O2"] / L
    alias = 2 * mM["O2"]

    total = explicit + alias
    difference = explicit - alias

    assert total.magnitude == pytest.approx(7, rel=0.0, abs=1e-12)
    assert difference.magnitude == pytest.approx(3, rel=0.0, abs=1e-12)
    assert total.units == explicit.units
    assert total.designation_signature == explicit.designation_signature


def test_in_place_arithmetic_and_conversion_preserve_designation() -> None:
    oxygen = 5.0 * mM["O2"]

    oxygen += 2 * mM["O2"]
    oxygen *= 2
    oxygen.ito(mmol["O2"] / L)
    oxygen.ito_base_units()

    assert oxygen.magnitude == pytest.approx(14, rel=0.0, abs=1e-12)
    assert oxygen.designation_signature == (DesignationTerm("[substance]", "O2", Fraction(1)),)
    assert "mole[O2] / meter ** 3" in str(oxygen)


def test_addition_rejects_designation_mismatch_and_plain_operand() -> None:
    oxygen = 5 * mM["O2"]

    with pytest.raises(TypeError, match="different factor designations"):
        oxygen + 2 * mM["NaCl"]
    with pytest.raises(TypeError, match="different factor designations"):
        oxygen + 2 * mM
    with pytest.raises(TypeError, match="ordinary scalar"):
        (2 * mL["blood"] / mL["tissue"]) + 1


def test_conversion_requires_matching_signature() -> None:
    oxygen = 5 * mmol["O2"] / L

    converted = oxygen.to(mM["O2"])

    assert converted.magnitude == pytest.approx(5, rel=0.0, abs=1e-12)
    assert converted.units == mM["O2"]
    with pytest.raises(TypeError, match="different factor designations"):
        oxygen.to(mM["NaCl"])
    assert oxygen.to(mM).designation_signature == oxygen.designation_signature


def test_base_conversion_preserves_signature_without_scale_duplication() -> None:
    ratio = (5 * mL["blood"] / L["tissue"]).to_base_units()

    assert ratio.magnitude == pytest.approx(0.005, rel=1e-15, abs=0.0)
    assert ratio.dimensionless
    assert ratio.has_designation
    assert "meter[blood] ** 3" in str(ratio)
    assert "meter[tissue] ** 3" in str(ratio)


def test_equality_and_ordering_apply_designation_policy() -> None:
    blood = 2 * mL["blood"]
    more_blood = 3 * mL["blood"]
    tissue = 2 * mL["tissue"]
    plain = 2 * mL

    assert blood != tissue
    assert blood != plain
    assert blood < more_blood
    with pytest.raises(TypeError, match="different factor designations"):
        _ = blood < tissue
    with pytest.raises(TypeError, match="different factor designations"):
        _ = blood < plain


def test_compatibility_requires_exact_designation_state() -> None:
    blood = 2 * mL["blood"]
    same = 3 * mL["blood"]
    tissue = 2 * mL["tissue"]
    plain = 2 * mL

    assert blood.is_compatible_with(same)
    assert not blood.is_compatible_with(tissue)
    assert not blood.is_compatible_with(plain)
    assert blood.is_compatible_with(mL)
    assert mL["blood"].is_compatible_with(blood)
    assert not mL["blood"].is_compatible_with(plain)


def test_designated_unit_equals_matching_unit_quantity_at_one() -> None:
    assert mM["O2"] == 1 * mM["O2"]
    assert mM["O2"] != 1 * mM["NaCl"]


def test_explicit_removal_restores_plain_pint_behavior() -> None:
    designated_unit = mL["blood"]
    designated_quantity = 2 * designated_unit

    assert designated_unit.as_undesignated() == mL
    assert designated_quantity.as_undesignated() == 2 * mL
    assert not designated_quantity.as_undesignated().has_designation
    assert (2 * mL).with_designation("blood") == designated_quantity


@pytest.mark.parametrize("coercion", [float, int, complex])
def test_scalar_coercion_rejects_retained_designation(coercion: Any) -> None:
    ratio = 2 * mL["blood"] / mL["tissue"]

    with pytest.raises(TypeError, match="retained factor designations"):
        coercion(ratio)
    with pytest.raises(TypeError, match="retained factor designations"):
        coercion(ratio.units)


def test_plain_tuple_serialization_rejects_designation_loss() -> None:
    ratio = 2 * mL["blood"] / mL["tissue"]

    with pytest.raises(TypeError, match="tuple schema cannot represent"):
        ratio.to_tuple()


def test_kind_does_not_survive_designated_dimensionless_multiplier() -> None:
    energy = (3 * J).as_kind("Energy")
    designated_ratio = 2 * mL["blood"] / mL["tissue"]

    assert (energy * designated_ratio).kind is None
    assert (energy * ureg.Quantity(2)).kind == "Energy"


def test_quantity_power_scales_designation_and_drops_kind() -> None:
    length = (2 * m["tissue"]).as_kind("Length")

    area = length**2
    unity = length**0

    assert area.kind is None
    assert area.designation_signature == (DesignationTerm("[length]", "tissue", Fraction(2)),)
    assert not unity.has_designation
    assert unity.is_semantic_unity


def test_scaled_dimensionless_exponent_uses_root_magnitude() -> None:
    result = (4 * m["tissue"]) ** (50 * ureg.percent)

    assert result.magnitude == pytest.approx(2, rel=0.0, abs=1e-12)
    assert result.designation_signature == (DesignationTerm("[length]", "tissue", Fraction(1, 2)),)


def test_copy_iteration_and_pickle_preserve_designations() -> None:
    oxygen = ureg.Quantity(np.array([1.0, 2.0]), mmol["O2"] / L)

    values = [copy.copy(oxygen), copy.deepcopy(oxygen), oxygen[0], next(iter(oxygen))]
    restored_quantity = pickle.loads(pickle.dumps(oxygen))
    restored_unit = pickle.loads(pickle.dumps(mM["O2"]))

    for value in [*values, restored_quantity]:
        assert value.designation_signature == oxygen.designation_signature
    assert restored_unit.designation_signature == mM["O2"].designation_signature


def test_tolist_preserves_kind_and_designation_on_nested_leaves() -> None:
    samples = ureg.Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), m["tissue"], kind="Length")

    nested = samples.tolist()

    assert nested[0][1].magnitude == 2.0
    for row in nested:
        for value in row:
            assert value.kind == "Length"
            assert value.designation_signature == samples.designation_signature


def test_scoped_registry_pickle_reconstructs_against_default_registry() -> None:
    registry = SemanticRegistry()
    unit = registry.meter["sample"]
    quantity = 2 * unit

    assert pickle.loads(pickle.dumps(unit)) == ureg.meter["sample"]
    restored = pickle.loads(pickle.dumps(quantity))
    assert restored.units == ureg.meter["sample"]
    assert restored.magnitude == 2


def test_numpy_reductions_preserve_designation() -> None:
    oxygen = ureg.Quantity(np.array([1.0, 2.0]), mM["O2"])

    total = cast(SemanticQuantity, np.sum(oxygen))
    average = cast(SemanticQuantity, np.mean(oxygen))

    assert total.designation_signature == oxygen.designation_signature
    assert average.designation_signature == oxygen.designation_signature
    with pytest.raises(TypeError, match="no xpint semantic propagation rule"):
        np.log(oxygen)
    variance = cast(SemanticQuantity, np.var(oxygen))
    assert variance.units == mM["O2"] ** 2
    assert variance.magnitude == pytest.approx(0.25, rel=1e-15, abs=0.0)


def test_registry_can_declare_custom_derived_designation_slot() -> None:
    registry = SemanticRegistry()
    registry.define("dose_rate = mole / second")
    registry.define_designation_slot("dose_rate", "[substance]")

    designated = registry.Unit("dose_rate")["agent"]

    assert designated.designation_signature == (
        DesignationTerm("[substance]", "agent", Fraction(1)),
    )


@pytest.mark.parametrize(
    ("unit", "dimension", "exponent", "error", "message"),
    [
        (3, "[substance]", Fraction(1), TypeError, "must both be strings"),
        ("molar", "[substance]", 1.5, TypeError, "integer or Fraction"),
        ("", "[substance]", Fraction(1), ValueError, "must not be empty"),
        ("mole / second", "[substance]", Fraction(1), ValueError, "one Pint unit token"),
        ("meter", "[substance]", Fraction(1), ValueError, "is absent"),
    ],
)
def test_invalid_designation_slot_is_rejected(
    unit: Any,
    dimension: str,
    exponent: Any,
    error: type[Exception],
    message: str,
) -> None:
    registry = SemanticRegistry()

    with pytest.raises(error, match=message):
        registry.define_designation_slot(unit, dimension, exponent=exponent)


def test_isotope_designation_separates_activity_of_different_nuclides() -> None:
    """An isotope is referent identity, so it belongs to the designation mechanism.

    Two activities share a dimension and a scale; only the nuclide distinguishes
    them, which is exactly what a designation carries.
    """

    fluorine = 5 * ureg.becquerel["F18"]
    technetium = 5 * ureg.becquerel["Tc99m"]

    assert fluorine.has_designation
    assert (fluorine + 5 * ureg.becquerel["F18"]).magnitude == 10
    with pytest.raises(TypeError, match="different factor designations"):
        fluorine + technetium


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (".2f", "5.00 millimole[O2] / liter"),
        (".3e", "5.000e+00 millimole[O2] / liter"),
        ("~.2f", "5.00 mmol[O2] / l"),
        (".2f~", "5.00 mmol[O2] / l"),
    ],
)
def test_designated_numeric_formatting(spec: str, expected: str) -> None:
    assert format(5 * mmol["O2"] / L, spec) == expected


def test_designated_array_formatting_and_unit_repr() -> None:
    values = ureg.Quantity(np.array([1.234, 2.345], dtype=np.float64), mmol["O2"])
    assert format(values, ".2f~") == "[1.23 2.35] mmol[O2]"
    assert format(mmol["O2"], "~") == "mmol[O2]"
    assert repr(mmol["O2"]) == "<Unit('millimole[O2]')>"
    assert repr((5 * mmol["O2"]).as_kind("AmountOfSubstance")) == (
        "<Quantity(5, 'millimole[O2]', kind='AmountOfSubstance')>"
    )


@pytest.mark.parametrize("spec", ["~P", "L", "H", "C", "D"])
@pytest.mark.parametrize("quantity", [False, True])
def test_designated_formatting_rejects_unsupported_unit_specs(spec: str, quantity: bool) -> None:
    value = 5 * mmol["O2"] if quantity else mmol["O2"]
    with pytest.raises(TypeError, match="unit specs '' and '~'"):
        format(value, spec)


@pytest.mark.parametrize("target", ["mol/L", ureg.mole / L])
def test_m_as_rejects_a_target_without_the_value_designations(target: Any) -> None:
    oxygen = 5 * mmol["O2"] / L
    with pytest.raises(TypeError, match=r"m_as\(\) needs a target that carries the designations"):
        oxygen.m_as(target)


@pytest.mark.parametrize("target", [None, "s", ureg.second])
def test_m_as_reports_a_plain_target_of_other_dimensions_as_dimensionality_error(
    target: Any,
) -> None:
    oxygen = 5 * mmol["O2"] / L
    with pytest.raises(pint.DimensionalityError):
        oxygen.m_as(target)


def test_m_as_rejects_mismatched_designations_after_cancellation() -> None:
    # The physical units cancel to 1/s, but gadobutrol and gadoteridol do not.
    rate = (4.5 / (mM["gadoteridol"] * s)) * (0.5 * mM["gadobutrol"])
    with pytest.raises(TypeError, match="needs a target that carries the designations"):
        rate.m_as(1 / s)
    with pytest.raises(TypeError, match="same designation signature"):
        rate.m_as(mM["Fe"] / (mM["gadoteridol"] * s))
    with pytest.raises(pint.DimensionalityError):
        rate.m_as(1 / (mM["gadoteridol"] * s))


def test_m_as_returns_the_magnitude_in_a_matching_designated_target() -> None:
    oxygen = 5 * mmol["O2"] / L
    assert oxygen.m_as(mol["O2"] / L) == pytest.approx(0.005, rel=1e-15, abs=0.0)
    relaxivity = 4.5 / (mM["gadoteridol"] * s)
    assert relaxivity.m_as(1 / (mM["gadoteridol"] * ms)) == pytest.approx(
        0.0045, rel=1e-15, abs=0.0
    )


def test_m_as_accepts_a_plain_target_after_designations_cancel_or_are_removed() -> None:
    rate = (4.5 / (mM["gadoteridol"] * s)) * (0.5 * mM["gadoteridol"])
    assert rate.m_as(1 / ms) == pytest.approx(0.00225, rel=1e-15, abs=0.0)
    oxygen = 5 * mmol["O2"] / L
    assert oxygen.as_undesignated().m_as("mol/L") == pytest.approx(0.005, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("target", ["mol/L", ureg.mole / L])
def test_plain_conversion_targets_adopt_designations(target: Any) -> None:
    oxygen = 5 * mmol["O2"] / L
    converted = oxygen.to(target)
    assert converted.designation_signature == oxygen.designation_signature
    assert str(converted.units) == "mole[O2] / liter"
    assert converted.magnitude == pytest.approx(0.005, rel=1e-15, abs=0.0)
    rebuilt = ureg.Quantity(oxygen, target)
    assert rebuilt == converted
    assert oxygen.is_compatible_with(target)
    oxygen.ito(target)
    assert oxygen == converted
    assert oxygen.designation_signature == converted.designation_signature


def test_designations_survive_compact_and_preferred_conversions() -> None:
    oxygen = 5000 * mmol["O2"] / L
    compact = oxygen.to_compact()
    assert compact.designation_signature == oxygen.designation_signature
    assert compact.to(oxygen.units).magnitude == pytest.approx(5000, rel=1e-15, abs=0.0)
    amount = 5000 * mmol["O2"]
    preferred = amount.to_preferred([ureg.mole])
    assert preferred.units == mol["O2"]
    assert preferred.magnitude == pytest.approx(5, rel=1e-15, abs=0.0)
    length = (1 * m["x"]).to("cm")
    assert str(length.units) == "centimeter[x]"
    assert length.magnitude == pytest.approx(100, rel=1e-15, abs=0.0)
    assert not oxygen.is_compatible_with(mol["CO2"] / L)
    assert oxygen.is_compatible_with(mol["O2"] / L)


def test_designated_hashes_match_across_scales_and_unit_quantity_equality() -> None:
    quantity = 1000 * mmol["O2"]
    converted = quantity.to("mol")
    assert hash(quantity) == hash(converted)
    assert {quantity: "oxygen"}[converted] == "oxygen"
    assert mol["O2"] == converted
    assert hash(mol["O2"]) == hash(converted)


@pytest.mark.parametrize("target", ["mM", "M", mM, "mM ** 2"])
def test_plain_slot_targets_adopt_designation_through_their_slot(target: Any) -> None:
    oxygen = 5 * mmol["O2"] / L
    source = oxygen**2 if isinstance(target, str) and "**" in target else oxygen

    converted = source.to(target)

    assert converted == source
    assert converted.units == ureg.Unit(target).with_designation("O2")
    assert "mole[O2] / mole" not in str(converted)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("mmol[X]/(L/min)", "mM*min", "millimolar[X] * minute"),
        ("mmol[X]*min/L", "mM*min", "millimolar[X] * minute"),
        ("mmol[X]*s/L", "uM*s", "micromolar[X] * second"),
        ("mmol[X]", "mM*L", "liter * millimolar[X]"),
    ],
)
def test_compound_plain_targets_adopt_designation_through_a_slot_token(
    source: str, target: str, expected: str
) -> None:
    quantity = ureg.Quantity(np.array([1.0, 2.0]), source)

    converted = quantity.to(target)

    assert str(converted.units) == expected
    assert converted.units == ureg.Unit(expected)
    assert converted.designation_signature == quantity.designation_signature
    assert converted.shape == quantity.shape
    assert np.all(converted == quantity)


def test_single_dimension_target_token_takes_the_label_before_a_slot_token() -> None:
    quantity = ureg.Quantity(1, "mmol[X]/L")
    converted = quantity.to("mol * mM / mmol")

    assert str(converted.units) == "millimolar * mole[X] / millimole"
    assert converted == quantity
