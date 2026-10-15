"""Dask, xarray, and pint-xarray interoperability boundaries."""

from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xpint import SemanticQuantity, SemanticRegistry
from xpint.units import m, ureg

pytest.importorskip("pint_xarray")
dask_array = cast(Any, da)


def test_dask_materialization_and_rechunk_preserve_semantics() -> None:
    magnitude = dask_array.from_array(np.arange(6.0).reshape(2, 3), chunks=(1, 3))
    values = ureg.Quantity(magnitude, m["tissue"], kind="Length")

    computed = values.compute()
    persisted = values.persist()
    rechunked = values.rechunk((2, 1))

    for result in (computed, persisted, rechunked):
        assert isinstance(result, SemanticQuantity)
        assert result.kind == "Length"
        assert result.designation_signature == values.designation_signature
    np.testing.assert_array_equal(
        cast(Any, computed.magnitude),
        np.arange(6.0).reshape(2, 3),
    )
    assert rechunked.magnitude.chunks == ((2,), (1, 1, 1))


def test_classified_dask_map_blocks_fails_closed() -> None:
    magnitude = dask_array.from_array(np.arange(3.0), chunks=2)
    values = ureg.Quantity(magnitude, m, kind="Length")

    with pytest.raises(TypeError, match="change values or dimensions"):
        values.map_blocks(lambda block: block + 1)


def test_plain_dask_map_blocks_matches_pint_delegation() -> None:
    magnitude = dask_array.from_array(np.arange(3.0), chunks=2)
    values = ureg.Quantity(magnitude, m)

    result = values.map_blocks(lambda block: block + 1)

    assert isinstance(result, da.Array)
    np.testing.assert_array_equal(dask_array.compute(result)[0], np.arange(3.0) + 1)


def test_plain_dask_rechunk_matches_pint_delegation() -> None:
    magnitude = dask_array.from_array(np.arange(4.0), chunks=2)
    values = ureg.Quantity(magnitude, m)

    result = values.rechunk(1)

    assert isinstance(result, da.Array)
    assert result.chunks == ((1, 1, 1, 1),)


def test_xarray_reductions_preserve_semantics_with_skipna() -> None:
    data = xr.DataArray(
        ureg.Quantity(np.array([1.0, np.nan, 3.0]), m["tissue"], kind="Length"),
        dims="sample",
    )

    total = data.sum()
    average = data.mean()

    assert total.data.magnitude == pytest.approx(4.0, rel=0.0, abs=0.0)
    assert average.data.magnitude == pytest.approx(2.0, rel=0.0, abs=0.0)
    for result in (total.data, average.data):
        assert result.kind == "Length"
        assert result.designation_signature == data.data.designation_signature


def test_xarray_concat_and_where_require_matching_semantics() -> None:
    left = xr.DataArray(
        ureg.Quantity(np.array([1.0, 2.0]), m["tissue"], kind="Length"),
        dims="sample",
    )
    right = xr.DataArray(
        ureg.Quantity(np.array([3.0, 4.0]), m["tissue"], kind="Length"),
        dims="sample",
    )

    concatenated = xr.concat([left, right], dim="sample")
    threshold = (1 * m["tissue"]).as_kind("Length")
    selected = left.where(left > threshold, right)

    np.testing.assert_array_equal(
        cast(Any, concatenated.data.magnitude),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    np.testing.assert_array_equal(cast(Any, selected.data.magnitude), np.array([3.0, 2.0]))
    assert concatenated.data.kind == selected.data.kind == "Length"
    assert concatenated.data.designation_signature == left.data.designation_signature
    assert selected.data.designation_signature == left.data.designation_signature

    wrong_kind = right.copy(data=right.data.as_unkinded())
    with pytest.raises(TypeError, match="different kinds"):
        xr.concat([left, wrong_kind], dim="sample")


def test_xarray_dask_compute_preserves_semantics() -> None:
    magnitude = dask_array.from_array(np.arange(4.0), chunks=2)
    data = xr.DataArray(ureg.Quantity(magnitude, m["tissue"], kind="Length"), dims="sample")

    computed = data.compute()

    assert computed.data.kind == "Length"
    assert computed.data.designation_signature == data.data.designation_signature


def test_xarray_plain_unit_multiplication_uses_pint_path() -> None:
    data = xr.DataArray(np.array([1.0, 2.0]), dims="sample")

    quantified = data * m

    assert isinstance(quantified.data, SemanticQuantity)
    assert quantified.data.kind is None
    assert quantified.data.units == m


def test_pint_xarray_quantifies_plain_units_with_configured_registry() -> None:
    registry = SemanticRegistry(force_ndarray_like=True)
    data = xr.DataArray([1.0, 2.0], dims="sample", attrs={"units": "meter"})

    quantified = data.pint.quantify(unit_registry=registry)

    assert isinstance(quantified.data, registry.Quantity)
    assert quantified.data.kind is None
    assert quantified.data.units == registry.meter


def test_classified_dask_metadata_is_readable() -> None:
    magnitude = dask_array.from_array(np.arange(6).reshape(2, 3), chunks=(1, 3))
    quantity = ureg.Quantity(magnitude, m["tissue"])
    for name in ("chunks", "chunksize", "numblocks", "name"):
        assert getattr(quantity, name) == getattr(magnitude, name)


def test_offset_temperature_equality_stays_lazy_for_dask() -> None:
    celsius = ureg.Quantity(dask_array.zeros((2, 3), chunks=(1, 3)), "degC")
    fahrenheit = ureg.Quantity(dask_array.zeros((2, 3), chunks=(1, 3)), "degF")
    for result in (celsius == fahrenheit, fahrenheit != celsius):
        assert isinstance(result, da.Array)
        assert result.chunks == ((1, 1), (3,))
    np.testing.assert_array_equal(
        dask_array.compute(celsius == fahrenheit)[0], np.zeros((2, 3), dtype=np.bool_)
    )


def test_semantic_mismatch_equality_preserves_dask_dispatch() -> None:
    scalar = ureg.Quantity(1, "joule", kind="Energy")
    array = ureg.Quantity(dask_array.ones((2, 3), chunks=(1, 3)), "joule", kind="Torque")
    for result in (scalar == array, array == scalar):
        assert isinstance(result, da.Array)
        assert result.chunks == ((1, 1), (3,))
        np.testing.assert_array_equal(
            dask_array.compute(result)[0], np.zeros((2, 3), dtype=np.bool_)
        )
