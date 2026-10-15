# Examples

These runnable examples begin with Pint's familiar quantity workflow and then
show the semantic distinctions that xpint adds. They follow the progression in
the [Pint tutorial](https://pint.readthedocs.io/en/stable/getting/tutorial.html)
and [Pint NumPy guide](https://pint.readthedocs.io/en/stable/user/numpy.html),
while using explicitly imported symbols to keep scientific expressions compact.

| Example | Demonstrates |
|---|---|
| [`pint_style_basics.py`](pint_style_basics.py) | Quantity construction, arithmetic, and conversion with minimally decorated symbols |
| [`designated_solutions.py`](designated_solutions.py) | `mM["O2"]`, equivalent decomposed units, analyte mismatch rejection, and physically dimensionless compartment ratios |
| [`affine_temperature.py`](affine_temperature.py) | Absolute, Delta, and Generic temperature roles and their conversion/arithmetic rules |
| [`semantic_arrays.py`](semantic_arrays.py) | Kind/designation propagation through supported NumPy operations and fail-closed metadata boundaries |
| [`angle_dimensions.py`](angle_dimensions.py) | `angle_dimension=True` against a default registry: gyromagnetic-ratio input normalization, torque vs energy, solid vs plane angle, photometry, rotational identities, and explicit scalar conversion |
| [`custom_semantics.py`](custom_semantics.py) | Registering your own kinds and designation slots, browsing the catalogue, and the mirrored pairs that add and remove metadata |

Run any example from the repository root:

```bash
uv run python examples/pint_style_basics.py
uv run python examples/designated_solutions.py
uv run python examples/affine_temperature.py
uv run python examples/semantic_arrays.py
uv run python examples/angle_dimensions.py
uv run python examples/custom_semantics.py
```

The examples are exercised by the test suite and strict type checking so their
output and public API use do not drift from the implementation.
