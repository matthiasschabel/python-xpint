"""Register your own kinds and designation slots, and move values in and out of them.

The other examples use the built-in catalogue. This one covers the extension points a
project reaches for once its own vocabulary matters, and the paired methods that add and
remove xpint's metadata: ``as_kind`` has ``as_unkinded`` and ``with_designation`` has
``as_undesignated``. Removal is explicit rather
than reversible: ``as_undesignated`` discards every designation at once, and
``with_designation`` adds only one, so a multi-factor signature cannot be rebuilt from it.
"""

from fractions import Fraction

from xpint import AffineRole, DesignationTerm, KindDefinition, SemanticRegistry


def _show(description: str, value: object) -> None:
    print(f"    {description:<40} {value}")


def browse_the_builtin_catalogue(registry: SemanticRegistry) -> None:
    """Kinds are data, so the catalogue can be inspected rather than guessed at."""

    print("1. What is already registered")

    _show("registered kinds", len(registry.kinds))
    energy = registry.kinds["Energy"]
    _show("kinds['Energy']", energy)
    _show("it is a KindDefinition", isinstance(energy, KindDefinition))
    _show("its reference unit", energy.reference_unit)

    # A kind only earns its place when something else shares its dimension.
    joule_like = registry.compatible_kinds("joule")
    _show("kinds sharing joule's dimension", joule_like)


def register_a_domain_kind(registry: SemanticRegistry) -> None:
    """define_kind constrains a name to one dimensionality, and rejects anything else."""

    print("\n2. Registering a kind of your own")

    definition = registry.define_kind("SpecificActivity", "becquerel / gram")
    _show("define_kind returns", definition)

    sample = registry.Quantity(4.2, "becquerel / gram", kind="SpecificActivity")
    _show("a classified value", f"{sample} ({sample.kind})")

    # The constraint is dimensional, so the wrong dimension is refused at classification.
    try:
        registry.Quantity(1.0, "becquerel").as_kind("SpecificActivity")
    except ValueError as error:
        _show("wrong dimension rejected", str(error)[:44] + "...")

    # And two kinds over one dimension are exactly what stops a silent mistake.
    registry.define_kind("DoseRate", "gray / second")
    registry.define_kind("PowerPerMass", "watt / kilogram")
    try:
        registry.Quantity(1.0, "gray / second", kind="DoseRate") + registry.Quantity(
            1.0, "watt / kilogram", kind="PowerPerMass"
        )
    except TypeError as error:
        _show("same dimension, different kind", str(error)[:44] + "...")


def register_a_designation_slot(registry: SemanticRegistry) -> None:
    """A slot says which single factor of a named derived unit a bracket designates.

    A slot is only needed for a derived token. A single-factor unit such as ``gram`` is
    designatable already, and an inline compound such as ``gram / liter`` stays rejected
    however many slots are declared, because the bracket has no unambiguous factor to
    attach to. Designate the eligible factor before composing instead.
    """

    print("\n3. Declaring a designation slot")

    _show("gram[glucose] needs no slot", registry.Unit("gram")["glucose"])

    registry.define("mass_concentration = gram / liter")
    try:
        registry.Unit("mass_concentration")["glucose"]
    except ValueError as error:
        _show("derived token before its slot", str(error)[:44] + "...")

    registry.define_designation_slot("mass_concentration", "[mass]", exponent=Fraction(1))
    labelled = registry.Unit("mass_concentration")["glucose"]
    _show("derived token after its slot", labelled)
    _show("designation signature", labelled.designation_signature)
    _show("signature holds terms", isinstance(labelled.designation_signature[0], DesignationTerm))

    # An inline compound is still refused; a slot cannot make the target unambiguous.
    try:
        (registry.Unit("gram") / registry.Unit("liter"))["glucose"]
    except ValueError as error:
        _show("inline compound still refused", str(error)[:44] + "...")


def add_and_remove_metadata(registry: SemanticRegistry) -> None:
    """Metadata is added deliberately and removed deliberately, never silently."""

    print("\n4. Adding and removing metadata")

    energy = registry.Quantity(10.0, "joule").as_kind("Energy")
    _show("as_kind then has_kind()", energy.has_kind())
    _show("as_unkinded() removes it", energy.as_unkinded().has_kind())

    plain = registry.Quantity(2.0, "gram")
    labelled = plain.with_designation("glucose")
    _show("with_designation", f"{labelled} (has_designation={labelled.has_designation})")
    _show("as_undesignated", labelled.as_undesignated())

    # Removal is what lets a designated ratio become an ordinary number again.
    ratio = (0.6 * registry.Unit("milliliter")["plasma"]) / (
        1.0 * registry.Unit("milliliter")["whole_blood"]
    )
    _show("a designated ratio", f"is_semantic_unity={ratio.is_semantic_unity}")
    _show("after as_undesignated", float(ratio.as_undesignated()))

    # Affine role has no direct setter or remover. It is derived from the unit token and,
    # for a linear temperature unit, from the kind: see examples/affine_temperature.py.
    room = registry.Quantity(20.0, "degC")
    _show(
        "degC role, from the unit",
        f"{room.affine_role} ({room.affine_role is AffineRole.ABSOLUTE})",
    )
    kelvin = registry.Quantity(293.15, "kelvin")
    _show("bare kelvin role", kelvin.affine_role)
    _show("after as_kind('Temperature')", kelvin.as_kind("Temperature").affine_role)


def main() -> None:
    # A scoped registry, so registering kinds here cannot disturb xpint.units.
    registry = SemanticRegistry()

    browse_the_builtin_catalogue(registry)
    register_a_domain_kind(registry)
    register_a_designation_slot(registry)
    add_and_remove_metadata(registry)

    print("\nKinds and slots are per registry; the default xpint.units registry is untouched.")


if __name__ == "__main__":
    main()
