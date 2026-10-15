"""Keep analytes and compartments distinct without changing physical dimensions."""

from xpint.units import L, mL, mM, mmol


def main() -> None:
    oxygen = 5 * mM["O2"]
    decomposed = 5 * mmol["O2"] / L
    sample_volume = 2 * mL
    oxygen_amount = (oxygen * sample_volume).to(mmol["O2"])

    print(f"concise concentration: {oxygen}")
    print(f"decomposed concentration: {decomposed}")
    print(f"representations agree: {oxygen == decomposed}")
    print(f"oxygen in sample: {oxygen_amount}")

    sodium_chloride = 5 * mM["NaCl"]
    try:
        oxygen + sodium_chloride
    except TypeError as error:
        print(f"analyte mismatch rejected: {error}")

    compartment_ratio = 0.6 * mL["plasma"] / mL["whole_blood"]
    print(f"compartment ratio: {compartment_ratio}")
    print(f"physically dimensionless: {compartment_ratio.dimensionless}")
    print(f"designation remains: {compartment_ratio.has_designation}")
    print(f"safe undesignated scalar: {compartment_ratio.is_semantic_unity}")

    try:
        float(compartment_ratio)
    except TypeError as error:
        print(f"implicit scalar coercion rejected: {error}")


if __name__ == "__main__":
    main()
