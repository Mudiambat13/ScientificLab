import scientificlab


def test_version_is_exposed():
    assert scientificlab.__version__ == "0.1.0"


def test_readme_examples_run():
    from scientificlab.mathematics.algebra import solve_quadratic
    from scientificlab.physics.mechanics import kinetic_energy

    assert solve_quadratic(a=1, b=-5, c=6).roots == (2.0, 3.0)
    assert kinetic_energy(mass=10, velocity=5) == 125.0
