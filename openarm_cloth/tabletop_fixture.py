"""Simulation fixture sampling only; never imported by visual recognition/IK.

Sampling creates a test scene, not a shape hint or a learned object model.
The controller must infer the resulting geometry from its camera images.
"""
from dataclasses import dataclass
import random


FIXTURE_SIZES = {
    'cube': (.0576, .0576, .0576),
    'box': (.070, .040, .055),
    'cylinder': (.045, .045, .075),
}
FIXTURE_COLOURS = ('blue', 'green', 'black', 'white', 'red', 'yellow', 'brown')


@dataclass(frozen=True)
class ObjectFixture:
    shape: str
    size_xyz: tuple[float, float, float]
    colour: str


def sample_fixture(rng=None):
    """Choose a supported test solid without requiring a user shape choice.

    Fixed, previously exercised sizes keep this a controlled simulation test,
    not an assertion that arbitrary dimensions or placements are graspable.
    An injected RNG is for reproducible tests; it is not a perception input.
    """
    rng = random.SystemRandom() if rng is None else rng
    shape = rng.choice(tuple(FIXTURE_SIZES))
    return ObjectFixture(shape, FIXTURE_SIZES[shape], rng.choice(FIXTURE_COLOURS))
