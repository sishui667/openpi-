"""Automatic scene setup is isolated from camera recognition inputs."""
import ast
from pathlib import Path
import random
import unittest

from tabletop_fixture import FIXTURE_COLOURS,FIXTURE_SIZES,sample_fixture


class AutomaticFixtureTests(unittest.TestCase):
    def test_random_test_solids_need_no_user_shape_and_cover_supported_classes(self):
        rng=random.Random(17)
        fixtures=[sample_fixture(rng) for _ in range(100)]
        self.assertEqual({f.shape for f in fixtures},set(FIXTURE_SIZES))
        self.assertEqual({f.colour for f in fixtures},set(FIXTURE_COLOURS))
        for fixture in fixtures:
            self.assertEqual(fixture.size_xyz,FIXTURE_SIZES[fixture.shape])

    def test_reproducible_fixture_randomness_does_not_require_a_vision_seed(self):
        a=random.Random(91);b=random.Random(91)
        self.assertEqual([sample_fixture(a) for _ in range(10)],
                         [sample_fixture(b) for _ in range(10)])

    def test_detector_and_ik_do_not_import_fixture_sampler(self):
        root=Path(__file__).resolve().parent
        for name in ('tabletop_perception.py','generic_grasp_planner.py'):
            with self.subTest(module=name):
                tree=ast.parse((root/name).read_text(encoding='utf-8'))
                modules=[]
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import): modules.extend(alias.name for alias in node.names)
                    elif isinstance(node,ast.ImportFrom): modules.append(node.module)
                self.assertNotIn('tabletop_fixture',modules)


if __name__=='__main__': unittest.main()
