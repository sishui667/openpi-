"""Test launcher defaults without creating windows or starting processes."""
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
import unittest


launcher=ModuleType('tabletop_launcher_for_test')
launcher.__file__=str(Path(__file__).with_name('launch_tabletop_grasp.pyw'))
SourceFileLoader(launcher.__name__,launcher.__file__).exec_module(launcher)


class AutomaticLauncherTests(unittest.TestCase):
    def test_default_launch_requires_no_type_colour_size_or_position_choice(self):
        command=launcher.build_launch_command('D:/runtime/pythonw.exe','D:/test/report.json')
        self.assertEqual(Path(command[0]).name,'python.exe')
        self.assertIn('--random-fixture',command)
        for forbidden in ('--shape','--color','--size','--fixture-pose'):
            self.assertNotIn(forbidden,command)

    def test_explicit_fixture_is_optional_simulation_setup(self):
        fixture=('cylinder','blue',(.045,.045,.075),(.43,-.10,15))
        command=launcher.build_launch_command('D:/runtime/python.exe','D:/test/report.json',fixture=fixture)
        self.assertNotIn('--random-fixture',command)
        self.assertEqual(command[command.index('--shape')+1],'cylinder')
        self.assertEqual(command[command.index('--size')+1:command.index('--size')+4],['0.045','0.045','0.075'])


if __name__=='__main__': unittest.main()
