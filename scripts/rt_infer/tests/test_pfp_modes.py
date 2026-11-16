"""PFP modes preserve strategy boundaries and resolve internal optimizations."""

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[3] / "third_party/vendor/rt-infer/2026.11.05/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("mode_control", SCRIPTS / "priorityctl.py")
ctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctl)


class ModeTests(unittest.TestCase):
    def args(self, **values):
        result = dict(
            scheduler="pfp",
            pfp_mode=None,
            pfp_periodic=None,
            prefetch=None,
            pfp_exclusive=None,
            pfp_batch_select=None,
            local_pool=None,
            pfp_min_cache=None,
        )
        return argparse.Namespace(**(result | values))

    def test_general_enables_internal_optimizations_without_periodic_admission(self):
        args = self.args()
        ctl.configure_pfp(args)
        self.assertEqual(args.pfp_mode, "general")
        self.assertIsNone(args.pfp_periodic)
        for key in ("prefetch", "pfp_exclusive", "pfp_batch_select", "local_pool", "pfp_min_cache"):
            self.assertEqual(getattr(args, key), "on")

    def test_periodic_requires_release_mapping(self):
        with self.assertRaises(ctl.base.DeploymentError):
            ctl.configure_pfp(self.args(pfp_mode="periodic"))
        args = self.args(pfp_mode="periodic", pfp_periodic=Path("release.bin"))
        ctl.configure_pfp(args)
        self.assertEqual(args.pfp_mode, "periodic")

    def test_incompatible_modes_fail_closed(self):
        for args in (
            self.args(scheduler="fifo", pfp_mode="periodic"),
            self.args(pfp_mode="general", pfp_periodic=Path("release.bin")),
            self.args(pfp_periodic=Path("release.bin"), pfp_exclusive="off"),
        ):
            with self.assertRaises(ctl.base.DeploymentError):
                ctl.configure_pfp(args)

    def test_fifo_nice_defaults_do_not_change(self):
        for strategy in ("fifo", "nice"):
            args = self.args(scheduler=strategy)
            ctl.configure_pfp(args)
            self.assertIsNone(args.pfp_mode)
            for key in ("prefetch", "pfp_exclusive", "pfp_batch_select", "local_pool", "pfp_min_cache"):
                self.assertEqual(getattr(args, key), "off")

    def test_legacy_explicit_flags_remain_available_for_comparison(self):
        args = self.args(prefetch="off", local_pool="off", pfp_periodic=Path("release.bin"))
        ctl.configure_pfp(args)
        self.assertEqual(args.pfp_mode, "periodic")
        self.assertEqual(args.prefetch, "off")
        self.assertEqual(args.local_pool, "off")


if __name__ == "__main__":
    unittest.main()
