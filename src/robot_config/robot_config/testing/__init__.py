"""Installed test-support API of robot_config.

The navigation staging helper (and, later, other cross-package test
support) lives here so tests OUTSIDE robot_config — the embodied bringup
E2E in particular — can use it without importing the source-path ``test``
package, which setup.py deliberately does not install
(lekiwi-nav-grasp-mobile-manipulation R8).
"""
