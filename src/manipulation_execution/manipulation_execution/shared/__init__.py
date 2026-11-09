"""ROS-free code shared by more than one subsystem (design D5).

``shared`` imports nothing from the package: its modules are the dependency
floor both ``pick`` and ``hri`` may build on. ``runtime_ownership`` holds the
runtime-ownership rules a stream that takes over a Gateway-switched mode
follows; it must stay free of ``rclpy`` and every ROS message package.
"""
