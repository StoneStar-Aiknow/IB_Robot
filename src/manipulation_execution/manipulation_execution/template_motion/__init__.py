"""Template motions blended into the mirror stream (X2 #3); off unless ``motion_template.enabled``.

``registry`` reads the segment whitelist the data plane also reads, ``labels``
keeps the matcher's per-template state, ``blender`` is the state machine (no
ROS) and ``hooks`` wraps the session hooks around it.
"""
