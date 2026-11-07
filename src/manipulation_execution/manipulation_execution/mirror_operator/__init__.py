"""X2 mirror operator: cover gestures, head touch and the light, around the session.

Modules:

``config``   the ``operator`` section of the business config and its defaults.
``judge``    ``GrayFrame`` and the cover judges, chosen by name: is the camera
             covered in this frame (the business half).
``frames``   camera messages to ``GrayFrame``, and the frame rate limit.
``gesture``  double cover and long cover from the covered/clear sequence (the
             framework half).
``touch``    a head touch held long enough.
``led``      when the light strip follows the session.
``gateway``  the ``mirror_start`` and ``mirror_stop`` requests to the Gateway:
             status, validate, goal.
``vendor``   the only place that names the vendor: its touch and light types,
             loaded by name.
``core``     ``Operator``: the above put together, sending through injected
             calls, at most one call in flight per channel.

Nothing in this package makes a ROS call, so all of it can be exercised with
fake calls. ``mirror_operator_node`` adapts it to ROS.
"""
