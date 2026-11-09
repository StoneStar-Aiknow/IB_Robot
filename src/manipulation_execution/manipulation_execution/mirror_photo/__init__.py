"""X2 mirror photo: one photo per MIRRORING round, sent to the upper PC.

Modules:

``config``     the ``photo`` section of ``mirror_session``, its defaults and
               checks; the upper PC address.
``stillness``  the body speed from the PEAR rotations; a pose held still.
``triggers``   the steady rhythm, the gestures and the external events as
               candidate requests.
``scoring``    the round's best candidate, picked after the round, within a
               deadline.
``selector``   ``PhotoCore``: one round, frames in, candidates kept, one photo
               out; the photo's metadata.
``outbox``     the photos not yet delivered, on disk, crash-safe and bounded.
``uploader``   the outbox to the upper PC over HTTP, retried until confirmed.
``worker``     the node's disk thread: a finished round chosen and queued, the
               running round's crash copy kept.

The external trigger message is in ``embodied_common.mirror_photo_contract``.
Nothing in this package makes a ROS call, so all of it can be exercised with
synthetic frames and a fake clock. ``mirror_photo_node`` adapts it to ROS.
"""
