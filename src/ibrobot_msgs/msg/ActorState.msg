# Which person the resident motion-imitation stream treats as the actor.
#
# One message per frame the stream's PEAR stage takes up, published before the
# PEAR call: also while no YOLOX result exists yet and when PEAR then fails, so
# it doubles as a heartbeat. The current selector keeps no identity: the actor
# is the highest-confidence qualifying person, actor_id is 0 whenever there is
# one, and start_ready repeats actor_present. The fields are the contract
# every selector keeps.

# header.stamp is the observation time of the frame this decision is for.
std_msgs/Header header

# Observation time of the YOLOX detections the decision rests on; zero before
# the first detection result. YOLOX refreshes less often than frames arrive,
# so this normally trails header.stamp.
builtin_interfaces/Time detections_stamp

# Qualifying persons (label "person", confidence at or above the threshold) in
# those detections. 0 when the decision is unknown.
uint32 persons_count

bool actor_present

# Whether a session may start mirroring this actor. The current selector sets
# it equal to actor_present.
bool start_ready

# Identity of the actor across frames; -1 when there is none.
int32 actor_id

# Actor box in source-image pixels, [x_min, y_min, x_max, y_max]; zeros when
# there is no actor.
float32[4] bbox

# Detection confidence of the actor; 0.0 when there is none.
float32 confidence

# "highest_confidence_person" or "no_person" when the answer is known.
# "no_detections" (no YOLOX result yet) or "stale_detections" (the newest
# result is older than the stream's actor_stale_sec) when it is unknown.
# Unknown always has actor_present false: it must never count as present.
string reason
