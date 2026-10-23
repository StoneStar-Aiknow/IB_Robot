# One thing the X2 visual interaction session decided to say, or decided not to.
#
# The session publishes exactly one message per observation round, including the
# rounds it stays quiet for: a reader can tell "nothing new to answer" from
# "the model is not ready" without watching the robot. Only a message whose
# kind is in SPEAKABLE_KINDS and whose suppress_reason is empty is a candidate
# to speak.
#
# Unlike VisualGameEvent, the words are IN this message. A visual game picks its
# announcement from a frozen table keyed by handler, so its event carries no
# text; here the reply is generated per round and the text is the payload. A
# consumer must never re-derive it from the enum fields.
#
# TODO(speech-coordinator): the first version has no global speech arbitration,
# so this topic is the whole delivery path: the session publishes, and the
# diagnostics and logs record the text. Three speech sources already call
# /voice_tts/synthesize and /voice_tts/play unaware of each other, and
# PlayAudioFile has no request-level cancel, so a coordinator owning the lease,
# the budget, the deadline and cancellation is a prerequisite for speaking out
# loud, not an enhancement. It consumes this message unchanged.

# Observation time of the newest frame the decision rests on. This is the
# evidence clock, not the time the decision was made: how old the picture was
# decides whether the reply may still be spoken.
std_msgs/Header header

# De-duplication scope. A session is an interaction range, never an identity:
# it ends on an explicit hand-over to the next visitor, on a confirmed absence,
# or on the maximum duration, and a covered topic stays covered within it.
string session_id

# Bumped whenever the short-term actor association breaks or the target
# changes, which invalidates every window and in-flight result from before it.
# It is not a visitor identity either: nothing here survives an occlusion.
uint32 actor_epoch

# Identifies this decision for cancellation and coverage bookkeeping. Assigned
# by the program, never by the model.
string event_id

# What the session would say, exactly as generated. Empty whenever the round
# produced no candidate. A reply only ever expresses the category and the
# display relation below: colour, material, brand, price, sign text, emotion
# and appearance are out of the first version's verifiable range, so a reply
# that implies one is rejected before it reaches this message.
string text

# "none", "uncertain", "object" or "display" (embodied_common
# .visual_interaction_contract.KINDS). "none" is nothing new worth answering
# and "uncertain" is cannot see it well enough; both are counted, and neither
# ever carries text.
string kind

# "toy", "sign", "hat" or "badge" (CATEGORIES). A closed whitelist with no
# fallback member on purpose: a generic category would return the "is this
# sensitive?" judgement to the model. Empty when kind carries no subject.
string category

# "hand", "head", "chest", "upper_body" or "unknown" (REGIONS). Empty when
# kind carries no subject. A body region is not an identity hint.
string region

# Counts display episodes of the same topic, for tracing and cancellation only.
# Coverage outranks it: a covered topic stays silent when the same object is
# put down and raised again, however high this climbs.
uint64 episode_seq

# Observation time of the earliest frame in the window the decision rests on,
# so a reader can see how wide the evidence was. header.stamp is the newest.
builtin_interfaces/Time evidence_start_stamp

# The latest moment speaking may START, derived from the evidence age and the
# measured latency budget. It is not a deadline for finishing playback. Zero
# when there is nothing to say.
builtin_interfaces/Time latest_start_at

# Why nothing was said, from SUPPRESS_REASONS; empty when text is set. This
# separates a working silence (NO_NEW_TOPIC) from a broken one
# (MODEL_NOT_READY), which is the difference a bystander cannot hear.
string suppress_reason
