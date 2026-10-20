"""Constants shared by the X2 visual interaction session and whoever wires or speaks for it.

The session node (visual_interaction), its launch builder (the same package),
the robot config that enables it, and the speech coordinator that will later own
delivery each need the same names, closed value sets and timing. They live only
here. Keep this module dependency-free: importing it must not pull in any launch
builder or ROS package.

Design: docs/x2_visual_interaction_design_zh.md
"""

# The session node's name, the utterance it publishes and its diagnostics.
NODE_NAME = "visual_interaction"
UTTERANCE_TOPIC = f"/{NODE_NAME}/utterance"
STATUS_TOPIC = f"/{NODE_NAME}/status"

# How the robot is asked to stop talking: ``std_srvs/SetBool``, data false to
# mute and true to resume. A visitor cannot reach a ROS service, so whatever
# calls this — an attendant console, a button — is part of the feature rather
# than an optional extra. Without it there is no way to ask for quiet, and only
# an explicit resume ever clears a mute.
SET_ENABLED_SERVICE = "set_enabled"
SET_ENABLED_FULL_NAME = f"/{NODE_NAME}/{SET_ENABLED_SERVICE}"

# What the model may report about one observation window.
#
# ``none`` is "nothing new worth answering" and ``uncertain`` is "cannot see it
# well enough to be sure". They are counted apart so a quiet run can be told
# from a blind one. ``action`` is absent on purpose: gestures need a real
# keypoint model and a measured end-to-end latency before the feature may claim
# them, so the first version has no way to report one.
KIND_NONE = "none"
KIND_UNCERTAIN = "uncertain"
KIND_OBJECT = "object"
KIND_DISPLAY = "display"
KINDS = (KIND_NONE, KIND_UNCERTAIN, KIND_OBJECT, KIND_DISPLAY)

# Kinds that carry a candidate to speak. The others never produce an utterance.
SPEAKABLE_KINDS = (KIND_OBJECT, KIND_DISPLAY)

# The only subjects the first version may talk about.
#
# A closed set with no fallback member: a generic "other object" category would
# hand the "is this sensitive?" judgement back to the model, which is what the
# whitelist exists to prevent. Pendants are a real request but stay out until
# small-object visibility and religious-symbol refusal are accepted separately;
# medicine bottles, blades and books are out of scope entirely.
CATEGORIES = ("toy", "sign", "hat", "badge")

# Where on the person the subject is. ``unknown`` is a valid answer, not a
# failure, but it never matches a more specific region for de-duplication.
REGIONS = ("hand", "head", "chest", "upper_body", "unknown")

# How a topic came to be settled. Coverage outranks event numbering: an already
# covered topic stays covered when the same object is put down and raised again,
# so a fresh episode_seq can never talk over it.
COVERED_ESTIMATED = "covered_estimated"
PARTIAL_COVERED = "partial_covered"
SUPPRESSED = "suppressed"
IN_FLIGHT = "in_flight"
QUARANTINED = "quarantined"
COVERAGE_STATES = (
    COVERED_ESTIMATED,
    PARTIAL_COVERED,
    SUPPRESSED,
    IN_FLIGHT,
    QUARANTINED,
)
# States that forbid speaking about the topic again in this session.
BLOCKING_COVERAGE_STATES = COVERAGE_STATES

# Why a round produced no utterance. Reported on the diagnostics topic so a long
# silence can be read: NO_NEW_TOPIC is the feature working, MODEL_NOT_READY is not.
SUPPRESS_NO_NEW_TOPIC = "NO_NEW_TOPIC"
SUPPRESS_UNCERTAIN = "UNCERTAIN"
SUPPRESS_DUPLICATE = "DUPLICATE"
SUPPRESS_UNCONFIRMED = "UNCONFIRMED"
SUPPRESS_PRIVACY = "PRIVACY"
SUPPRESS_STALE = "STALE"
SUPPRESS_AMBIGUOUS = "AMBIGUOUS"
SUPPRESS_COOLDOWN = "COOLDOWN"
SUPPRESS_IN_FLIGHT = "IN_FLIGHT"
SUPPRESS_MUTED = "MUTED"
SUPPRESS_MODEL_NOT_READY = "MODEL_NOT_READY"
SUPPRESS_SCHEMA_INVALID = "SCHEMA_INVALID"
SUPPRESS_REASONS = (
    SUPPRESS_NO_NEW_TOPIC,
    SUPPRESS_UNCERTAIN,
    SUPPRESS_DUPLICATE,
    SUPPRESS_UNCONFIRMED,
    SUPPRESS_PRIVACY,
    SUPPRESS_STALE,
    SUPPRESS_AMBIGUOUS,
    SUPPRESS_COOLDOWN,
    SUPPRESS_IN_FLIGHT,
    SUPPRESS_MUTED,
    SUPPRESS_MODEL_NOT_READY,
    SUPPRESS_SCHEMA_INVALID,
)

# Timing defaults. Every one of these is a proposal until phase zero measures
# the real VL and TTS latency on the board; none of them is a measured value.
#
# A static topic must be seen in two different sampling windows before it may be
# spoken, so the confirmation itself costs at least one more round.
DEFAULT_CONFIRMATION_COUNT = 2
# How old the evidence behind a candidate may be when it is finally submitted.
DEFAULT_MAX_EVIDENCE_AGE_SEC = 8.0
# Quiet time measured from the estimated end of the previous utterance.
DEFAULT_MIN_GAP_SEC = 10.0
# Explicit hand-over is the normal way to start a new visitor; the absent timer
# is the missed-hand-over guard and a camera dropout is unknown, not absent.
DEFAULT_ABSENT_RESET_SEC = 30.0
DEFAULT_MAX_SESSION_SEC = 120.0
# Reply length band, in characters.
DEFAULT_MIN_REPLY_CHARS = 8
DEFAULT_MAX_REPLY_CHARS = 30

# Sentinel for "no actor". Matches manipulation_execution.actor_selector.NO_ACTOR;
# repeated here to keep this module dependency-free.
NO_ACTOR = -1
