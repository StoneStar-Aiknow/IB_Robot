# One human hand in a HandFrame (the person's own left or right, not the robot's).
#
# Same fields as video2x2_runtime hri_pear_msgs/HandObservation (570c53c), so
# its hand_core consumes them unchanged.

bool detected
# Handedness score of the landmark model; it gives no separate landmark confidence.
float32 score

# Sum of bone-segment angles per finger (thumb, index, middle, ring, pinky), radians.
float32[5] curl_rad
# curl_rad normalised by the calibration end points: 0 open, 1 fist, not clamped.
# The thumb entry is informative only.
float32[5] curl_norm

# 21 x 3 metres, origin at the palm (the landmark model's world output).
float32[63] world_landmarks
# 21 x 2 source-image pixels, x y.
float32[42] image_landmarks

# Gesture label for hand_core: an official MediaPipe label (Victory, Thumb_Up,
# ...) or a keypoint rule label (six, thumb_index_middle); "" when none.
string gesture
float32 gesture_score
