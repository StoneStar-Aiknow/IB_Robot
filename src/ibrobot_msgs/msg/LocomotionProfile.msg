# Vendor speed-profile identifier; values are platform-specific, not a universal enum.
int32 mode
# Signed bounds: [minimum, maximum]. Linear speeds in m/s, angular in rad/s.
float64[2] forward
float64[2] lateral
float64[2] angular
