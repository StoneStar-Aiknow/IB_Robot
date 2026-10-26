# A grasp plan (manipulation_execution grasp_pipeline.contracts.GraspPlan) the X2
# grasp head made.

GraspPlanStep[] steps
# What release_after_success runs; without one the forward steps are retraced.
bool has_release
GraspPlanStep[] release
int32 candidate_index
# The planner's report (JSON object).
string report_json
