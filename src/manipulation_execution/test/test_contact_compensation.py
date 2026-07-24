from manipulation_execution.contact_compensation import ContactPrediction, compensate_contact_xy


def test_compensation_corrects_both_axes():
    def predict(command, _previous):
        return ContactPrediction(
            contact_base=(command[0] - 0.01, command[1] + 0.02, command[2]),
            payload=command,
        )

    result = compensate_contact_xy(
        (0.2, -0.1, 0.1),
        (0.2, -0.1, 0.1),
        predict,
        tolerance_m=0.001,
        max_iterations=2,
        max_correction_m=0.03,
    )
    assert result.converged
    assert result.command_xyz == (0.21000000000000002, -0.12000000000000001, 0.1)
    assert abs(result.residual_x) <= 0.001
    assert abs(result.residual_y) <= 0.001
