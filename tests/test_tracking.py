import unittest

import numpy as np

from tracking.app import VideoClock, config_for_frame, parser
from tracking.core import Config, MotionFilter


class MotionTests(unittest.TestCase):
    def test_4k_resolution_scales_default_noise(self):
        args = parser().parse_args(["example.mp4", "--weights", "model.pt"])
        config = config_for_frame(args, (2160, 3840, 3))
        self.assertEqual(config.initial_velocity_std, 600)

    def test_explicit_noise_parameters_override_resolution_defaults(self):
        args = parser().parse_args(["example.mp4", "--weights", "model.pt",
                                    "--measurement-std", "3", "--acceleration-std", "700",
                                    "--initial-velocity-std", "150"])
        config = config_for_frame(args, (2160, 3840, 3))
        self.assertEqual((config.measurement_std, config.acceleration_std,
                          config.initial_velocity_std), (3, 700, 150))

    def test_anisotropic_gate(self):
        kf = MotionFilter((100, 200), Config(measurement_std=1))
        kf.P = np.diag([99., 24., 1., 1.])
        self.assertAlmostEqual(kf.distance_squared((120, 205)), 5)
        self.assertAlmostEqual(kf.distance_squared((105, 220)), 16.25)

    def test_constant_velocity_with_irregular_timestamps(self):
        kf = MotionFilter((0, 0), Config())
        time = 0
        for dt in [.04, .08, .12, .04] * 10:
            time += dt
            kf.predict(dt)
            kf.correct((50 * time, 20 * time))
        np.testing.assert_allclose(kf.x, [50 * time, 20 * time, 50, 20], atol=.1)
        self.assertGreaterEqual(np.linalg.eigvalsh(kf.P).min(), -1e-9)

    def test_timestamp_fallback_is_monotonic(self):
        clock = VideoClock(25)
        stamps = [clock.get(ms, i) for i, ms in enumerate([0, 40, 40, 0, 160])]
        np.testing.assert_allclose([s[0] for s in stamps], [0, .04, .08, .12, .16])
        self.assertEqual(stamps[2][1], "fps_fallback")


if __name__ == "__main__":
    unittest.main()
