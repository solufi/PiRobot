import unittest

from server.task_policy import validate_steps


class TaskPolicyTests(unittest.TestCase):
    def test_normalizes_valid_task(self):
        steps, total = validate_steps([
            {"action": "drive", "direction": "forward", "duration_ms": 500},
            {"action": "wait", "duration_ms": 200},
        ])
        self.assertEqual(total, 700)
        self.assertEqual(steps[0]["direction"], "forward")

    def test_rejects_long_step(self):
        with self.assertRaisesRegex(ValueError, "durée invalide"):
            validate_steps([{"action": "drive", "direction": "forward", "duration_ms": 1501}])

    def test_rejects_too_many_steps(self):
        with self.assertRaisesRegex(ValueError, "1 à 8 étapes"):
            validate_steps([{"action": "wait", "duration_ms": 1500}] * 9)

    def test_rejects_unknown_direction(self):
        with self.assertRaisesRegex(ValueError, "direction invalide"):
            validate_steps([{"action": "drive", "direction": "spin"}])


if __name__ == "__main__":
    unittest.main()
