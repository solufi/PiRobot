"""Exercise the real controller with hardware and camera interfaces replaced."""
import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


class FakeApp:
    def __init__(self, *args):
        self.config = {}

    def route(self, *args, **kwargs):
        return lambda fn: fn


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flask = MagicMock()
        flask.Flask = FakeApp
        sock = MagicMock()
        sock.Sock.return_value.route.side_effect = lambda *a, **k: lambda fn: fn
        modules = {name: MagicMock() for name in (
            "picarx", "picamera2", "cv2", "numpy", "gpt_brain",
            "face_identity", "robot_hat", "robot_hat.utils",
        )}
        modules.update(flask=flask, flask_sock=sock)
        modules["task_policy"] = __import__("server.task_policy", fromlist=["validate_steps"])
        spec = importlib.util.spec_from_file_location(
            "controller_test", Path(__file__).parents[1] / "server/picar_server.py"
        )
        cls.controller = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.object(threading.Thread, "start"), \
                patch("os.getlogin", return_value="test-user"):
            spec.loader.exec_module(cls.controller)

    def setUp(self):
        self.c = self.controller
        self.c.px.reset_mock()
        self.c.settings.update(safety=True, cliff=False, obstacle=False,
                               follow_me=True, tracking=True, distance_valid=True)
        self.c.control.update(dict.fromkeys(self.c.control, False))
        self.c.last_frame_at = time.monotonic()
        self.c.drive_cancel.clear()
        self.c.last_motor_command = None
        self.c.last_sensor_at = time.monotonic()
        self.c.motor_action = "stop"

    def test_stop_prevents_inflight_tracker_from_restarting_motors(self):
        self.c.stop_all()
        self.c.px.reset_mock()
        self.assertFalse(self.c.guarded_drive("forward", 40, follow=True))
        self.c.px.forward.assert_not_called()
        self.assertTrue(self.c.drive_cancel.is_set())

    def test_follow_forward_blocked_by_obstacle(self):
        self.c.settings["obstacle"] = True
        self.assertFalse(self.c.guarded_drive("forward", 40, follow=True))
        self.c.px.forward.assert_not_called()
        self.c.px.stop.assert_called_once()

    def test_invalid_distance_blocks_autonomous_forward(self):
        self.c.settings["distance_valid"] = False
        self.assertFalse(self.c.guarded_drive("forward", 40, follow=True))
        self.c.px.forward.assert_not_called()

    def test_sensor_stall_stops_active_motors(self):
        self.c.last_sensor_at = time.monotonic() - 2
        self.c.motor_action = "backward"
        self.c.check_control_timeout(1.5)
        self.c.px.stop.assert_called_once()

    def test_invalid_ultrasonic_samples_expire_previous_clear_distance(self):
        self.c.px.get_distance.side_effect = [100, -1, -1, -1]
        self.c.px.get_grayscale_data.return_value = [1000, 1000, 1000]
        try:
            with patch.object(self.c.time, "sleep", side_effect=[None, None, None, RuntimeError("end loop")]):
                with self.assertRaisesRegex(RuntimeError, "end loop"):
                    self.c.sensor_loop()
            self.assertEqual(self.c.settings["distance_cm"], -1)
            self.assertFalse(self.c.settings["distance_valid"])
        finally:
            self.c.px.get_distance.side_effect = None

    def test_missing_grayscale_fails_closed(self):
        self.c.px.get_distance.return_value = 100
        self.c.px.get_grayscale_data.return_value = []
        with patch.object(self.c.time, "sleep", side_effect=RuntimeError("end loop")):
            with self.assertRaisesRegex(RuntimeError, "end loop"):
                self.c.sensor_loop()
        self.assertTrue(self.c.settings["cliff"])
        self.assertEqual(self.c.settings["sensor_error"], "Capteurs au sol indisponibles")

    def test_can_reverse_away_from_obstacle(self):
        self.c.settings["obstacle"] = True
        self.assertTrue(self.c.guarded_drive("backward", 25, follow=True))
        self.c.px.backward.assert_called_once_with(25)

    def test_cliff_blocks_reverse(self):
        self.c.settings["cliff"] = True
        self.assertFalse(self.c.guarded_drive("backward", 25, follow=True))
        self.c.px.backward.assert_not_called()

    def test_stalled_camera_stops_follow(self):
        self.c.last_frame_at = time.monotonic() - 2
        self.assertFalse(self.c.guarded_drive("forward", 40, follow=True))
        self.c.px.stop.assert_called_once()

    def test_camera_stall_watchdog_stops_without_another_frame(self):
        self.c.last_frame_at = time.monotonic() - 2
        self.c.check_control_timeout(1.5)
        self.assertFalse(self.c.settings["follow_me"])
        self.c.px.stop.assert_called_once()

    def test_follow_cache_does_not_skip_new_obstacle(self):
        self.c.guarded_drive("forward", 40, follow=True)
        self.c.settings["obstacle"] = True
        self.assertFalse(self.c.guarded_drive("forward", 40, follow=True))
        self.c.px.stop.assert_called_once()

    def test_follow_repeated_command_avoids_redundant_writes(self):
        self.c.guarded_drive("forward", 40, follow=True)
        self.c.guarded_drive("forward", 40, follow=True)
        self.c.px.forward.assert_called_once_with(40)

    def test_manual_watchdog_stops_stale_command(self):
        self.c.settings["follow_me"] = False
        self.c.control["forward"] = True
        self.c.last_manual_command = time.monotonic() - 3
        self.c.check_control_timeout(1.5)
        self.c.px.stop.assert_called_once()

    def test_manual_command_takes_over_follow(self):
        self.c.dispatch_action("forward")
        self.assertFalse(self.c.settings["follow_me"])
        self.c.px.forward.assert_called_once_with(70)

    def test_cliff_blocks_manual_command(self):
        self.c.settings["cliff"] = True
        self.c.dispatch_action("backward")
        self.c.px.backward.assert_not_called()

    def test_jpeg_is_encoded_once_for_multiple_viewers(self):
        self.c.cv2.reset_mock()
        buf = MagicMock()
        buf.tobytes.return_value = b"jpeg"
        self.c.cv2.imencode.return_value = (True, buf)
        self.c.latest_frame = object()
        self.c.frame_seq = 10
        self.c.jpeg_seq = -1
        first, second = self.c.mjpeg_gen(), self.c.mjpeg_gen()
        self.assertEqual(next(first), next(second))
        self.assertEqual(self.c.cv2.imencode.call_count, 1)
        self.c.frame_seq += 1
        next(first)
        self.assertEqual(self.c.cv2.imencode.call_count, 2)
        first.close()
        second.close()


if __name__ == "__main__":
    unittest.main()
