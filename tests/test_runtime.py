"""Exercise the real controller with hardware and camera interfaces replaced."""
import importlib.util
from pathlib import Path
import sys
import threading
import time
import tempfile
import logging
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
        self.c.video_restart_pending = False
        self.c.settings.update(video_resolution="640x480", video_error="")
        self.c.CAM_SIZE = (640, 480)
        self.c.latest_frame = None
        self.c.frame_seq = 0
        self.c.jpeg_seq = -1
        self.c.latest_jpeg = None

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

    def test_camera_settings_drive_both_servos_and_stop_follow(self):
        self.c.apply_camera_settings(12, -12)
        self.c.px.set_cam_pan_angle.assert_called_once_with(12)
        self.c.px.set_cam_tilt_angle.assert_called_once_with(-12)
        self.assertFalse(self.c.settings["tracking"])
        self.assertFalse(self.c.settings["follow_me"])

    def test_camera_adjustment_preserves_manual_driving(self):
        self.c.settings.update(tracking=False, follow_me=False)
        self.c.control["forward"] = True
        self.c.apply_camera_settings(5, None)
        self.assertTrue(self.c.control["forward"])
        self.c.px.stop.assert_not_called()

    def test_snapshot_and_mjpeg_share_jpeg_cache(self):
        self.c.cv2.imencode.reset_mock()
        buffer = MagicMock()
        buffer.tobytes.return_value = b"jpeg"
        self.c.cv2.imencode.return_value = True, buffer
        self.c.latest_frame = object()
        self.c.frame_seq = 4
        data, seq = self.c.encoded_video_frame()
        self.assertEqual((data, seq), (b"jpeg", 4))
        stream = self.c.mjpeg_gen()
        self.assertIn(data, next(stream))
        stream.close()
        self.c.cv2.imencode.assert_called_once()

    def test_stale_video_frame_is_not_served_as_live(self):
        self.c.latest_frame = object()
        self.c.last_frame_at = time.monotonic() - 3
        self.assertIsNone(self.c.encoded_video_frame()[0])

    def test_many_finite_frame_requests_keep_receiving_new_images(self):
        buffer = MagicMock()
        buffer.tobytes.return_value = b"jpeg"
        self.c.cv2.imencode.return_value = True, buffer
        self.c.latest_frame = object()
        with patch.object(self.c, "Response", side_effect=lambda data, **kw: (data, kw)):
            for seq in range(1, 51):
                self.c.frame_seq = seq
                data, options = self.c.video_frame()
                self.assertEqual(data, b"jpeg")
                self.assertEqual(options["headers"]["X-Frame-Sequence"], str(seq))
                self.assertEqual(options["mimetype"], "image/jpeg")

    def test_video_access_filter_keeps_failures(self):
        access_filter = self.c.VideoAccessFilter()
        def record(text):
            return logging.LogRecord("werkzeug", logging.INFO, "", 0, text, (), None)
        self.assertFalse(access_filter.filter(record('"GET /video_frame?t=1 HTTP/1.1" 200 -')))
        self.assertTrue(access_filter.filter(record('"GET /video_frame?t=1 HTTP/1.1" 503 -')))

    def test_tracking_geometry_is_stable_across_resolutions(self):
        for size in ((640, 480), (1280, 720), (1920, 1080)):
            with self.subTest(size=size):
                self.c.CAM_SIZE = size
                sx, sy = size[0] / 640, size[1] / 480
                geometry = self.c.tracking_geometry(400*sx, 120*sy, 80*sx, 80*sy)
                self.assertEqual(geometry, (120, -80, 80))

    def test_video_resolution_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models/video_settings.json"
            with patch.object(self.c, "VIDEO_SETTINGS", path):
                self.c.save_video_resolution("1280x720")
                self.assertEqual(self.c.load_video_resolution(), "1280x720")
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_corrupt_video_settings_use_standard_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_settings.json"
            with patch.object(self.c, "VIDEO_SETTINGS", path):
                for contents in ('{bad', '[]', '{"resolution": []}', '{"resolution": "8k"}'):
                    path.write_text(contents)
                    self.assertEqual(self.c.load_video_resolution(), "640x480")

    def test_rejected_hd_mode_falls_back_and_closes_failed_camera(self):
        first, second = MagicMock(), MagicMock()
        first.configure.side_effect = RuntimeError("unsupported mode")
        with patch.object(self.c, "Picamera2", side_effect=[first, second]):
            instance, resolution, error = self.c.create_video_camera("1920x1080")
        self.assertIs(instance, second)
        self.assertEqual(resolution, "640x480")
        self.assertTrue(error)
        first.close.assert_called_once()
        second.start.assert_called_once()

    def test_standard_camera_failure_is_reported(self):
        instance = MagicMock()
        instance.start.side_effect = RuntimeError("camera disconnected")
        with patch.object(self.c, "Picamera2", return_value=instance):
            with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
                self.c.create_video_camera("640x480")

    def video_request(self, data):
        self.c.request.method = "POST"
        self.c.request.get_json.return_value = data
        return self.c.video_settings_route()

    def test_resolution_request_stops_motors_before_restart(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(self.c, "VIDEO_SETTINGS", Path(directory) / "video.json"), \
                patch.object(self.c, "jsonify", side_effect=lambda data: data), \
                patch.object(threading.Thread, "start") as start:
            result, status = self.video_request({"resolution": "1280x720"})
            self.assertEqual(status, 202)
            self.assertTrue(result["restarting"])
            self.assertFalse(self.c.settings["follow_me"])
            self.c.px.stop.assert_called_once()
            self.assertEqual(self.c.load_video_resolution(), "1280x720")
            start.assert_called_once()

    def test_invalid_resolution_never_stops_or_restarts_robot(self):
        with patch.object(self.c, "jsonify", side_effect=lambda data: data), \
                patch.object(threading.Thread, "start") as start:
            for data in ({"resolution": "8k"}, {"resolution": []}, ["1280x720"]):
                self.assertEqual(self.video_request(data)[1], 400)
            self.c.px.stop.assert_not_called()
            start.assert_not_called()

    def test_same_resolution_does_not_restart(self):
        with patch.object(self.c, "jsonify", side_effect=lambda data: data), \
                patch.object(threading.Thread, "start") as start:
            result = self.video_request({"resolution": "640x480"})
            self.assertFalse(result["restarting"])
            self.c.px.stop.assert_not_called()
            start.assert_not_called()

    def test_failed_video_save_does_not_interrupt_robot(self):
        with patch.object(self.c, "jsonify", side_effect=lambda data: data), \
                patch.object(self.c, "save_video_resolution", side_effect=OSError("read only")), \
                patch.object(threading.Thread, "start") as start:
            self.assertEqual(self.video_request({"resolution": "1280x720"})[1], 500)
            self.c.px.stop.assert_not_called()
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
