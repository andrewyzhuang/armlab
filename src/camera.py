"""
Camera wrapper for the Intel RealSense L515.

Runtime behavior:
- use RealSense intrinsics directly
- provide live RGB/depth for the control station
- expose a student-facing extrinsic calibration entrypoint that uses
  AprilTags + solvePnP at runtime
"""
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

COLOR_W, COLOR_H = 1280, 720
DEPTH_W, DEPTH_H = 1024, 768
TAG_SIZE_MM = 25.0
TILE_SIZE_MM = 50.0
WORKSPACE_MARGIN_MM = 50.0
# Known workspace tag centers in millimeters in the robot/world frame.
# Students can adjust these for their physical station.
TAG_WORLD_POINTS = {
    1: (100.0,  300.0, 0.0),
    2: (100.0, -300.0, 0.0),
    3: (400.0, -300.0, 0.0),
    4: (400.0,  300.0, 0.0),
}
MAX_X = 450
MIN_X = -50
MAX_Y = 450
MIN_Y = -450

class Camera:
    """Holds live frame data and owns the RealSense pipeline."""

    def __init__(self):
        self.video_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.depth_frame_raw = np.zeros((COLOR_H, COLOR_W), dtype=np.uint16)
        self.depth_frame_rgb = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.tag_image_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.workspace_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.grid_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)

        # Camera Calibration: extrinsic_matrix maps world-frame homogeneous points
        # to camera-frame homogeneous points; extrinsic_matrix_inv does the reverse.
        self.camera_calibrated = False
        self.intrinsic_matrix = np.eye(3, dtype=np.float64)
        self.extrinsic_matrix = np.eye(4, dtype=np.float64) # world to camera
        self.extrinsic_matrix_inv = np.eye(4, dtype=np.float64) # camera to world (cached inverse)
        # Workspace view: raw image pixel -> top-down workspace pixel.
        self.workspace_homography = None
        # Inverse lets mouse clicks in workspace view map back to raw image pixels.
        self.workspace_homography_inv = None

        # Click inspection state used by the GUI.
        self.last_click = np.array([0, 0], dtype=int)
        self.new_click = False

        # Toggled from the UI to overlay the Z-height heat map on the workspace view.
        self.heat_map_enabled = False

        # AprilTag state
        self.tag_world_points = dict(TAG_WORLD_POINTS)
        self.tag_detections = []
        self.tag_detect_interval = 6  # ~5 Hz at 30 FPS
        self.depth_scale_mm = 1.0   # raw depth units -> mm; overwritten once the pipeline starts
        self._pipeline_running = False
        self.camera_connected = False

        if len(rs.context().query_devices()) == 0:
            print("WARNING: No RealSense device found - running without video.")
            return

        try:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
            profile = self.pipeline.start(cfg)
            self.align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0
            # L515 visual_preset: 0=Custom, 1=Default (deprecated), 2=No Ambient,
            # 3=Low Ambient, 4=Max Range, 5=Short Range.
            # Short Range lowers the firmware depth floor toward the 250 mm spec minimum.
            # Otherwise the depth cut off from 411 mm.
            if depth_sensor.supports(rs.option.visual_preset):
                try:
                    depth_sensor.set_option(rs.option.visual_preset, 5)
                except RuntimeError:
                    pass
            if depth_sensor.supports(rs.option.min_distance):
                depth_sensor.set_option(rs.option.min_distance, 0)

            intr = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .intrinsics
            )
            self.intrinsic_matrix = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            self.tag_detector = Detector(families="tagStandard41h12")

            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.1)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 90)
            self.temporal_filter.set_option(rs.option.holes_fill, 3)

            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, 2)

            self._pipeline_running = True
            self.camera_connected = True
        except RuntimeError as exc:
            print(
                f"WARNING: Camera unavailable ({exc})\n"
                "  -> Connect the L515 via USB 3.2 for video."
            )

    def stop(self):
        if not self._pipeline_running:
            return
        self.pipeline.stop()
        self._pipeline_running = False

    def colorize_depth_frame(self, depth_lo_mm=300.0, depth_hi_mm=1200.0):
        # Jet colormap (blue=near, red=far); pixels outside range masked black.
        depth_mm = self.depth_frame_raw.astype(np.float32) * self.depth_scale_mm
        in_range = (depth_mm >= depth_lo_mm) & (depth_mm <= depth_hi_mm)

        # stretch to the depth span actually present, so small real variations still show up
        if np.any(in_range):
            depth_lo_mm = depth_mm[in_range].min()
            depth_hi_mm = max(depth_mm[in_range].max(), depth_lo_mm + 1e-3)

        normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
        normalized[in_range] = np.clip(
            (depth_mm[in_range] - depth_lo_mm) / (depth_hi_mm - depth_lo_mm) * 255.0,
            0, 255,
        ).astype(np.uint8)

        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        colorized[~in_range] = 0

        self.depth_frame_rgb = colorized

    def draw_tags_in_rgb_image(self):

        self.tag_image_frame = self.video_frame.copy()
        for detection in self.tag_detections:

            center = tuple(detection.center.astype(int))
            cv2.circle(self.tag_image_frame, center, 5, (0, 0, 255), -1)

            corners = list(detection.corners.astype(int))
            cv2.rectangle(self.tag_image_frame, corners[0], corners[2], (0, 255, 0), 2)

            cv2.putText(self.tag_image_frame, f"{detection.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            world = self.image_to_world(*center)
            if world is not None:
                cv2.putText(self.tag_image_frame, f"({world[0]:.0f},{world[1]:.0f},{world[2]:.0f})",
                            (center[0], center[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 3)


    def estimate_extrinsics_from_tags(self):
        """
        Uses AprilTag corners and known world-frame tag locations to estimate a
        camera-to-world transform. The result is stored in memory only and also
        printed to the terminal every time the user clicks Calibrate.
        """
        if not self.camera_connected:
            return False, "Camera offline - cannot calibrate."

        # pupil_apriltags orders detection.corners counter-clockwise starting at
        # the tag's top-right corner (verified empirically against a synthetic,
        # canonically-oriented tag): top-right, top-left, bottom-left,
        # bottom-right. Build the matching world-frame corner offsets (tags
        # assumed to lie flat, printed upright, with world +X = image "right"
        # and world +Y = image "up" as seen from directly overhead) from
        # TAG_SIZE_MM. If a tag is physically rotated relative to the world
        # axes, or the camera isn't mounted looking straight down, these
        # offsets need to be adjusted to match.
        half = TAG_SIZE_MM / 2.0
        
        corner_offsets = np.array([
                    [-half,  half, 0.0],  # top-right
                    [-half,  -half, 0.0],  # top-left
                    [ half, -half, 0.0],  # bottom-left
                    [ half, half, 0.0],  # bottom-right
                ])

        # Given tags detected in the image and their known world positions, solve for the transform between the two frames.
        world_pts = []
        img_pts = []

        for detection in self.tag_detections:
            center = self.tag_world_points.get(detection.tag_id)
            if center is None:
                continue
            center = np.array(center)
            for offset, corner in zip(corner_offsets, detection.corners):
                world_pts.append(center + offset)
                img_pts.append(corner)

        world_pts = np.array(world_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)

        if len(world_pts) < 4:
            return False, "Not enough AprilTag corners detected to calibrate."

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            world_pts,
            img_pts,
            self.intrinsic_matrix,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_IPPE
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)

            self.extrinsic_matrix = np.eye(4, dtype=float) # world to camera
            self.extrinsic_matrix[:3, :3] = R
            self.extrinsic_matrix[:3, 3] = tvec.ravel()

            self.extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)
            self.camera_calibrated = True

            self._update_workspace_transform()

            cam_pos_world = self.extrinsic_matrix_inv[:3, 3]
            return True, f"Calibrated! Camera position in World Frame: ({cam_pos_world[0]:.1f}, {cam_pos_world[1]:.1f}, {cam_pos_world[2]:.1f})"

        return False, f"solvePnP failed to converge."

    def depth_to_camera_point(self, x, y, depth_raw):
        """Convert an image pixel + raw depth unit to a camera-frame 3D point (mm)."""
        # a pixel plus a raw depth reading, to a 3D point in the camera frame.
        z_c = depth_raw * self.depth_scale_mm
        if z_c <= 0:
            return None

        fx, fy = self.intrinsic_matrix[0,0], self.intrinsic_matrix[1,1]
        cx, cy = self.intrinsic_matrix[0,2], self.intrinsic_matrix[1,2]
        x_c = (x - cx) * z_c / fx
        y_c = (y - cy) * z_c / fy
        return np.array([x_c, y_c, z_c, 1.0])

    def camera_to_world(self, camera_point):
        """Convert a camera-frame 3D point to a world-frame 3D point."""
        if camera_point is None or not self.camera_calibrated:
            return None

        world_pt = (self.extrinsic_matrix_inv @ camera_point)[:3]
        return world_pt

    def image_to_world(self, x, y):
        """Pixel (x, y) -> world-frame (X, Y, Z) in mm using live depth."""
        d_raw = self.depth_frame_raw[int(y), int(x)]
        p_cam = self.depth_to_camera_point(x,y,d_raw)
        return self.camera_to_world(p_cam)

    def workspace_pixel_to_image(self, x_px, y_px):
        """Map a workspace-view pixel back to the original image pixel."""

        if self.workspace_homography_inv is None:
            return x_px, y_px

        pt = np.array([[[float(x_px), float(y_px)]]])
        pt_raw = cv2.perspectiveTransform(pt, self.workspace_homography_inv)

        u_raw = float(pt_raw[0][0][0])
        v_raw = float(pt_raw[0][0][1])
        return u_raw, v_raw

    def _update_workspace_transform(self):
        """Build the homography used for the top-down workspace view."""
        self.workspace_homography = None
        self.workspace_homography_inv = None

        detections_by_id = {d.tag_id: d for d in self.tag_detections}
        dest_pts = np.array([
            [WORKSPACE_MARGIN_MM, COLOR_H - WORKSPACE_MARGIN_MM],
            [COLOR_W - WORKSPACE_MARGIN_MM, COLOR_H - WORKSPACE_MARGIN_MM],
            [COLOR_W - WORKSPACE_MARGIN_MM, WORKSPACE_MARGIN_MM],
            [WORKSPACE_MARGIN_MM, WORKSPACE_MARGIN_MM],
        ])

        src_pts = np.array([
            [303, 562],
            [1088, 551],
            [1154, 91],
            [232, 112]
        ])
        matched_dest_pts = []

        for tag_id, dest in zip(self.tag_world_points, dest_pts):
            detection = detections_by_id.get(tag_id)
            if detection is None:
                continue
            matched_dest_pts.append(dest)

        dest_pts = np.array(matched_dest_pts, dtype=np.float64)

        H, _ = cv2.findHomography(src_pts, dest_pts)

        if H is not None:
            self.workspace_homography = H
            self.workspace_homography_inv = np.linalg.inv(H)
            print(H)

    def update_workspace_frame(self):
        """Warp the RGB frame into the top-down workspace view."""
        if self.video_frame is None or self.workspace_homography is None:
            return

        self.workspace_frame = cv2.warpPerspective(
            self.video_frame,
            self.workspace_homography,
            (COLOR_W, COLOR_H),
            flags=cv2.INTER_NEAREST
        )

    def _to_qimage(self, frame):
        """Convert a NumPy RGB image into a Qt image for display."""
        try:
            h, w = frame.shape[:2]
            return QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888).copy()
        except Exception:
            return None

    def project_world_points(self, world_points):

        if not self.camera_calibrated:
            return None
        rvec, _ = cv2.Rodrigues(self.extrinsic_matrix[:3, :3])
        tvec = self.extrinsic_matrix[:3, 3]
        img_pts, _ = cv2.projectPoints(world_points, rvec, tvec, self.intrinsic_matrix, None)
        return img_pts.reshape(-1, 2)

    def project_grid(self):

        xsteps = int((MAX_X - MIN_X)/TILE_SIZE_MM) + 1
        xs = np.linspace(MIN_X, MAX_X, xsteps)

        ysteps = int((MAX_Y - MIN_Y)/TILE_SIZE_MM) + 1
        ys = np.linspace(MIN_Y, MAX_Y, ysteps)

        X, Y = np.meshgrid(xs, ys)
        world_pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)

        frame = self.video_frame.copy()
        img_pts = self.project_world_points(world_pts)
        if img_pts is not None:
            for u, v in img_pts:
                cv2.circle(frame, (int(u), int(v)), 3, (0, 255, 255), -1)

        self.grid_frame = frame

    def heat_map(self):
        if not self.heat_map_enabled or self.workspace_homography_inv is None:
            return

        x_ws = np.linspace(50, 1234, 100)
        y_ws = np.linspace(50, 670, 100)

        X, Y = np.meshgrid(x_ws, y_ws)

        z_map = np.zeros((len(y_ws), len(x_ws)))

        for i in range(len(y_ws)):
            for j in range(len(x_ws)):
                px = self.workspace_homography_inv @ np.array([X[i, j], Y[i, j], 1])
                px = px / px[2]
                x_px, y_px = px[0], px[1]

                world_pt = self.image_to_world(x_px, y_px)
                if world_pt is None:
                    continue

                z_map[i, j] = world_pt[2]

        # plot: colorize the Z heights (same jet-colormap pattern as colorize_depth_frame).
        # Use percentiles instead of a fixed range so a few outlier-tall pixels (the arm)
        # don't wash out small, real height changes across the table.
        lo, hi = np.percentile(z_map, [10, 90])
        normalized = np.clip((z_map - lo) / max(hi - lo, 1e-3) * 255, 0, 255)
        colorized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        colorized = cv2.resize(colorized, (COLOR_W, COLOR_H))
        self.workspace_frame = cv2.addWeighted(self.workspace_frame, 0.5, colorized, 0.5, 0)


class VideoThread(QThread):
    """Polls the RealSense pipeline at ~30 Hz and emits Qt signals for the GUI."""

    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        self._running = True
        self._frame_counter = 0

    def stop(self):
        self._running = False

    def run(self):
        if not self.camera.camera_connected:
            while self._running:
                time.sleep(0.5)
            return

        while self._running:
            start = time.time()

            try:
                frames = self.camera.pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                if not self._running:
                    break
                continue

            aligned = self.camera.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = self.camera.spatial_filter.process(depth_frame)
            depth_frame = self.camera.temporal_filter.process(depth_frame)
            depth_frame = self.camera.hole_filling.process(depth_frame)

            bgr = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            self.camera.video_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.camera.depth_frame_raw = depth
            self.camera.colorize_depth_frame()

            if self._frame_counter % self.camera.tag_detect_interval == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                self.camera.tag_detections = self.camera.tag_detector.detect(gray)
            self._frame_counter += 1

            self.camera.draw_tags_in_rgb_image()
            self.camera.update_workspace_frame()
            self.camera.heat_map()
            self.camera.project_grid()

            rgb_qt   = self.camera._to_qimage(self.camera.video_frame)
            depth_qt = self.camera._to_qimage(self.camera.depth_frame_rgb)
            tag_qt   = self.camera._to_qimage(self.camera.tag_image_frame)
            workspace_qt = self.camera._to_qimage(self.camera.workspace_frame)
            grid_qt = self.camera._to_qimage(self.camera.grid_frame)

            if rgb_qt and depth_qt and tag_qt and workspace_qt and grid_qt:
                self.updateFrame.emit(rgb_qt, depth_qt, tag_qt, workspace_qt, grid_qt)

            elapsed = time.time() - start
            time.sleep(max(0.0, 1 / 30 - elapsed))


if __name__ == '__main__':
    import sys

    from PyQt5.QtCore import Qt, pyqtSlot
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication, QLabel

    app = QApplication(sys.argv)
    camera = Camera()
    print(f"Intrinsic matrix:\n{camera.intrinsic_matrix}")

    wins = {}
    for title in ("RGB", "Depth", "Tags", "Workspace", "Grid"):
        label = QLabel(title)
        label.setWindowTitle(title)
        label.setAlignment(Qt.AlignCenter)
        label.resize(640, 360)
        label.show()
        wins[title] = label

    @pyqtSlot(QImage, QImage, QImage, QImage, QImage)
    def on_frame(rgb, depth, tags, workspace, grid):
        for label, image in zip(wins.values(), (rgb, depth, tags, workspace, grid)):
            label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    thread = VideoThread(camera)
    thread.updateFrame.connect(on_frame)
    thread.start()

    result = app.exec_()
    thread.stop()
    thread.wait(500)
    camera.stop()
    sys.exit(result)
