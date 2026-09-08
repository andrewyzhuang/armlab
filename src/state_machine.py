"""
State machine for the arm lab runtime shell.
"""
import csv
import os
import time

from PyQt5.QtCore import QThread, pyqtSignal

# Coded by Claude
WAYPOINTS_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.csv")

class Waypoint:

    def __init__(self, joint_angles, gripper_state):
        self.joint_angles = joint_angles
        self.gripper_state = gripper_state

class StateMachine:

    def __init__(self, arm, camera):
        self.arm = arm
        self.camera = camera
        self.status_message = "Idle - waiting for input."
        self.current_state = "idle"
        self.next_state = "idle"
        self.gripper_state = "open"
        self.waypoints = list()

        self._handlers = {
            "initial_pose":      self.initial_pose,
            "sleep_arm":         self.sleep_arm,
            "estop":             self.estop,
            "calibrate":         self.calibrate,
            "add_waypoint":      self.add_waypoint,
            "clear_waypoints":   self.clear_waypoints,
            "playback_waypoints":self.playback_waypoints,
            "direct_control":    self.direct_control,
            "pick_place":        self.pick_place,
        }

    def set_next_state(self, state):
        self.next_state = state

    def _go_idle(self, message):
        self.status_message = message
        self.current_state = "idle"
        self.next_state = "idle"

    def run(self):
        self._handlers.get(self.next_state, self.idle)()

    def idle(self):
        if self.current_state != "idle":
            self.status_message = "Idle - waiting for input."
        self.current_state = "idle"

    def direct_control(self):
        if not self.arm.connected or not self.arm.initialized:
            self._go_idle("Direct control unavailable until the arm is initialized.")
            return
        if self.current_state != "direct_control":
            self.status_message = "Direct Control - use jog controls to move the arm."
        self.current_state = "direct_control"

    def estop(self):
        if self.current_state != "estop":
            self.arm.stop()
            self.status_message = "STOP active - clear the arm and click Go Home to recover."
            self.current_state = "estop"

    def initial_pose(self):
        if not self.arm.connected:
            self._go_idle("Arm offline - cannot initialize.")
            return
        self.status_message = "Initializing arm..."
        if not self.arm.initialize():
            self._go_idle("Failed to initialize arm.")
        else:
            self._go_idle("Arm initialized - at initial pose.")

    def sleep_arm(self):
        if not self.arm.connected:
            self._go_idle("Arm offline - cannot send Sleep.")
            return
        self.status_message = "Sleep: returning to home position, motors will shut off..."
        self.arm.sleep()
        self._go_idle("Sleep: home position reached, motors off.")

    def calibrate(self):
        _, message = self.camera.estimate_extrinsics_from_tags()
        self._go_idle(message)

    def add_waypoint(self):
        # Each waypoint stores joint angles + gripper state together.
        # Two consecutive waypoints can have identical joint angles but different gripper states
        # (e.g. WP: arm at grasp position, gripper open -> next WP: same position, gripper closed).
        # Playback executes each waypoint sequentially: move joints first, then apply gripper state.
        joint_angles = self.arm.get_joint_angles()

        if joint_angles is None:
            self._go_idle("Joint angles not recovered from function")
            return

        if self.current_state != "add_waypoint":
            self.current_state = "add_waypoint"

        new_wp = Waypoint(list(joint_angles), self.gripper_state)
        self.waypoints.append(new_wp)
        self._save_waypoints_csv()

        self._go_idle(f"Waypoint {len(self.waypoints)} recorded "
                      f"Joint angles: {list(joint_angles)}, gripper: {self.gripper_state}")


    def clear_waypoints(self):
        wp = len(self.waypoints)
        self.waypoints = list()
        self._go_idle(f"Cleared {wp} waypoints")

    # Coded by Claude
    def _save_waypoints_csv(self):
        with open(WAYPOINTS_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper_state"])
            for i, wp in enumerate(self.waypoints):
                writer.writerow([i, *wp.joint_angles, wp.gripper_state])

    def playback_waypoints(self):
        # For each waypoint: move to joint angles (wait), then apply gripper state (wait), then advance.
        
        if not self.arm.connected or not self.arm.initialized:
            self._go_idle("Playback waypoints unavailable until the arm is initialized.")
            return

        if not self.waypoints:
            self._go_idle("No waypoints to playback.")
            return

        # Uncomment if gripper does not move!
        # self.arm.enable()
        
        for waypoint in self.waypoints:

            self.arm.set_joint_angles(waypoint.joint_angles, wait=True)
            
            match waypoint.gripper_state:

                case "open":
                    self.open_gripper()

                case "close":
                    self.close_gripper()

                case "off":
                    self.stop_gripper()

        self._go_idle(f"Playback complete: finished {len(self.waypoints)} waypoints")

    def open_gripper(self):
        self.arm.open_gripper(wait=True)
        self.gripper_state = "open"

    def close_gripper(self):
        self.arm.close_gripper(wait=True)
        self.gripper_state = "close"

    def stop_gripper(self):
        self.arm.stop_gripper()
        self.gripper_state = "off"

    def pick_place(self):
        # TODO: student lab
        # Student lab: click an object in the video feed to pick it up and place it.
        # 1. Poll camera.new_click; when True, read camera.last_click for pixel (u, v).
        # 2. camera.image_to_world(u, v) -> world XYZ using live depth + extrinsics.
        # 3. arm.get_ik(pose) -> joint angles; arm.set_joint_angles(...) to move above object.
        # 4. arm.close_gripper() to grasp, move to drop position, arm.open_gripper().
        # State persists until the GUI toggle is unchecked.
        if self.current_state != "pick_place":
            self.status_message = "Pick & Place - click an object in the video feed."
            self.current_state = "pick_place"

class StateMachineThread(QThread):
    updateStatusMessage = pyqtSignal(str)

    def __init__(self, state_machine, parent=None):
        QThread.__init__(self, parent=parent)
        self.sm = state_machine
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)
