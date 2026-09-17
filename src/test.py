"""
Connects to the real Lite 6 and runs kinematics.error_test to check
FK_dh against the SDK's forward kinematics.
"""

from lite6arm import Lite6Arm
from kinematics import error_test

if __name__ == '__main__':
    arm = Lite6Arm()
    if not arm.connected:
        print("Arm unavailable.")
    else:
        error_test(arm, iterations=200)
