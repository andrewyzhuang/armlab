"""
Kinematics checks.

Offline (no arm needed):
  - error_test_ik:        IK_geometric vs numerical IK round trip
  - reachability_map:     top-down grasp reachability over the board

On the real Lite 6 (the arm never moves):
  - error_test:           FK_dh against the SDK's forward kinematics
  - error_test_ik_vendor: SDK inverse kinematics, checked through our FK
"""

from lite6arm import Lite6Arm
from kinematics import error_test, error_test_ik, error_test_ik_vendor, reachability_map

if __name__ == '__main__':
    #error_test_ik(iterations=300)
    reachability_map()

    arm = Lite6Arm()
    if not arm.connected:
        print("Arm unavailable; skipping vendor FK/IK tests.")
    #else:
        #error_test(arm, iterations=200)
        #error_test_ik_vendor(arm, iterations=300)
