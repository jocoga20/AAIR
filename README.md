This project emulates a robot (red square) moving in a grid world. The robot starts at the charge station (green square) and visits *n* waypoints (blue squares) before coming back to the station.

Every move drains one unit of battery, when it ends the robot cannot move anymore, if it happens before completing the task, the task is failed.

The robot follows a policy but there is a small chance (default 0.1) that policy action is ignored and one of the two orthogonal direction is chosen instead. This emulates any uncertainty of the terrain like obstacles or unstable ground.