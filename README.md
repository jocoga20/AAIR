This project applies Temporal Difference algorithms to estimate the value function of a robot moving in a grid world.

## Task
The robot starts the episode at the charge station, where the battery fully recharges every time it lands there.
Every step costs one unit of battery, when it is fully drained the robot stops and the episode is failed.
The objective is to visit all waypoints and then come back to the charge station without running out of battery. 

## States of the value function
The state is determined by the position of the robot (x, y), its battery level and the waypoint status.
Assuming *n* waypoints to visit, waypoint status is the integer number corresponding to a binary string constructed in the following way: each bit corresponds to a waypoint, the value is 1 if that waypoint was visited during the episode and 0 otherwise.
By default the grid world is 20x20, *n*=5 and battery level assumes integer values between 0 and `FULL_BATTERY` (default is 80). These settings are configurable in the `config.py` file.

## Rewards
COMPLETE_REWARD = given if the episode ended with success (all waypoints visited and charge station reached)

WAYPOINT_REWARD = given when a waypoint is visited for the first time. Note

CHARGE_REWARD = given when robot is at the charge station

EMPTY_REWARD = given if the robot is on a normal cell (not a waypoint and not a charge station either)

FAIL_REWARD = given if the task is failed

## Policies
These are the policies used in the experiments:

### Greedy Policy
The robot will always head for the nearest waypoint. If they are all visited it will head to the charge station.

### Pedant Policy
The robot will head to the nearest waypoint if the battery is enough to visit it and then going back to the charge station, assuming it will always follow the minimal path. Formally it goes to the nearest_waypoint if the following condition is satisfied:

`robot.battery >= min_path_len(robot.position, nearest_waypoint) + min_path_len(nearest_waypoint, charge_station)`

### Secure Policy
In order to take into account deviations from the minimal path, Secure Policy adds some margin *m* to the minimal path. The condition to satisfy becomes:

`robot.battery >= min_path_len(robot.position, nearest_waypoint) + min_path_len(nearest_waypoint, charge_station) + m`