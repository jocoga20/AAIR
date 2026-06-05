In this project Temporal Difference algorithms are applied to estimate the value function of a robot moving in a grid world.

Both TD(0) and TD($\lambda$) works in real-time: they can approximate the value function during the episode.

## TD(0)
Given a timestep $t$ we have a status ($S_t$) and a reward ($R_{t+1}$) that depends on the status we arrived in.

After the transition from $t$ to $t+1$ we update the Value Function ($V$) as follows:

$V(S_t) = V(S_t) + \alpha(t) D$

Where $D$ is the TD Error:

$D = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

$\gamma$ = reward discount

$\alpha(t)$ = learning rate (or step size). A decreasing function depending on $t$ (`step_size_default_rule` in `config.py`).

## TD($\lambda$)
This algorithm adds the eligibility trace of a state.

$E_t(s)$ = $\lambda \alpha E_{t-1}(s)$ + (1 if $x$ = $x(t)$ else 0)

For each state s of trajectory, the value function is updated as follows:

$V(s) = V(s) + \alpha(t) E_t(s) D$

## Task
The robot starts the episode at the charge station, where the battery fully recharges every time it lands there.

Each step costs one unit of battery, when it is fully drained the robot stops and the episode fails.

The objective is to visit all waypoints and then come back to the charge station without running out of battery. 

## States of the value function
The state is determined by the position of the robot (x, y), the battery level and the waypoint status.

Assuming *n* waypoints to visit, waypoint status is a binary string built as follows: each bit corresponds to a waypoint, the value is 1 if that waypoint was visited during the episode and 0 otherwise.

By default the grid world is 20x20, *n*=5 and battery level assumes integer values between 0 and `FULL_BATTERY` (default is 80). These settings are configurable in the `config.py` file.

## Rewards
Rewards (configurable in `config.py`) are added to the score if a certain condition applies

COMPLETE_REWARD = episode ended with success (all waypoints visited and charge station reached)

WAYPOINT_REWARD = a waypoint is visited for the first time

CHARGE_REWARD = every time the robot reaches the charge station

EMPTY_REWARD = the robot is on a normal cell (not a waypoint and not a charge station either)

FAIL_REWARD = if the episode failed

## Policies
Temporal Difference Learning aims to evaluate the value function under a certain policy. The policy determines the next action to do based on the current status. However there is a small chance it is not possible to follow the action, this is controlled by the `pmax` value in the `Experiment.run()` function.

The following policies are available in `policies.py`:

### Greedy Policy
The robot will always head for the nearest waypoint. If they are all visited it will head to the charge station.

### Pedant Policy
The robot will head to the nearest waypoint if the battery is enough to visit it and then going back to the charge station, assuming it will always follow the minimal path. Formally it goes to the `nearest_waypoint` if the following condition is true:

`robot.battery >= min_path_len(robot.position, nearest_waypoint) + min_path_len(nearest_waypoint, charge_station)`

### Secure Policy
In order to take into account deviations from the minimal path, Secure Policy adds some margin *m* to the minimal path. The condition to satisfy becomes:

`robot.battery >= min_path_len(robot.position, nearest_waypoint) + min_path_len(nearest_waypoint, charge_station) + m`

## How to run
In `main.timeplot.py` it is possible to run the task for some episodes and then view an animation of it.

In `main.convergence.py` it is possible to generate plots for certain status, following the convergence of some elements of the value function through the episodes. By default it saves the plots in the `imgs` folder (it will be automatically created if not present) then it will put plots in n subfolders depending on the number of waypoints reached in that status ($i$), subfolders will be named [$i$]wps. 