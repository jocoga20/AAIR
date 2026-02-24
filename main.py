import numpy as np

class Robot:
    def __init__(self, x, y, full_battery = 20):
        self.position = np.array([x, y])
        self.battery = full_battery
        self.full_battery = full_battery

running_episode = True
charge_station = np.array([0, 0])
robot = Robot(0, 0)
N = 5
np.random.seed(42)
waypoints = np.random.randint(20, size=(N, 2))
score = 0

while running_episode:
    reward = 0
    
    score += reward