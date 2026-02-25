import numpy as np

class Robot:
    def __init__(self, x, y, full_battery = 20):
        self.position = np.array([x, y])
        self.battery = full_battery
        self.full_battery = full_battery
    
    def move(self, direction):
        self.position += direction
        self.battery -= 1
    
    def can_move(self):
        return self.battery > 0

    def recharge(self):
        self.battery = self.full_battery