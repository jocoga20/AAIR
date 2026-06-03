from config import step_size_default_rule
from fileutils import *


class ValueFunction:
    def __init__(self, step_size_lambda = step_size_default_rule, reward_discount = 1):
        self.t = 0
        self.values_dict = dict()
        self.step_size_lambda = step_size_lambda
        self.reward_discount = reward_discount
    
    def update_state(self, key, value):
        self.values_dict.setdefault(key, [0]).append(value)
        
    def update(self, old_state_key, new_state_key, reward):
        v1 = self.get(old_state_key)
        v2 = self.get(new_state_key)
        
        new_value = v1 + self.step_size_lambda(self.t) * (reward + self.reward_discount * v2 - v1)
        self.update_state(old_state_key, new_value)
        self.t += 1
    
    def get(self, key):
        values = self.values_dict.get(key)
        return 0 if values is None else values[-1]
    
    def autoname_path(self):
        return f'vf.{self.reward_discount}rd.pkl'

    def save(self, path='vfs', filename=None):
        if filename is None:
            filename = self.autoname_path()
        path = f'{path}/{filename}'
        save(path, self.values_dict)
        print(f'Saved {path}')
        return self
    
    def load(self, path='vfs', filename=None):
        if filename is None:
            filename = self.autoname_path()
        self.values_dict = load(f'{path}/{filename}')
        return self