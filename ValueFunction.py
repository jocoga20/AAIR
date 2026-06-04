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
    
    def _autoname_filepath(self):
        return f'vfs/vf.{self.reward_discount}rd.pkl'

    def save(self, filepath: str = None):
        filepath = self.__filepath_logic(filepath)

        save(filepath, self.values_dict)
        print(f'Saved {filepath}')
        return self
    
    def load(self, filepath: str = None):
        filepath = self.__filepath_logic(filepath)

        self.values_dict = load(filepath)
        return self

    def __filepath_logic(self, filepath: str):
        if filepath is None:
            filepath = self._autoname_filepath()
        return filepath