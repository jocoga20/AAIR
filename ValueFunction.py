class ValueFunction:
    def __init__(self, step_size, discount_factor = 1, default_value = 0):
        self.sparse_dict = {}
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.default_value = default_value

    def update(self, old_state_key, new_state_key, reward):
        v1 = self.get(old_state_key)
        v2 = self.get(new_state_key)
        self.sparse_dict[old_state_key] = v1 + self.step_size * (reward + self.discount_factor * v2 - v1)
    
    def get(self, key):
        return self.sparse_dict.get(key, self.default_value)