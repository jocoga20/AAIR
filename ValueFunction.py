class ValueFunction:
    def __init__(self, step_size_lambda, reward_discount = 1):
        self.t = 0
        self.value_dict = {}
        self.step_size_lambda = step_size_lambda
        self.reward_discount = reward_discount

    def update(self, old_state_key, new_state_key, reward):
        v1 = self.get(old_state_key)
        v2 = self.get(new_state_key)
        
        self.value_dict[old_state_key] = v1 + self.step_size_lambda(self.t) * (reward + self.reward_discount * v2 - v1)
        self.t += 1
    
    def get(self, key):
        return self.value_dict.get(key, 0)