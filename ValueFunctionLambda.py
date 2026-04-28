import warnings

from Eligibility import Eligibility
from ValueFunction import ValueFunction

class ValueFunctionLambda(ValueFunction):
    def __init__(self, step_size_lambda, reward_discount = 1, eligibility_decay = 0.5):
        if eligibility_decay == 0:
            warnings.warn('If decay is zero it would be better to use ValueFunction class instead.', UserWarning)
        super().__init__(step_size_lambda, reward_discount)
        self.eligibility = Eligibility(decay=eligibility_decay * reward_discount)

    def update(self, old_state_key, new_state_key, reward):
        v1 = self.get(old_state_key)
        v2 = self.get(new_state_key)

        self.eligibility.increase_skipped_updates()
        e = self.eligibility.get_eligibility(old_state_key)
        
        new_value = v1 + self.step_size_lambda(self.t) * e * (reward + self.reward_discount * v2 - v1)
        self.value_dict[old_state_key] = new_value
        self.monitor_state(old_state_key, new_value)
        self.t += 1