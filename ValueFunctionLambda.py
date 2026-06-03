import warnings

from Eligibility import Eligibility
from ValueFunction import ValueFunction
from config import step_size_default_rule

class ValueFunctionLambda(ValueFunction):
    def __init__(self, step_size_lambda = step_size_default_rule, reward_discount = 1, eligibility_decay = 0.5):
        if eligibility_decay == 0:
            warnings.warn('If decay is zero it would be better to use ValueFunction class instead.', UserWarning)
        super().__init__(step_size_lambda, reward_discount)
        self.eligibility = Eligibility(decay=eligibility_decay * reward_discount)

    def update(self, old_state_key, new_state_key, reward):
        v1 = self.get(old_state_key)
        v2 = self.get(new_state_key)

        self.eligibility.increase_skipped_updates()
        td_error = reward + self.reward_discount * v2 - v1
        td_error *= self.step_size_lambda(self.t)
        self.update_state(key=old_state_key, value=v1 + self.eligibility.get_eligibility(old_state_key) * td_error)

        for state in self.values_dict.keys():
            if state == old_state_key:
                continue
            et = self.eligibility.get_eligibility(state)
            old_value = self.get(state)
            self.update_state(key=state, value=old_value + et * td_error)

        self.t += 1
    
    def autoname_path(self):
        return f'vfl{self.eligibility.decay}.{self.reward_discount}rd.pkl'