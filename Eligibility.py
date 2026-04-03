class Eligibility:
    def __init__(self, decay):
        self.eligibility_dict = {}
        self.skipped_updates = {}
        self.decay = decay
    
    def increase_skipped_updates(self):
        for key in self.skipped_updates.keys():
            self.skipped_updates[key] += 1
    
    def get_eligibility(self, key):
        if not key in self.eligibility_dict.keys():
            self.eligibility_dict[key] = 1
            self.skipped_updates[key] = 0
            return 1
        
        if self.skipped_updates[key] > 0:
            self.eligibility_dict[key] *= self.decay ** self.skipped_updates[key]
            self.eligibility_dict[key] += 1
            self.skipped_updates[key] = 0

        return self.eligibility_dict[key]