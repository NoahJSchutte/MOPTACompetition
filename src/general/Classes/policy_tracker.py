
class PolicyTracker:
    def __init__(
            self,
            policy: str
    ):
        self.policy = policy
        self.cancellations = list()
        self.block_types = list()
        self.waited_till_free = list()

    def add_cancellations(self, cancellations: int):
        self.cancellations.append(cancellations)

    def add_block_type(self, block_type: str):
        self.block_types.append(block_type)

    def add_waited_till_free(self, waited_till_free: bool):
        self.waited_till_free.append(waited_till_free)

    def get_type_rates(self):
        rate_dict = dict()
        total_types = len(self.block_types)

        for block_type in self.block_types:
            if block_type not in rate_dict.keys():
                rate_dict[block_type] = 1/total_types
            else:
                rate_dict[block_type] += 1/total_types

        return rate_dict

    def get_average_cancellations(self):
        return sum(self.cancellations) / len(self.cancellations)
