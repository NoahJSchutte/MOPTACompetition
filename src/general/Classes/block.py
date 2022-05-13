from src.general.Classes.type import Type
from typing import List
import random


class Block:
    def __init__(
            self,
            block_type: Type,
            key: str,
            end_time: int  # in minutes
    ):
        self.block_type = block_type
        self.key = key
        self.start_time = 0
        self.end_time = end_time
        self.start_times_surgeries: List[int] = list()  # this list is always sorted from first to last

    def add_surgery(self, surgery_start_time: int):
        self.start_times_surgeries.append(surgery_start_time)
        self.sort_surgeries()

    def get_number_of_surgeries(self):
        return len(self.start_times_surgeries)

    def get_key(self):
        return self.key

    def sort_surgeries(self):
        self.start_times_surgeries.sort()

    def get_type_name(self):
        return self.block_type.name

    def get_exp_rate(self):
        return self.block_type.exp_rate

    # SOLUTION CHANGE HEURISTIC
    def absolutely_perturb_random_start_time(self, perturbation: int):
        random_surgery = random.randint(0, self.get_number_of_surgeries()-1)
        self.start_times_surgeries[random_surgery] += perturbation
        self.sort_surgeries()

    # SOLUTION CHANGE HEURISTIC
    def relatively_perturb_random_start_time(self, perturbation: float):
        random_surgery = random.randint(0, self.get_number_of_surgeries()-1)
        self.start_times_surgeries[random_surgery] = \
            max(round((1 + perturbation) * self.start_times_surgeries[random_surgery]), self.start_time)
        self.sort_surgeries()

    # SOLUTION CHANGE HEURISTIC
    def tighten_schedule(self, space_reduction: float):
        # this function changes reduces the distance between each elective surgery by 'space reduction'-percent
        start_times_surgeries_base = self.start_times_surgeries.copy()
        cumulative_change = 0
        for i in range(self.get_number_of_surgeries()):
            if i > 0:
                difference = start_times_surgeries_base[i] - start_times_surgeries_base[i-1]
                change = round(difference * space_reduction)
                cumulative_change += change
                self.start_times_surgeries[i] -= cumulative_change
            else:  # for the first elective surgery the distance between the block start time is used
                difference = start_times_surgeries_base[i] - self.start_time
                change = round(difference * space_reduction)
                cumulative_change += change
                self.start_times_surgeries[i] -= cumulative_change
        self.sort_surgeries()

    def get_difference_list(self):
        start_times_minus_start = self.start_times_surgeries[1:]
        start_times_minus_end = self.start_times_surgeries[:-1]
        difference_list = [start_times_minus_start[i] - start_times_minus_end[i]
                           for i in range(len(self.start_times_surgeries)-1)]

        return difference_list

    def get_average_difference(self):
        difference_list = self.get_difference_list()

        return sum(difference_list) / len(difference_list)

    def equals(self, comparison_block: 'Block'):
        number_of_surgeries = self.get_number_of_surgeries()
        if number_of_surgeries != comparison_block.get_number_of_surgeries():
            return False

        self.start_times_surgeries.sort()
        comparison_block.start_times_surgeries.sort()
        for i in range(number_of_surgeries):
            if self.start_times_surgeries[i] != comparison_block.start_times_surgeries[i]:
                return False

        return True

    def get_distribution(self):
        return self.block_type.get_distribution()

    def get_mean_duration(self):
        return self.block_type.mean_duration

    def get_length(self):
        return self.end_time - self.start_time




