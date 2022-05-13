from typing import Dict
from src.general.Classes.day import Day
import pandas as pd

class Solution:
    def __init__(
            self, nr_surgeries
    ):
        self.days: Dict[str: Day] = dict()  # dict so we can compare solutions
        self.simulated_objective: float
        self.simulated_cost_info: dict
        self.predicted_objective: float
        self.simulated_cost_info: float
        self.id: int
        self.nr_surgeries = nr_surgeries
        self.adjusted_day: Day

    def add_id(self, id: int):
        self.id = id

    def add_day(self, day: Day):
        self.days[day.name] = day

    def set_simulated_objective(self, simulated_objective: float):
        self.simulated_objective = simulated_objective

    def set_simulated_cost_info(self, simulated_info: dict):
        self.simulated_cost_info = simulated_info

    def set_predicted_objective(self, predicted_objective: float):
        self.predicted_objective = predicted_objective

    def set_adjusted_day(self, day: Day):
        self.adjusted_day = day

    def get_adjusted_day(self):
        return self.adjusted_day

    def equals(self, comparison_solution: 'Solution'):
        for day_name in self.days.keys():
            if not self.days[day_name].equals(comparison_solution.days[day_name]):
                return False

        return True

    def get_number_of_scheduled_surgeries(self):
        number_of_scheduled_surgeries = 0
        for name, day in self.days.items():
            number_of_scheduled_surgeries += day.get_number_of_scheduled_surgeries()
        return number_of_scheduled_surgeries

    def count_surgeries(self):
        self.nr_surgeries = {"CARD": 0, "GASTRO": 0, "GYN": 0, "MED": 0, "ORTH": 0, "URO": 0}
        for day in self.days:
            for block in self.days[day].blocks:
                # First add neighbor with one extra surgery in block
                specialty = self.days[day].blocks[block].block_type.name
                nr = len(self.days[day].blocks[block].start_times_surgeries)
                self.nr_surgeries[specialty] += nr

    def save_solution(self, solution_location):
        # Save the scenario pool
        result = []
        i = 0
        for day in self.days:
            for block in self.days[day].blocks:
                for surgery in self.days[day].blocks[block].start_times_surgeries:
                    result.append({
                        "Day": day,
                        "Block": block,
                        "Surgery": i,
                        "Start": surgery,
                        "Duration": 0,
                        "Finish": surgery + 0,
                        "Specialty": self.days[day].blocks[block].block_type.name,
                    })
                    i = i + 1
        result = pd.DataFrame(result)
        result.to_csv(solution_location, index=False, sep=",")

    def update_simulated_objective_based_on_days(self, not_scheduling_costs):
        total_objective = not_scheduling_costs
        for day_name, day in self.days.items():
            total_objective += day.get_simulated_objective()

        self.simulated_objective = total_objective

        return self.simulated_objective

    def reset_day_objectives(self):
        for day_name, day in self.days.items():
            day.reset_objective()


