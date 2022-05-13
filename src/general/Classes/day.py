from typing import Dict
from src.general.Classes.block import Block


class Day:
    def __init__(
            self,
            name: str,
            start_time: int,
            end_time: int
    ):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.simulated_objective = None
        self.number_of_scenarios = 0
        self.blocks = {}  # Dict[str: Block]  # dict so we can compare equal solutions

    def add_block(self, block: Block):
        self.blocks[block.key] = block

    def equals(self, comparison_day: 'Day'):
        for block_key in self.blocks.keys():
            if not self.blocks[block_key].equals(comparison_day.blocks[block_key]):
                return False

        return True

    def get_number_of_scheduled_surgeries(self):
        number_of_scheduled_surgeries = 0
        for name, block in self.blocks.items():
            number_of_scheduled_surgeries += block.get_number_of_surgeries()
        return number_of_scheduled_surgeries

    def set_simulated_objective(self, objective):
        self.simulated_objective = objective

    def get_simulated_objective(self):
        return self.simulated_objective

    def reset_objective(self):
        self.simulated_objective = 0
        self.number_of_scenarios = 0

    def adjust_simulated_objective(self, scenario_objective):
        if not self.simulated_objective:
            self.simulated_objective = scenario_objective
        else:
            self.simulated_objective = ((self.simulated_objective * self.number_of_scenarios) + scenario_objective) / (self.number_of_scenarios + 1)
        self.number_of_scenarios += 1


