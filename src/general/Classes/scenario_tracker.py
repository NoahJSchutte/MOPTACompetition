from src.general.Classes.scenario_pool import ScenarioPool

from typing import Dict, List

import numpy as np


class ScenarioTracker:
    def __init__(
            self,
            scenario_pool: ScenarioPool,
            base_solution_len=10000
    ):
        self.main_scenario_pool = scenario_pool  # this pool should never be adjusted, as it hold all scenarios
        self.main_pool_size = scenario_pool.get_size()

        self.evaluation_dict_per_scenario: Dict[int, Dict[int, float]] = dict() # {scenario.id: {solution.id: objective}}
        for scenario in scenario_pool.scenarios:
            self.evaluation_dict_per_scenario[scenario.id] = dict()

        self.scenario_per_solution_matrix = np.zeros((base_solution_len, scenario_pool.get_size()))
        self.solution_ids = list()  # track the solution ids, for the scenarios the ids are equal to the indices
        self.base_solution_len = base_solution_len
        self.iteration_space = base_solution_len
        self.iteration = -1

    def new_iteration(self):
        self.iteration += 1

    def set_objective_matrix(self, scenario_id, iteration, objective):
        if not iteration < self.iteration_space:
            extra_rows = np.zeros((self.base_solution_len, self.main_scenario_pool.get_size()))
            self.scenario_per_solution_matrix = np.concatenate((self.scenario_per_solution_matrix, extra_rows))
        self.scenario_per_solution_matrix[iteration, scenario_id] = objective

    def get_test_row(self):
        return self.scenario_per_solution_matrix[self.iteration-1, :]

    def get_train_rows(self, number_of_rows: int):
        return self.scenario_per_solution_matrix[(self.iteration - number_of_rows):self.iteration, :]

    #def set_objective(self, scenario_id, solution_id, objective):
    #    self.evaluation_dict_per_scenario[scenario_id][solution_id] = objective

    def set_objective(self, scenario_id, solution_id, objective):
        self.solution_ids.append(solution_id)
        self.scenario_per_solution_matrix[self.iteration, scenario_id] = objective

    def get_scenario_objectives(self, scenario_id):
        return self.evaluation_dict_per_scenario[scenario_id]

    def get_solution_objectives(self, solution_id):
        scenario_dict = dict()
        for scenario_id, solution_dict in self.evaluation_dict_per_scenario:
            if solution_id in solution_dict:
                scenario_dict[scenario_id] = solution_dict[solution_id]

        return scenario_dict

    def get_solution_per_scenario_matrix(self):
        # Note that this function only works when each solution is evaluated over the same amount of scenarios
        number_of_scenarios = len(self.evaluation_dict_per_scenario)
        key = list(self.evaluation_dict_per_scenario.keys())[0]
        number_of_solutions = len(self.evaluation_dict_per_scenario[key])
        matrix = np.zeros((number_of_solutions, number_of_scenarios))
        for scenario_id, solution_dict in self.evaluation_dict_per_scenario.items():
            for solution_id, objective in solution_dict.items():
                matrix[solution_id, scenario_id] = objective

        return matrix

    def get_number_of_evaluations_per_iteration(self):
        return (self.scenario_per_solution_matrix > 0).sum(axis=1)[:self.iteration]

    def get_number_of_evaluations(self):
        return (self.scenario_per_solution_matrix > 0).sum()




