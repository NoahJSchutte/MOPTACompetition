from src.general.Functions.solution_functions import convert_solution
from src.general.Functions.general_functions import save_object
from src.general.Classes.solution import Solution
from src.general.Classes.predictor import Predictor
from src.general.Classes.simulator import Simulator
from src.general.Classes.instance_info import InstanceInfo
from src.general.Classes.scenario_tracker import ScenarioTracker
from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Classes.evaluator import Evaluator
from src.general.Functions.mutation import remove_one_random, add_one_random, only_spread

import copy
import time
import random
import numpy as np
import pandas as pd


class ALNS:
    def __init__(self,
                 final_temperature=10,
                 h=1.1,
                 p=0.5,
                 alpha=0.5,
                 iterations=10,
                 segments=100,
                 track_solutions=False,
                 single_day_updates=False,
                 print_progress=True,
                 print_detailed=False,
                 seed = 0
    ):
        self.instance_info: InstanceInfo
        self.predictor: Predictor
        self.simulator: Simulator
        self.scenario_pool: ScenarioPool = None
        self.scenario_tracker: ScenarioTracker
        self.evaluator: Evaluator = None
        self.force_accept = False
        self.single_day_updates = single_day_updates

        self.candidate: Solution
        self.initial_solution: Solution
        self.current_solution: Solution
        self.best_solution: Solution
        self.current_solution_objective = float('inf')
        self.best_solution_objective = float('inf')
        self.nr_accepted = 0
        self.best_found_iteration = 0
        self.current_iteration = 0
        self.budget_left = True

        self.print_progress = print_progress
        self.print_detailed = print_detailed

        # Hyper parameters
        # Initially we accept a candidate with f(x') = h * obj_init    with probability p
        self.h = h
        self.p = p
        self.alpha = alpha
        self.initial_temperature = float('inf')
        self.final_temperature = final_temperature
        self.iterations = iterations
        self.segments = segments

        self.seed = seed
        # Heuristics
        self.nr_heuristics = {"add_remove": 3, "spread": 5}
        self.heuristics_names = {"add_remove": ["same_nr", "add", "remove"],
                                 "spread": ["spread_increasing", "slack_begin", "slack_end", "shift_left", "shift_right"]}
        self.heuristic_values = {"add_remove": [i for i in range(self.nr_heuristics["add_remove"])],
                                 "spread": [i for i in range(self.nr_heuristics["spread"])]}
        self.heuristic_probs = {"add_remove": [1/self.nr_heuristics["add_remove"] for i in range(self.nr_heuristics["add_remove"])],
                                 "spread": [1/self.nr_heuristics["spread"] for i in range(self.nr_heuristics["spread"])]}
        self.heuristic_scores = {"add_remove": [0 for i in range(self.nr_heuristics["add_remove"])],
                                 "spread": [0 for i in range(self.nr_heuristics["spread"])]}
        self.heuristic_attempts = {"add_remove": [0 for i in range(self.nr_heuristics["add_remove"])],
                                 "spread": [0 for i in range(self.nr_heuristics["spread"])]}

        # Tracking stuff
        self.track_solutions = track_solutions
        self.validation_pool: ScenarioPool = None
        self.objectives_val_x = list()
        self.objectives_val_y = list()
        self.objectives_x = list()      # x-axis, iteration number
        self.objectives_y = list()
        self.acceptance_x = list()
        self.acceptance_y = list()
        self.best_x = list()
        self.best_y = list()

    def initialize(self, initial_solution, instance_info: InstanceInfo,  random_seed=5):
        # Initialize current solution
        self.seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.instance_info = instance_info
        self.initial_solution = convert_solution(initial_solution, instance_info.block_types)
        self.candidate = copy.deepcopy(self.initial_solution)
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.current_solution.add_id(0)
        self.best_solution = self.current_solution
        self.best_solution_objective = self.evaluate(self.best_solution, initialization=True)
        if self.track_solutions:
            self.best_x.append(self.current_iteration)
            self.best_y.append(self.best_solution_objective)
        self.current_solution_objective = self.best_solution_objective
        self.initial_temperature = self.best_solution_objective * (1-self.h)/np.log(self.p)
        self.temperature = self.initial_temperature
        self.final_temperature = 1
        if self.print_progress:
            print("Initial temperature is", self.initial_temperature)
            print("Initial solution is ", self.best_solution_objective)

    def heuristics(self, mutator, i):

        heuristic = self.heuristics_names["spread"][i]

        if mutator == "same_nr":
            self.candidate = only_spread(self.current_solution, heuristic)
            # TODO: Candidate should inherit daily costs from current solution
        elif mutator == "remove":
            self.candidate = remove_one_random(self.current_solution, spread="spread_equally", slack_begin=False,
                                         slack_end=False)
        elif mutator == "add":
            self.candidate = add_one_random(self.current_solution, instance_info=self.instance_info, spread="spread_equally", slack_begin=False,
                                               slack_end=False)


    def add_scenario_pool(self, scenario_pool: ScenarioPool):
        self.scenario_pool = scenario_pool

    def add_validation_pool(self, validation_pool: ScenarioPool):
        self.validation_pool = validation_pool

    def add_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.evaluator.set_total_number_iterations(self.iterations*self.segments)

    def add_simulator(self, simulator: Simulator):
        self.simulator = simulator

    def evaluate(self, solution, initialization=False):
        """
        Evaluate candidate with simulator
        """
        if self.evaluator:
            objective, force_accept = self.evaluator.evaluate(solution, self.simulator, self.best_solution_objective)
            self.force_accept = force_accept  # gets stuck with no force
        elif self.single_day_updates and not initialization:
            objective = self.simulator.evaluate_adjusted_day(solution, self.scenario_pool)
        else:
            objective = self.simulator.evaluate(solution, self.scenario_pool, return_information=False)
 
        if self.track_solutions:
            self.objectives_x.append(self.current_iteration)
            self.objectives_y.append(objective)
            if self.validation_pool:
                val_objective = self.simulator.evaluate(solution, self.validation_pool, return_information=False)
                self.objectives_val_x.append(self.current_iteration)
                self.objectives_val_y.append(val_objective)
        return objective

    def accept(self, mut, mut_transl, spread, spread_transl):
        candidate_objective = self.evaluate(self.candidate)
        if self.print_progress:
            print(f"at iteration {self.current_iteration}, the new candidate objective is {candidate_objective}")
        if (random.uniform(0, 1) <= np.exp(np.divide((self.current_solution_objective-candidate_objective),
                                                     self.temperature))) or self.force_accept:
            self.current_solution = self.candidate
            self.current_solution_objective = candidate_objective
            if self.print_detailed:
                print(f'solution of iteration {self.current_iteration}  accepted,  mutator type was {mut_transl} heuristic was {spread_transl}')
            self.nr_accepted += 1
            self.heuristic_scores["add_remove"][mut] += 1
            self.heuristic_scores["spread"][spread] += 1
            if self.track_solutions:
                self.acceptance_x.append(self.current_iteration)
                self.acceptance_y.append(self.current_solution_objective)

        if candidate_objective < self.best_solution_objective:
            self.best_solution = self.candidate
            self.best_found_iteration = self.current_iteration
            self.best_solution_objective = candidate_objective
            if self.print_progress:
                print("solution of iteration ", self.current_iteration, " is best so far")
            self.heuristic_scores["add_remove"][mut] += 2
            self.heuristic_scores["spread"][spread] += 2
            if self.track_solutions:
                self.best_x.append(self.current_iteration)
                self.best_y.append(self.best_solution_objective)

    def update_probabilities(self):
        # update probabilities
        for heuristic_type in ["add_remove", "spread"]:
            new_probs = []
            for i in range(self.nr_heuristics[heuristic_type]):
                if self.heuristic_attempts[heuristic_type][i] == 0:
                    new_probs.append(self.heuristic_probs[heuristic_type][i])
                else:
                    new_probs.append(self.heuristic_probs[heuristic_type][i] * (1 - self.alpha) + self.alpha * self.heuristic_scores[heuristic_type][i]/self.heuristic_attempts[heuristic_type][i])
            self.heuristic_probs[heuristic_type] = [float(i) / sum(new_probs) for i in new_probs]

    def search(self, csv_output):
        now = time.time()
        self.current_iteration = 1
        j = 0
        while (j < self.segments) and self.budget_left:
            k = 0
            self.heuristic_scores = {"add_remove": [0 for i in range(self.nr_heuristics["add_remove"])],
                                     "spread": [0 for i in range(self.nr_heuristics["spread"])]}
            self.heuristic_attempts = {"add_remove": [0 for i in range(self.nr_heuristics["add_remove"])],
                                       "spread": [0 for i in range(self.nr_heuristics["spread"])]}

            while (k < self.iterations) and self.budget_left:

                # Select
                mut = np.random.choice(self.heuristic_values["add_remove"], 1, p=self.heuristic_probs["add_remove"])[0]
                spread = np.random.choice(self.heuristic_values["spread"], 1, p=self.heuristic_probs["spread"])[0]

                # Correct if we can't add currently
                self.current_solution.count_surgeries()
                if self.heuristics_names["add_remove"][mut] == "add":
                    if sum(self.current_solution.nr_surgeries.values()) == self.instance_info.instance_size:
                        mut = 0
                        if self.print_detailed:
                            print("WARNING: solution is full, can not add")

                elif self.heuristics_names["add_remove"][mut] == "remove" or self.heuristics_names["add_remove"][mut] == "same_nr":
                    if sum(self.current_solution.nr_surgeries.values()) == 0:
                        mut = 1
                        if self.print_detailed:
                            print("WARNING: solution is empty, can not remove or have same nr")

                # Translate mutator
                mut_transl = self.heuristics_names["add_remove"][mut]
                spread_transl = self.heuristics_names["spread"][spread]

                # Count attempts
                self.heuristic_attempts["spread"][spread] += 1
                self.heuristic_attempts["add_remove"][mut] += 1

                self.heuristics(mut_transl, spread)
                self.candidate.count_surgeries()
                self.candidate.add_id(self.current_iteration+1)
                self.accept(mut, mut_transl, spread, spread_transl)
                self.temperature = self.temperature * np.divide(self.final_temperature, self.initial_temperature) \
                                   **(1/(self.iterations*self.segments))
                if self.print_detailed:
                    print(f'Temp: {self.temperature}')
                k += 1
                self.current_iteration += 1
                if self.evaluator:
                    self.budget_left = self.evaluator.has_budget_left()

            self.update_probabilities()
            if self.print_detailed:
                print(self.heuristic_probs)
            j += 1
        new_now = time.time()

        if self.print_progress:
            print("final solution is, ", self.best_solution_objective)
            print("initial assignment is ", self.initial_solution.nr_surgeries)
            print("final solution assignment ", self.best_solution.nr_surgeries)
        self.best_solution.save_solution(csv_output)

        resultAttributes = {
            "h": self.h,
            "InitialTemperature": self.initial_temperature,
            "FinalTemperature": self.final_temperature,
            "NrIterations": self.iterations * self.segments,
            "AcceptedCandidates": self.nr_accepted,
            "BestSolutionIteration": self.best_found_iteration,
            "SolutionObjective": self.best_solution_objective,
            "SolveTime": new_now - now,
            "TimeOut": False
        }

        attr_output = csv_output.split(".csv", 1)[0] + "attr.csv"
        attributes = pd.DataFrame([resultAttributes])
        attributes.to_csv(attr_output, index=False)

    def save_tracked_stuff(self, initial_solution='', directory='TrackingResults', policy='exponential',
                           emergency_rate=4, costs_id=1, addition=''):
        save_dict = dict()
        save_dict['objective x'] = self.objectives_x
        save_dict['objective y'] = self.objectives_y
        save_dict['acceptance x'] = self.acceptance_x
        save_dict['acceptance y'] = self.acceptance_y
        save_dict['best x'] = self.best_x
        save_dict['best y'] = self.best_y
        save_dict['validation x'] = self.objectives_val_x
        save_dict['validation y'] = self.objectives_val_y
        if self.scenario_pool:
            main_pool_size = self.scenario_pool.get_size()
        else:
            main_pool_size = self.evaluator.main_pool_size
        file_name = f'n={self.instance_info.instance_size}_e={emergency_rate}_c={costs_id}_' \
                    f'i={int(self.iterations*self.segments)}_' \
                    f'h={self.h}_a={self.alpha}_s={main_pool_size}_p={policy}_init={initial_solution}_seed={self.seed}' \
                    f'{addition}'

        if self.evaluator:
            save_dict['evaluator'] = self.evaluator.get_tracked_stuff()
            file_name = f'{file_name}_{self.evaluator.get_setting_string()}'

        save_object(save_dict, directory, file_name)







