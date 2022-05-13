from src.general.Classes.simulator import Simulator
from src.general.Classes.scenario_tracker import ScenarioTracker
from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Classes.solution import Solution

from typing import List
import numpy as np
import random
from gurobipy import *


class Evaluator:
    # The evaluator is used to more smartly evaluate during a search. It uses the scenario tracker for this
    def __init__(
            self,
            simulator: Simulator,
            main_scenario_pool: ScenarioPool,
            method='basic',
            budget=10000000
    ):
        self.scenario_tracker = ScenarioTracker(main_scenario_pool)

        self.main_scenario_pool = main_scenario_pool
        self.main_pool_size = main_scenario_pool.get_size()
        self.estimate_scenario_pool: ScenarioPool
        self.test_scenario_pool: ScenarioPool
        self.estimate_scenario_pools: List[ScenarioPool] = list()
        self.clustering_method = method   # None, 'basic', 'intermediate', 'advanced'
        self.total_iterations: int
        self.budget = budget

        # Below are settings for multiple cluster methods
        self.state = 'train'
        self.train_iteration = 0
        self.train_threshold = 5
        self.estimate_pool_size = 40  # TODO: enforce that this is a divider of main pool size
        # Below are settings for only the basic cluster method
        self.apply_iteration = 0
        self.apply_threshold = 50
        self.test_accuracy_threshold = 0.02
        # Below for only intermediate method
        self.discrepancy_threshold = 0.01
        self.test_pool_size = int(self.estimate_pool_size / 2)
        # Below for only advanced method
        self.do_representativeness_updates = True
        self.representativeness_threshold_start = 0.1
        self.representativeness_threshold_end = 0.01
        self.representativeness_threshold = self.representativeness_threshold_start
        self.minimum_pool_size = 4  # note that this should be 0 when using flexible number of pools in mip approach
        #self.pool_combination_size = 30
        self.pool_number_threshold = 3
        # Below things to track
        self.number_of_train_procedures = 1

        # Advanced plus approach
        self.flexible_number_of_sets = False
        self.number_of_pools_mip = 2
        #self.minimum_pool_size = 10
        self.maximum_pool_size = 50
        self.use_discrepancy_in_objective = False
        self.time_limit = 5
        self.mip_model = Model('PoolSelection')

        self.best_objective = None
        self.wrong_best_objective_counter = 0
        self.use_force_accept = False
        self.representativeness_ratio = 1


    def get_number_of_evaluations_per_iteration(self):
        return self.scenario_tracker.get_number_of_evaluations_per_iteration()

    def has_budget_left(self):
        return (self.budget > self.scenario_tracker.get_number_of_evaluations())

    def set_total_number_iterations(self, total_iterations: int):
        self.total_iterations = total_iterations

    def set_representativeness_parameters(self, do_representativeness_updates, start_representativeness_threshold,
                                          end_representativeness_threshold):
        self.do_representativeness_updates = do_representativeness_updates
        self.representativeness_threshold_start = start_representativeness_threshold
        self.representativeness_threshold_end = end_representativeness_threshold
        self.representativeness_threshold = self.representativeness_threshold_start

    def set_advanced_mip_parameters(self, flexible_number_of_sets: bool, number_of_pools: int,
                                    minimum_pool_size: int = 0, maximum_pool_size: int = 50, time_limit: int = 5,
                                    use_discrepancy_in_objective: bool = False, use_force_accept: bool = False,
                                    representativeness_ratio: int = 2, train_threshold: int = 10):
        self.flexible_number_of_sets = flexible_number_of_sets
        self.number_of_pools_mip = number_of_pools
        self.minimum_pool_size = minimum_pool_size
        self.time_limit = time_limit
        self.use_discrepancy_in_objective = use_discrepancy_in_objective
        self.maximum_pool_size = maximum_pool_size
        self.use_force_accept = use_force_accept
        self.representativeness_ratio = representativeness_ratio
        self.train_threshold = train_threshold

    def set_train_threshold(self, train_threshold):
        self.train_threshold = train_threshold

    def update_representativeness_threshold(self):
        self.representativeness_threshold = self.representativeness_threshold * \
                                            (self.representativeness_threshold_end /
                                             self.representativeness_threshold_start) ** (1/self.total_iterations)

    def evaluate(self, solution: Solution, simulator: Simulator, best_objective: float):
        self.best_objective = best_objective
        self.scenario_tracker.new_iteration()
        if self.do_representativeness_updates:
            self.update_representativeness_threshold()
        if self.clustering_method == 'basic':
            scenario_pool = self.cluster_basic()
            solution_objective = simulator.evaluate(solution, scenario_pool, scenario_tracker=self.scenario_tracker)
        elif self.clustering_method == 'intermediate':
            solution_objective = self.cluster_intermediate(solution, simulator)
        elif self.clustering_method == 'advanced' or self.clustering_method == 'advanced_mip':
            solution_objective = self.cluster_advanced(solution, simulator)
        else:
            solution_objective = None

        solution.set_simulated_objective(solution_objective)
        return solution_objective

    def cluster_basic(self):
        if self.state == 'train':
            if self.train_iteration < self.train_threshold:
                self.train_iteration += 1
                return self.main_scenario_pool
            else:
                self.estimate_scenario_pool = self.do_train_procedure_basic(self.estimate_pool_size)
                self.train_iteration = 0
                self.state = 'apply'
                self.apply_iteration += 1
                return self.estimate_scenario_pool
        elif self.state == 'apply':  # apply state
            if self.apply_iteration < self.apply_threshold:
                self.apply_iteration += 1
                return self.estimate_scenario_pool
            else:
                self.apply_iteration = 0
                self.state = 'test'
                return self.main_scenario_pool
        elif self.state == 'test':
            test_succeeded = self.do_test_procedure()
            if test_succeeded:
                self.state = 'apply'
                self.apply_iteration += 1
                return self.estimate_scenario_pool
            else:
                self.number_of_train_procedures += 1
                self.state = 'train'
                self.train_iteration += 1  # the test row can be used for training as it is a full row
                return self.main_scenario_pool

    def cluster_intermediate(self, solution: Solution, simulator: Simulator):
        if self.state == 'train':
            if self.train_iteration < self.train_threshold:
                self.train_iteration += 1
                return simulator.evaluate(solution, self.main_scenario_pool, scenario_tracker=self.scenario_tracker)
            else:
                self.do_train_procedure_intermediate()
                self.train_iteration = 0
                self.state = 'apply'
                return self.cluster_intermediate(solution, simulator)
        elif self.state == 'apply':
            # first test
            objective_test = simulator.evaluate(solution, self.test_scenario_pool, scenario_tracker=self.scenario_tracker)
            objective_estimate = simulator.evaluate(solution, self.estimate_scenario_pool, scenario_tracker=self.scenario_tracker)
            discrepancy = self.discrepancy([objective_test, objective_estimate])
            if discrepancy > self.discrepancy_threshold:
                print(f'With a discrepancy of {discrepancy}, the discrepancy threshold was violated')
                self.number_of_train_procedures += 1
                self.state = 'train'
                self.train_iteration += 1
                union_test_estimate = self.estimate_scenario_pool.union(self.test_scenario_pool)
                remainder_pool = self.main_scenario_pool.difference(union_test_estimate)
                objective_remainder = simulator.evaluate(solution, remainder_pool, scenario_tracker=self.scenario_tracker)
                average_objective = (objective_remainder * remainder_pool.get_size() +
                                     objective_test * self.test_scenario_pool.get_size() +
                                     objective_estimate * self.estimate_scenario_pool.get_size()) / self.main_pool_size
                return average_objective
            else:
                return objective_estimate

    def cluster_advanced(self, solution: Solution, simulator: Simulator):
        force_accept = False
        if self.state == 'train':
            if self.train_iteration < self.train_threshold:
                self.train_iteration += 1
                return simulator.evaluate(solution, self.main_scenario_pool, scenario_tracker=self.scenario_tracker), \
                    force_accept
            else:
                self.do_train_procedure_advanced()
                self.train_iteration = 0
                self.wrong_best_objective_counter = 0
                self.state = 'apply'
                return self.cluster_advanced(solution, simulator)
        elif self.state == 'apply':
            # first test
            pool_1, pool_2 = self.pick_two_pools()
            objective_1 = simulator.evaluate(solution, pool_1, scenario_tracker=self.scenario_tracker)
            objective_2 = simulator.evaluate(solution, pool_2, scenario_tracker=self.scenario_tracker)
            discrepancy = 2*abs(objective_1 - objective_2) / (objective_1 + objective_2)
            if discrepancy > self.representativeness_threshold:
                force_accept = True if self.use_force_accept else False
                print(f'With a discrepancy of {discrepancy}, the representativeness threshold was violated')
                self.number_of_train_procedures += 1
                self.state = 'train'
                self.train_iteration += 1
                if pool_1.get_size() + pool_2.get_size() < self.main_pool_size:
                    union_pools = pool_1.union(pool_2)
                    remainder_pool = self.main_scenario_pool.difference(union_pools)
                    objective_remainder = simulator.evaluate(solution, remainder_pool,
                                                             scenario_tracker=self.scenario_tracker)
                    average_objective = (objective_remainder * remainder_pool.get_size() +
                                         objective_1 * pool_1.get_size() +
                                         objective_2 * pool_2.get_size()) / self.main_pool_size
                    return average_objective, force_accept
                else:
                    average_objective = (objective_1 * pool_1.get_size() +
                                         objective_2 * pool_2.get_size()) / self.main_pool_size
                    return average_objective, force_accept
            else:
                force_accept_best = False
                average_objective = (objective_1 + objective_2) / 2
                if force_accept_best:
                    if average_objective < self.best_objective:
                        self.wrong_best_objective_counter += 1
                        print('Evaluator: Best objective found')
                        #force_accept = True
                        union_pools = pool_1.union(pool_2)
                        remainder_pool = self.main_scenario_pool.difference(union_pools)
                        objective_remainder = simulator.evaluate(solution, remainder_pool,
                                                                 scenario_tracker=self.scenario_tracker)
                        average_objective = (objective_remainder * remainder_pool.get_size() +
                                             objective_1 * pool_1.get_size() +
                                             objective_2 * pool_2.get_size()) / self.main_pool_size
                        if self.wrong_best_objective_counter == self.train_threshold:
                            self.state = 'train'
                            self.train_iteration = self.train_threshold
                    else:
                        self.wrong_best_objective_counter = 0

                return average_objective, force_accept

    def pick_two_pools(self):
        i, j = random.sample(range(0, len(self.estimate_scenario_pools)), 2)
        return self.estimate_scenario_pools[i], self.estimate_scenario_pools[j]

    def do_train_procedure_intermediate(self):
        self.estimate_scenario_pool = self.do_train_procedure_basic(self.estimate_pool_size)
        self.test_scenario_pool = self.do_train_procedure_basic(self.test_pool_size)

    def do_train_procedure_advanced(self):
        if self.clustering_method == 'advanced':
            self.estimate_scenario_pools = list()
            forbidden_indices = list()
            new_pool_possible = True
            next_pool_size = self.minimum_pool_size

            while new_pool_possible and (2*next_pool_size <= self.main_pool_size) and \
                    (len(self.estimate_scenario_pools) < self.pool_number_threshold):
                new_pool, forbidden_indices = self.do_train_procedure_sequentially(next_pool_size, forbidden_indices)
                if new_pool:
                    self.estimate_scenario_pools.append(new_pool)
                    next_pool_size = 2*next_pool_size
                else:
                    new_pool_possible = False

            print(f'{len(self.estimate_scenario_pools)} where found')

        else: # self.clustering_method == 'advanced mip'
            self.do_train_procedure_mip()

    def do_train_procedure_mip(self):
        rho = self.representativeness_threshold / self.representativeness_ratio
        self.mip_model.params.timeLimit = self.time_limit

        # Sets
        POOLS = set([i for i in range(self.number_of_pools_mip)])
        SCENARIOS = set([i for i in range(self.main_pool_size)])
        SOLUTIONS = set([i for i in range(self.train_threshold)])
        SCENARIO_SOLUTIONS = set([(i, j) for i in SOLUTIONS for j in SCENARIOS])
        SCENARIO_POOLS = set([(i, j) for i in POOLS for j in SCENARIOS])
        SOLUTION_POOLS = set([(i, j) for i in POOLS for j in SOLUTIONS])

        # Variables
        x = self.mip_model.addVars(SCENARIO_POOLS, vtype=GRB.BINARY, name='x')

        # Parameters
        m = self.scenario_tracker.get_train_rows(self.train_threshold) # note this is a numpy array
        m_per_solution = m.mean(axis=1)

        # Constraints
        if not self.use_discrepancy_in_objective:
            self.mip_model.addConstrs(
                (m_per_solution[j]*quicksum(x[k, i] for i in SCENARIOS)*(1+rho) >=
                quicksum(x[k, i]*m[j, i] for i in SCENARIOS)
                for j in SOLUTIONS for k in POOLS), 'deviation positive')
        self.mip_model.addConstrs(
            (m_per_solution[j]*quicksum(x[k, i] for i in SCENARIOS)*(1) <=  # (1-rho)
            quicksum(x[k, i]*m[j, i] for i in SCENARIOS)
            for j in SOLUTIONS for k in POOLS), 'deviation negative')
        self.mip_model.addConstrs((quicksum(x[k, i] for k in POOLS) <= 1 for i in SCENARIOS), 'disjoint sets')
        self.mip_model.addConstrs(((quicksum(x[k, i] for i in SCENARIOS) <= self.maximum_pool_size)
                                   for k in POOLS),
                                  'maximum size')

        if self.use_discrepancy_in_objective:
            total_discrepancy = self.mip_model.addVar(vtype=GRB.CONTINUOUS, name='discrepancy total')
            discrepancy = self.mip_model.addVars(SOLUTION_POOLS, vtype=GRB.CONTINUOUS, name='discrepancy')
            discrepancy_signed = self.mip_model.addVars(SOLUTION_POOLS, vtype=GRB.CONTINUOUS, name='discrepancy')
            # scenarios_per_pool = quicksum(x[k, i] for i in SCENARIOS)
            discrepancy_weight = 10000
            self.mip_model.addConstrs(
                ((discrepancy[k, j] == quicksum(x[k, i]*m[j, i] for i in SCENARIOS) -
                  m_per_solution[j]*quicksum(x[k, i] for i in SCENARIOS)
                  ) for j in SOLUTIONS for k in POOLS),
                name='discrepancy signed')
            #self.mip_model.addConstrs(((discrepancy[k, j] == abs_(discrepancy_signed[k, j]))
            #                           for k in POOLS for j in SOLUTIONS), name='discrepancy abs')
            #self.mip_model.addGenConstrAbs()
            self.mip_model.addConstr(total_discrepancy == quicksum(discrepancy[k, j]/
                                                                   (m_per_solution[j]*self.train_threshold*self.main_pool_size) for k in POOLS for j in SOLUTIONS),
                                     name='discrepancy total')

        # Objective
        if self.flexible_number_of_sets:
            pool_nr = self.mip_model.addVar(vtype=GRB.INTEGER, name='pool_nr')
            pool_exists = self.mip_model.addVars(POOLS, vtype=GRB.BINARY, name='pool_exists')
            pool_weight = 10
            self.mip_model.addConstrs((pool_exists[k] == max_([x[k, i] for i in SCENARIOS], constant=0))
                                      for k in POOLS)
            self.mip_model.addConstr(pool_nr <= quicksum(pool_exists[k] for k in POOLS))
            if self.minimum_pool_size > 0 and False:
                self.mip_model.addConstrs(((quicksum(x[k, i] for i in SCENARIOS) >= self.minimum_pool_size*pool_exists[k])
                                           for k in POOLS),
                                          'minimum size')
            #At least 2 pools
            pools_guarantee = False
            if pools_guarantee:
                j = 0
                for pool in POOLS:
                    if j == 0:
                        pool_1 = pool
                    elif j == 1:
                        pool_2 = pool
                    j += 1

                self.mip_model.addConstr((quicksum(x[pool_1, i] for i in SCENARIOS) >= 1), 'pool 1')
                self.mip_model.addConstr((quicksum(x[pool_2, i] for i in SCENARIOS) >= 1), 'pool 2')



            if self.use_discrepancy_in_objective:
                self.mip_model.setObjective(quicksum(x[k, i] for i in SCENARIOS for k in POOLS)
                                            - pool_weight*pool_nr
                                            + total_discrepancy * discrepancy_weight, GRB.MINIMIZE)
            else:
                self.mip_model.setObjective(-quicksum(x[k, i] for i in SCENARIOS for k in POOLS), GRB.MINIMIZE)

            # Set starting solution
            if False:
                for i in SCENARIOS:
                    for k in POOLS:
                        if k == 0:
                            x[0, i].Start = 1
                        else:
                            x[k, i].Start = 0

        else:
            self.mip_model.addConstrs((quicksum(x[k, i] for i in SCENARIOS) >= self.minimum_pool_size for k in POOLS),
                                      'minimum size')
            if self.use_discrepancy_in_objective:
                self.mip_model.setObjective(-quicksum(x[k, i] for i in SCENARIOS for k in POOLS) +
                                            total_discrepancy * discrepancy_weight, GRB.MINIMIZE)
            else:
                self.mip_model.setObjective(quicksum(x[k, i] for i in SCENARIOS for k in POOLS), GRB.MINIMIZE)

        self.mip_model.optimize()

        #self.x = x
        #self.pool_nr = pool_nr
        #self.pool_exists = pool_exists

        scenarios_per_pool = self.mip_model.getAttr('X', x)
        self.estimate_scenario_pools = list()
        number_of_used_scenarios = 0
        if self.flexible_number_of_sets:
            resulting_existing_pools = self.mip_model.getAttr('X', pool_exists) #dict
            resulting_number_of_pools = round(self.mip_model.getAttr('X', {pool_nr})[0]) #int
            map_id_to_num_pools = dict()
            k = 0
            for pool, resulting_pool_exists in resulting_existing_pools.items():
                if round(resulting_pool_exists) > 0:
                    map_id_to_num_pools[pool] = k
                    self.estimate_scenario_pools.append(ScenarioPool(self.main_scenario_pool.get_instance_info()))
                    k += 1
            for pool, scenario in scenarios_per_pool:
                if scenarios_per_pool[pool, scenario] > 0:
                    number_of_used_scenarios += 1
                    self.estimate_scenario_pools[map_id_to_num_pools[pool]].add_scenario(
                        self.main_scenario_pool.get_scenario(round(scenario)))
            print(f'Number of pools found: {resulting_number_of_pools}')
        else:
            for i in range(self.number_of_pools_mip):
                self.estimate_scenario_pools.append(ScenarioPool(self.main_scenario_pool.get_instance_info()))
            for pool, scenario in scenarios_per_pool:
                if scenarios_per_pool[pool, scenario] > 0:
                    number_of_used_scenarios += 1
                    self.estimate_scenario_pools[pool].add_scenario(self.main_scenario_pool.get_scenario(round(scenario)))
        print(f'Number of used scenarios: {number_of_used_scenarios}')
        if self.use_discrepancy_in_objective:
            print(f"Discrepancy: {self.mip_model.getAttr('X', {total_discrepancy})[0]}")
            #print(f"Absolute discrepancy: {self.mip_model.getAttr('X', discrepancy)}")
        #print(f'Scenarios per pool: {scenarios_per_pool}')
        # TODO: It seems like there is som error as the discrepancy is able to be exactly 0

    def do_train_procedure_basic(self, pool_size):
        train_rows = self.scenario_tracker.get_train_rows(self.train_threshold)
        train_mean = train_rows.mean(axis=0)
        group_size = int(self.main_pool_size / pool_size)
        new_pool = ScenarioPool(self.main_scenario_pool.get_instance_info())
        total_estimation_error = 0
        sorted_indices = np.argsort(train_mean)
        sorted_train_mean = train_mean[sorted_indices]

        for i in range(pool_size):
            group = sorted_train_mean[i*group_size:(i+1)*group_size]
            group_mean = group.mean()
            closest_scenario_index = sorted_indices[np.argmin(abs(group - group_mean)) + i*group_size]
            representative_scenario = self.main_scenario_pool.get_scenario(int(closest_scenario_index))
            new_pool.add_scenario(representative_scenario)
            total_estimation_error += (group_mean - train_mean[closest_scenario_index]) / pool_size

        accurate_objective = train_mean.mean()
        percentage_estimation_error = abs(total_estimation_error) / accurate_objective
        print(percentage_estimation_error)
        if percentage_estimation_error > self.test_accuracy_threshold:
            print(f'Error: The new selected pool of size {pool_size} already exceeds the accuracy threshold: '
                  f'{percentage_estimation_error} > {self.test_accuracy_threshold} \n'
                  f'It is still used for the next {self.apply_threshold} iterations')

        return new_pool

    def do_train_procedure_sequentially(self, pool_size, forbidden_indices):
        train_rows = self.scenario_tracker.get_train_rows(self.train_threshold)
        train_mean = train_rows.mean(axis=0)
        group_size = round(self.main_pool_size / pool_size)
        new_pool = ScenarioPool(self.main_scenario_pool.get_instance_info())
        total_estimation_error = 0
        sorted_indices = np.argsort(train_mean)
        sorted_train_mean = train_mean[sorted_indices]

        for i in range(pool_size):
            nth_smallest = 0
            group = sorted_train_mean[i*group_size:(i+1)*group_size]
            group_mean = group.mean()
            min_sorted_index = np.argpartition(abs(group - group_mean), nth_smallest)[nth_smallest]
            min_index_in_group = sorted_indices[i*group_size + min_sorted_index]
            while (min_index_in_group in forbidden_indices) and (nth_smallest < len(group)):
                nth_smallest += 1
                min_sorted_index = np.argpartition(abs(group - group_mean), nth_smallest)[nth_smallest]
                min_index_in_group = sorted_indices[i*group_size + min_sorted_index]
            if min_index_in_group in forbidden_indices:
                return False, None
            closest_scenario_index = int(min_index_in_group)
            forbidden_indices.append(closest_scenario_index)
            representative_scenario = self.main_scenario_pool.get_scenario(closest_scenario_index)
            new_pool.add_scenario(representative_scenario)
            total_estimation_error += (group_mean - train_mean[closest_scenario_index]) / pool_size

        accurate_objective = train_mean.mean()
        percentage_estimation_error = abs(total_estimation_error) / accurate_objective
        if percentage_estimation_error > self.representativeness_threshold:
            return False, None
            #print(f'Error: The new selected pool of size {pool_size} already exceeds the accuracy threshold: '
            #      f'{percentage_estimation_error} > {self.test_accuracy_threshold} \n'
            #      f'It is still used for the next {self.apply_threshold} iterations')

        return new_pool, forbidden_indices

    def do_test_procedure(self):
        get_test_row = self.scenario_tracker.get_test_row()
        scenario_ids_estimate = self.estimate_scenario_pool.get_scenario_ids()
        estimate_mean = get_test_row[scenario_ids_estimate].mean()
        actual_mean = get_test_row.mean()
        estimation_error = abs((estimate_mean - actual_mean) / actual_mean)
        test_succeeded = estimation_error < self.test_accuracy_threshold
        print(f'Success? {test_succeeded}, estimation error: {estimation_error}')

        return test_succeeded

    def get_tracked_stuff(self):
        tracked_dict = dict()
        tracked_dict['evaluations per iteration'] = self.get_number_of_evaluations_per_iteration()
        return tracked_dict

    def get_setting_string(self):
        return f'm={self.clustering_method}_tt={self.train_threshold}' \
               f'_sr={self.representativeness_threshold_start}_se{self.representativeness_threshold_end}' \
               f'_mp={self.minimum_pool_size}_pn={self.number_of_pools_mip}'

    @staticmethod
    def create_subset(scenario_pool, scenario_ids: List[int]):
        subset_pool = ScenarioPool(scenario_pool.instance_info)
        for scenario_id in scenario_ids:
            subset_pool.add_scenario(scenario_pool.get_scenario(scenario_id))

        return subset_pool

    @staticmethod
    def discrepancy(values: List[float]):
        value1 = values[0]
        value2 = values[1]
        return abs((value2 - value1) / value1)

