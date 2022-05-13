from src.general.Classes.solution import Solution
from src.general.Classes.day import Day
from src.general.Classes.sim_block import SimBlock
from src.general.Classes.sim_surgery import SimSurgery
from src.general.Classes.instance_info import InstanceInfo
from src.general.Classes.scenario import Scenario
from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Classes.scenario_tracker import ScenarioTracker
from src.general.Classes.policy_tracker import PolicyTracker
from typing import List,Dict

import numpy as np
import random


class Simulator:
    def __init__(
            self,
            instance_info: InstanceInfo
    ):
        self.instance_info = instance_info
        self.current_time: float = 0
        self.sim_blocks: Dict[str, List[SimBlock]] = dict()
        self.emergency_surgeries: List[SimSurgery] = list()
        self.policy = 'greedy'
        self.track_policy = False
        self.policy_tracker: PolicyTracker
        self.print_status = False

    def reset(self):
        self.current_time = 0
        self.sim_blocks = dict()
        self.emergency_surgeries = {}

    def set_policy(self, policy: str): # currently working: greedy and deterministic
        if policy == 'log-normal':
            self.policy = 'optimal'
        else:
            self.policy = policy

    def set_emergency_surgery_distributions(self, emergency_surgery_duration_distribution,
                                            emergency_surgery_duration_parameters,
                                            emergency_surgery_arrival_distribution,
                                            emergency_surgery_arrival_rate):
        self.instance_info.emergency_duration_distribution = emergency_surgery_duration_distribution
        self.instance_info.emergency_duration_parameters = emergency_surgery_duration_parameters
        self.instance_info.emergency_arrival_distribution = emergency_surgery_arrival_distribution
        self.instance_info.emergency_arrival_rate = emergency_surgery_arrival_rate

    def evaluate(self, solution: Solution, scenario_pool: ScenarioPool, print_status=False,
                 return_information=False, scenario_tracker: ScenarioTracker = None):
        sum_objectives = 0
        sum_wait, sum_wait_emergency, sum_over, sum_idle, sum_not_scheduled, sum_cancellation = 0, 0, 0, 0, 0, 0
        sum_wait_costs, sum_wait_emergency_costs, sum_over_costs, sum_idle_costs, sum_not_scheduled_costs, sum_cancellation_costs = 0, 0, 0, 0, 0, 0
        solution.reset_day_objectives()
        for scenario in scenario_pool.scenarios:
            objective = self.simulate_full_solution(solution, scenario, return_information=return_information)
            if scenario_tracker:
                scenario_tracker.set_objective(scenario.id, solution.id, objective)
            if return_information:
                objective, info, criteria = objective
                sum_wait_costs += info["waiting_costs"]
                sum_wait_emergency_costs += info["emergency_waiting_costs"]
                sum_over_costs += info["over_costs"]
                sum_idle_costs += info["idle_costs"]
                sum_not_scheduled_costs += info["not_scheduling_costs"]
                sum_cancellation_costs += info["cancellation_costs"]
                sum_wait += criteria["waiting"]
                sum_wait_emergency += criteria["emergency_waiting"]
                sum_over += criteria["over"]
                sum_idle += criteria["idle"]
                sum_not_scheduled += criteria["not_scheduling"]
                sum_cancellation += criteria["cancellation"]
            sum_objectives += objective

            if print_status:
                print(f'Scenario {scenario.id} has an objective of {objective}')
        average_objective = sum_objectives / scenario_pool.get_size()
        solution.set_simulated_objective(average_objective)  # when the evaluator is used this gets overwritten

        if return_information:
            info = {"waiting_costs": sum_wait_costs / scenario_pool.get_size(), "emergency_waiting_costs": sum_wait_emergency_costs / scenario_pool.get_size(),
                    "over_costs": sum_over_costs / scenario_pool.get_size(), "idle_costs": sum_idle_costs / scenario_pool.get_size(),
                    "cancellation_costs": sum_cancellation_costs / scenario_pool.get_size(), "not_scheduling_costs": sum_not_scheduled_costs / scenario_pool.get_size()}
            criteria = {"waiting": sum_wait / scenario_pool.get_size(),
                    "emergency_waiting": sum_wait_emergency / scenario_pool.get_size(),
                    "over": sum_over / scenario_pool.get_size(),
                    "idle": sum_idle / scenario_pool.get_size(),
                    "cancellation": sum_cancellation / scenario_pool.get_size(),
                    "not_scheduling": sum_not_scheduled / scenario_pool.get_size()}
            solution.set_simulated_cost_info(info)
            return average_objective, info, criteria
        else:
            return average_objective

    def evaluate_adjusted_day(self, solution: Solution, scenario_pool: ScenarioPool):
        self.reset()
        not_scheduling_costs = (self.instance_info.instance_size - solution.get_number_of_scheduled_surgeries()) * \
                               self.instance_info.not_scheduling_costs
        day = solution.get_adjusted_day()
        sum_objectives = 0
        for scenario in scenario_pool.scenarios:
            costs, info, criteria = self.simulate_day(day, scenario)
            sum_objectives += costs

        day_objective = sum_objectives / scenario_pool.get_size()
        day.set_simulated_objective(day_objective)

        return solution.update_simulated_objective_based_on_days(not_scheduling_costs)

    def evaluate_with_policy_tracking(self, solution: Solution, scenario_pool: ScenarioPool, print_status=False):
        self.print_status = print_status
        self.track_policy = True
        self.policy_tracker = PolicyTracker(self.policy)

        sum_objectives = 0
        pool_size = scenario_pool.get_size()
        scenario_objectives = np.zeros(pool_size)

        for num, scenario in enumerate(scenario_pool.scenarios):
            objective, info, criteria = self.simulate_full_solution(solution, scenario)
            scenario_objectives[num] = objective
            if print_status:
                print(f'Scenario {num} has an objective of {objective}')

        #average_objective = sum(scenario_objectives) / pool_size
        #solution.set_simulated_objective(average_objective)

        return scenario_objectives, self.policy_tracker

    def simulate_full_solution(self, solution: Solution, scenario: Scenario, return_information=True):
        self.reset()
        not_scheduling_costs = (self.instance_info.instance_size - solution.get_number_of_scheduled_surgeries()) * \
            self.instance_info.not_scheduling_costs
        objective = not_scheduling_costs

        if return_information:
            info = {"waiting_costs": 0, "emergency_waiting_costs": 0, "over_costs": 0, "idle_costs": 0, "cancellation_costs": 0,
                    "not_scheduling_costs": not_scheduling_costs}
            criteria = {"waiting": 0, "emergency_waiting": 0, "over": 0, "idle": 0, "cancellation": 0, "not_scheduling": not_scheduling_costs}
        for day_name, day in solution.days.items():
            day_costs, day_info, day_criteria = self.simulate_day(day, scenario)
            day.adjust_simulated_objective(day_costs)
            objective += day_costs
            if return_information:
                for cost in ["waiting_costs", "emergency_waiting_costs", "over_costs", "idle_costs", "cancellation_costs"]:
                    info[cost] += day_info[cost]
                for crit in ["waiting", "emergency_waiting", "over", "idle",  "cancellation"]:
                    criteria[crit] += day_criteria[crit]

        if return_information:
            return objective, info, criteria
        else:
            return objective

    def simulate_day(self, day: Day, scenario: Scenario):
        # create objects that are needed
        self.current_time = 0
        self.sim_blocks[day.name] = []    # empty sim_blocks
        for name, block in day.blocks.items():
            sim_block = SimBlock(block, self.instance_info)
            # TODO: Make it such that these do not reinitialize but restart
            self.sim_blocks[day.name].append(sim_block)

        for sim_block in self.sim_blocks[day.name]:
            sim_block.set_surgery_durations(scenario)
            sim_block.set_policy(self.policy)
            #sim_block.draw_surgery_durations()

        self.set_emergency_surgeries(scenario)
        #self.draw_emergency_surgeries(day)

        while self.emergency_surgeries[day.name]:  # or while not end day
            emergency_surgery = self.emergency_surgeries[day.name].pop(0)
            next_decision_point = max(emergency_surgery.get_scheduled_start_time(), self.current_time)
            self.current_time = next_decision_point
            self.apply_policy(day, emergency_surgery, next_decision_point)

        # When there are no surgeries left we still need to finish all surgeries to be able to het costs
        certain_finish_time = float('inf')
        for block in self.sim_blocks[day.name]:
            block.move_to_next_decision_point(certain_finish_time)

        costs, info, criteria = self.calculate_realized_costs(day)

        return costs, info, criteria

    def apply_policy(self, day: Day, emergency_surgery, next_decision_point):
        at_least_one_block_is_free = False
        for sim_block in self.sim_blocks[day.name]:
            sim_block.move_to_next_decision_point(next_decision_point)
            if not sim_block.has_ongoing_surgery:
                at_least_one_block_is_free = True

        # TODO: Issue with moving when there is another surgery coming in before a room is free
        # TODO: Adjust to move to next actual decision point
        if False: # not at_least_one_block_is_free:  # we wait till a room becomes free
            next_actual_decision_point = self.get_next_free_block_time(day)
            self.current_time = next_actual_decision_point
            for sim_block in self.sim_blocks[day.name]:
                sim_block.move_to_next_decision_point(next_actual_decision_point)

        if self.policy == 'greedy':
            free_blocks = list()
            for block in self.sim_blocks[day.name]:
                if not block.has_ongoing_surgery:
                    free_blocks.append(block)

            if free_blocks:
                if len(free_blocks) == 1:
                    best_block = free_blocks[0]
                else:
                    best_block = self.pick_best_free_block_greedy(free_blocks)
            else:
                best_block = self.pick_best_occupied_block_greedy(day)
            best_block.add_emergency_surgery(emergency_surgery)
            cancellations = 0
            if self.print_status:
                print(f'Best block is {best_block.block.block_type.name} with {cancellations} cancellations')
        elif self.policy == 'random':
            best_block = random.choice(self.sim_blocks[day.name])
            cancellations = random.randint(0, best_block.get_number_of_surgeries_left(including_ongoing=False,
                                                                                        includes_emergencies=False))
            best_block.cancel_surgeries(cancellations)
            best_block.add_emergency_surgery(emergency_surgery)
        else:  # All other policies currently use the same procedure
            best_block, cancellations = self.pick_best_block(day)
            best_block.cancel_surgeries(cancellations)
            best_block.add_emergency_surgery(emergency_surgery)
            #print(f'{cancellations} cancellations were done')
            
        if self.track_policy:
            self.policy_tracker.add_cancellations(cancellations)
            self.policy_tracker.add_block_type(best_block.block.block_type.name)
            self.policy_tracker.add_waited_till_free(at_least_one_block_is_free)

    def ask_policy(self, day: Day, next_decision_point):
        for sim_block in self.sim_blocks[day.name]:
            sim_block.move_to_next_decision_point(next_decision_point)

        if self.policy == 'greedy' or self.policy == 'random':
            print('Asking for a policy is not possible for the greedy or random policy')
            return None
        else:
            best_block, cancellations = self.pick_best_block(day)
            return best_block, cancellations

    def apply_decision(self, day: Day, emergency_surgery, block_id: int, cancellations: int):
        # This function should be run after ask_policy, as in ask_policy the blocks current state is updated
        chosen_block = self.get_sim_block(day, block_id)
        chosen_block.cancel_surgeries(cancellations)
        chosen_block.add_emergency_surgery(emergency_surgery)

    def get_sim_block(self, day: Day, block_id: int):
        for sim_block in self.sim_blocks[day.name]:
            if int(sim_block.get_block_key()) == block_id:
                return sim_block.get_block()
        print('There was no matching block to the provided id')
        return None

    def get_next_free_block_time(self, day:Day):
        next_free_block_time = float('inf')
        for sim_block in self.sim_blocks[day.name]:
            free_time = sim_block.get_next_free_time()
            if free_time < next_free_block_time:
                next_free_block_time = free_time

        return next_free_block_time

    def pick_best_free_block_greedy(self, free_blocks):
        minimum_total_expected_duration = float('inf')
        best_block = free_blocks[0]
        for block in free_blocks:
            block_expected_duration = block.get_remaining_total_expected_duration()
            if block_expected_duration < minimum_total_expected_duration:
                minimum_total_expected_duration = block_expected_duration
                best_block = block

        return best_block

    # TODO: Note below is not used anymore
    def pick_best_occupied_block_greedy(self, day:Day):
        # we only get here if all blocks are occupied
        best_expected_remaining_duration = float('inf')
        best_block = self.sim_blocks[day.name][0]
        for block in self.sim_blocks[day.name]:
            expected_remaining_duration = block.get_next_expected_finish_time()
            if expected_remaining_duration < best_expected_remaining_duration:
                best_expected_remaining_duration = expected_remaining_duration
                best_block = block

        return best_block

    def pick_best_block(self, day:Day):
        best_marginal_costs = float('inf')
        best_cancellations = 0
        best_block = self.sim_blocks[day.name][0]  # TODO: Do something more logical if there are emergency surgeries scheduled everywhere
        for sim_block in self.sim_blocks[day.name]:
            if not sim_block.has_emergency_scheduled():
                marginal_costs, cancellations = sim_block.get_marginal_costs_and_cancellations() # TODO: Add best marginal costs to stop calculation early
                # TODO: When there are emergencies just do deterministic evaluation
                if marginal_costs < best_marginal_costs:
                    best_marginal_costs = marginal_costs
                    best_cancellations = cancellations
                    best_block = sim_block
        if self.print_status and False:
            print(f'Best block is {best_block.block.block_type.name} with {best_cancellations} cancellations and '
                  f'marginal costs of {best_marginal_costs}')
        return best_block, best_cancellations

    def pick_best_occupied_block(self, day:Day):
        # TODO: Write this function. Return also cancellations
        for sim_block in self.sim_blocks[day.name]:
            sim_block.calculate_future_costs()
        return self.sim_blocks[day.name][0], 0

    def calculate_realized_costs(self, day:Day):  # sim blocks are from a certain day
        total_costs = 0
        info = {"waiting_costs": 0, "emergency_waiting_costs": 0, "over_costs": 0, "idle_costs": 0,
                "cancellation_costs": 0}
        criteria = {"waiting": 0, "emergency_waiting": 0, "over": 0, "idle": 0,
                "cancellation": 0}
        for sim_block in self.sim_blocks[day.name]:
            additional_cost, block_info, block_criteria = sim_block.calculate_realized_costs()
            total_costs += additional_cost
            for cost in ["waiting_costs", "emergency_waiting_costs", "over_costs", "idle_costs", "cancellation_costs"]:
                info[cost] += block_info[cost]
            for crit in ["waiting", "emergency_waiting", "over", "idle",
                         "cancellation"]:
                criteria[crit] += block_criteria[crit]
        return total_costs, info, criteria

    def draw_emergency_surgeries(self, day: Day):
        time = day.start_time
        while time < day.end_time:
            new_emergency_arrival_time = getattr(np.random, self.instance_info.emergency_arrival_distribution)(
                *self.instance_info.emergency_arrival_parameters)
            new_emergency_duration = getattr(np.random, self.instance_info.emergency_duration_distribution)(
                *self.instance_info.emergency_duration_parameters)
            time = time + new_emergency_arrival_time
            new_emergency_surgery = SimSurgery(time, day, is_emergency=True)
            new_emergency_surgery.set_realized_duration(new_emergency_duration)
            self.emergency_surgeries.append(new_emergency_surgery)
        self.emergency_surgeries.pop()  # remove the last time as it is outside the day.end_time

    def set_emergency_surgeries(self, scenario: Scenario):
        emergency_arrivals = scenario.emergency_surgery_arrival_times
        emergency_durations = scenario.emergency_surgery_durations
        for day in emergency_arrivals:
            self.emergency_surgeries[day] = []
            for i, arrival_time in enumerate(emergency_arrivals[day]):
                duration = emergency_durations[day][i]
                new_emergency_surgery = SimSurgery(arrival_time, day, is_emergency=True)
                new_emergency_surgery.set_realized_duration(duration)
                self.emergency_surgeries[day].append(new_emergency_surgery)

    @staticmethod
    def pick_best_free_block_strategy(free_blocks: List[SimBlock]):
        # note might be nicer to move the sim_block part in the loop to SimBlock
        current_best_marginal_costs = 100000
        for sim_block in free_blocks:
            marginal_costs, cancellations = sim_block.get_marginal_costs_and_cancellations()
            if marginal_costs < current_best_marginal_costs:
                best_block = sim_block
                best_block_cancellations = cancellations

        return best_block, best_block_cancellations

