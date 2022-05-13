import numpy as np
from scipy.integrate import quad, simpson
from math import exp
import math

from src.general.Classes.block import Block
from src.general.Classes.sim_surgery import SimSurgery
from src.general.Classes.instance_info import InstanceInfo
from src.general.Classes.calculator import Calculator
from src.general.Classes.scenario import Scenario

from src.general.Functions.general_functions import get_mean, draw_from_distribution
from src.general.Functions.policy_functions import *

from typing import List


class SimBlock:
    def __init__(
            self,
            block: Block,
            instance_info: InstanceInfo
    ):
        self.block = block
        self.instance_info = instance_info
        self.current_time: int = 0
        self.current_surgery_index: int = -1   # this index keeps track of which surgery is currently being performed
        # or was last finished

        self.has_ongoing_surgery = False
        self.last_surgery_finish_time: int = 0
        self.surgeries: List[SimSurgery] = list()  # these are sorted from first to last as in Block. Note that
        # emergency surgeries will be added here as well. Cancelled surgeries will be removed!
        self.number_of_cancelled_surgeries = 0
        self.calculator = Calculator(instance_info, *block.get_distribution())
        self.policy = 'deterministic'

        for scheduled_time in self.block.start_times_surgeries:
            self.surgeries.append(SimSurgery(scheduled_time))

    def add_calculator(self, calculator: Calculator):
        self.calculator = calculator

    def get_block_key(self):
        return self.block.get_key()

    def get_block(self):
        return self.block

    def set_policy(self, policy: str):
        self.policy = policy

    def draw_surgery_durations(self):
        distribution, parameters = self.block.get_distribution()
        for sim_surgery in self.surgeries:
            sim_surgery.set_realized_duration(draw_from_distribution(distribution, parameters))

    def last_surgery_has_finished(self):
        if self.current_surgery_index == len(self.surgeries)-1:
            if self.current_surgery_index == -1:
                return True
            elif self.surgeries[self.current_surgery_index].has_finished:
                return True
        return False

    def has_emergency_scheduled(self):
        not_finished_surgery_index = self.current_surgery_index if self.has_ongoing_surgery else \
            self.current_surgery_index+1
        for surgery in self.surgeries[not_finished_surgery_index:]:
            if surgery.is_emergency:
                return True

        return False

    def move_to_next_decision_point(self, new_time: int):
        self.current_time = new_time
        if self.last_surgery_has_finished():
            self.time = new_time
        else:
            if self.has_ongoing_surgery:
                current_surgery = self.surgeries[self.current_surgery_index]
                break_time = current_surgery.get_finish_time() - 1
            else:  # elif not self.has_ongoing_surgery
                scheduled_start_time_next_surgery = self.surgeries[
                    self.current_surgery_index + 1].get_scheduled_start_time()
                break_time = scheduled_start_time_next_surgery
            while break_time < self.current_time:
                if not self.has_ongoing_surgery:
                    self.current_surgery_index += 1
                    current_surgery = self.surgeries[self.current_surgery_index]
                    realized_start_time = max(current_surgery.get_scheduled_start_time(), self.last_surgery_finish_time)
                    current_surgery.set_realized_start_time(realized_start_time)

                    current_surgery.set_finish_time()
                if current_surgery.get_finish_time() <= self.current_time:  # ignore warning
                    current_surgery.set_has_finished()  # only set finish time if it actually finishes
                    self.last_surgery_finish_time = current_surgery.get_finish_time()
                    self.has_ongoing_surgery = False
                    if self.last_surgery_has_finished():  # break out when the last surgery has finished
                        break_time = self.current_time
                    else:
                        scheduled_start_time_next_surgery = self.surgeries[
                            self.current_surgery_index + 1].get_scheduled_start_time()
                        break_time = scheduled_start_time_next_surgery
                else:
                    self.has_ongoing_surgery = True
                    break_time = current_surgery.get_finish_time()

    def add_emergency_surgery(self, emergency_surgery):
        self.surgeries.insert(self.current_surgery_index+1, emergency_surgery)

    def calculate_future_costs(self, includes_emergency=True, cancellations=0):
        # we assume that there is no emergency ongoing TODO: Make sure this is the case or this is not the case
        remaining_scheduled_surgeries = self.surgeries[self.current_surgery_index + 1:]
        remaining_scheduled_start_times_without_cancellations = [surgery.get_scheduled_start_time()
                                                                 for surgery in remaining_scheduled_surgeries]
        remaining_scheduled_start_times = remaining_scheduled_start_times_without_cancellations[cancellations:]
        if self.policy == 'deterministic':
            base_costs = self.get_expected_costs_deterministic(remaining_scheduled_start_times,
                                                               includes_emergency=includes_emergency)
        elif self.policy == 'exponential':
            base_costs = self.get_expected_costs_exponential(remaining_scheduled_start_times,
                                                             includes_emergency=includes_emergency)
        elif self.policy == 'optimal':
            if self.has_ongoing_surgery:
                start_time_ongoing = self.surgeries[self.current_surgery_index].get_realized_start_time()
            else:
                start_time_ongoing = 0
            base_costs = self.get_expected_costs_optimal(remaining_scheduled_start_times,
                                                         includes_emergency=includes_emergency,
                                                         start_time_ongoing=start_time_ongoing)
        additional_cancellation_costs = cancellations * self.instance_info.cancellation_costs

        return base_costs + additional_cancellation_costs

    def get_number_of_surgeries_left(self, including_ongoing=False, includes_emergencies=True):
        # returns the number of not started surgeries that are left
        if self.last_surgery_has_finished():
            return 0
        else:
            not_ongoing_surgeries_left = len(self.surgeries[self.current_surgery_index+1:])  # note that this can be 0
            if includes_emergencies:
                if including_ongoing and self.has_ongoing_surgery:
                    return not_ongoing_surgeries_left + 1
                else:
                    return not_ongoing_surgeries_left
            else:
                number_of_surgeries = 0
                for surgery in self.surgeries[self.current_surgery_index+1:]:
                    if not surgery.is_emergency:
                        number_of_surgeries += 1
                return number_of_surgeries

    def cancel_surgeries(self, number_of_cancellations):
        self.number_of_cancelled_surgeries += number_of_cancellations
        for i in range(number_of_cancellations):
            if not self.surgeries[self.current_surgery_index+1].is_emergency:
                cancelled_surgery = self.surgeries.pop(self.current_surgery_index+1)  # current_surgery_index stays as is
                cancelled_surgery.cancel_surgery()
            #else:
            #    print("Error: And emergency is being cancelled")

    def set_surgery_durations(self, scenario: Scenario):
        elective_surgery_durations = scenario.get_elective_surgery_durations(self.block.get_type_name())
        for i, sim_surgery in enumerate(self.surgeries):
            sim_surgery.set_realized_duration(elective_surgery_durations[i])

    def get_optimal_cancellations(self):
        # TODO: use_instance_info
        cancellations = 0
        number_of_surgeries_left = self.get_number_of_surgeries_left()
        if number_of_surgeries_left > 0:
            expected_costs_includes_emergency = self.calculate_future_costs(includes_emergency=True,
                                                                             cancellations=0)
            expected_costs_with_one_more_cancellation = self.calculate_future_costs(includes_emergency=True,
                                                                                    cancellations=1)
            if expected_costs_with_one_more_cancellation < expected_costs_includes_emergency:
                cancellations += 1
                expected_costs_with_cancellation = expected_costs_includes_emergency
                while (expected_costs_with_one_more_cancellation < expected_costs_with_cancellation) and \
                        (cancellations < number_of_surgeries_left):
                    cancellations += 1
                    expected_costs_with_cancellation = expected_costs_with_one_more_cancellation
                    expected_costs_with_one_more_cancellation = self.calculate_future_costs(includes_emergency=True,
                                                                                            cancellations=cancellations)
                if expected_costs_with_one_more_cancellation < expected_costs_with_cancellation:
                    cancellations -= 1  # extra cancellation was not better

        return cancellations

    def get_marginal_costs_and_cancellations(self):
        cancellations = 0
        number_of_surgeries_left = self.get_number_of_surgeries_left()
        expected_costs_without_emergency = self.calculate_future_costs(includes_emergency=False, cancellations=0)
        expected_costs_includes_emergency = self.calculate_future_costs(includes_emergency=True, cancellations=0)
        if number_of_surgeries_left > 0:
            expected_costs_with_one_more_cancellation = self.calculate_future_costs(includes_emergency=True,
                                                                                    cancellations=1)
            if expected_costs_with_one_more_cancellation < expected_costs_includes_emergency:
                cancellations += 1
                expected_costs_with_cancellation = expected_costs_includes_emergency
                while (expected_costs_with_one_more_cancellation < expected_costs_with_cancellation) and \
                        (cancellations < number_of_surgeries_left):
                    cancellations += 1
                    expected_costs_with_cancellation = expected_costs_with_one_more_cancellation
                    expected_costs_with_one_more_cancellation = self.calculate_future_costs(includes_emergency=True,
                                                                                            cancellations=cancellations)
                if expected_costs_with_one_more_cancellation < expected_costs_with_cancellation:
                    expected_costs_with_cancellation = expected_costs_with_one_more_cancellation
                else:
                    cancellations -= 1  # extra cancellation was not better
        if cancellations > 0:
            minimal_with_emergency_costs = expected_costs_with_cancellation
        else:
            minimal_with_emergency_costs = expected_costs_includes_emergency
        marginal_costs = minimal_with_emergency_costs - expected_costs_without_emergency

        return marginal_costs, cancellations

    def calculate_realized_costs(self):
        total_duration = 0
        total_waiting_time = 0
        total_emergency_waiting_time = 0
        for surgery in self.surgeries:
            total_duration += surgery.get_realized_duration()
            if surgery.is_emergency:
                total_emergency_waiting_time += surgery.get_realized_waiting_time()
            elif surgery.is_cancelled:
                print("Error: Cancelled surgery not removed")
            else:
                total_waiting_time += surgery.get_realized_waiting_time()

        if len(self.surgeries) > 0:
            total_over_time = max(self.surgeries[-1].get_finish_time() - self.instance_info.day_finish_time, 0)
        else:
            total_over_time = 0
        total_idle_time = total_over_time + self.instance_info.day_finish_time - total_duration
        total_costs = total_waiting_time*self.instance_info.waiting_costs + \
                      total_emergency_waiting_time*self.instance_info.waiting_emergency_costs + \
                      total_over_time*self.instance_info.over_time_costs + \
                      total_idle_time+self.instance_info.idle_time_costs +\
                      self.number_of_cancelled_surgeries*self.instance_info.cancellation_costs

        costs = {"waiting_costs": total_waiting_time*self.instance_info.waiting_costs, "emergency_waiting_costs":
                total_emergency_waiting_time * self.instance_info.waiting_emergency_costs, "over_costs": total_over_time*
                self.instance_info.over_time_costs, "idle_costs": total_idle_time*self.instance_info.idle_time_costs,
                "cancellation_costs": self.number_of_cancelled_surgeries*self.instance_info.cancellation_costs}

        criteria = {"waiting": total_waiting_time, "emergency_waiting": total_emergency_waiting_time, "over": total_over_time,
                 "idle": total_idle_time,  "cancellation": self.number_of_cancelled_surgeries}

        return total_costs, costs, criteria

    def get_remaining_total_expected_duration(self):  # do not split on has_ongoing and not
        if not self.last_surgery_has_finished():
            number_of_surgeries_left = len(self.surgeries[self.current_surgery_index+1:])
            mean_duration = get_mean(*self.block.get_distribution())
            return number_of_surgeries_left * mean_duration
        else:
            return 0

    def get_next_expected_finish_time(self):
        current_surgery = self.surgeries[self.current_surgery_index]
        return current_surgery.get_expected_finish_time(*self.block.get_distribution(), self.current_time)

    def get_next_free_time(self):
        # we know there is an ongoing surgery
        return self.surgeries[self.current_surgery_index].get_finish_time()

    def get_expected_costs_deterministic(self, remaining_scheduled_start_times, includes_emergency):
        mean_duration = self.block.get_mean_duration()
        number_of_remaining_scheduled_surgeries = len(remaining_scheduled_start_times)
        deterministic_start_times = np.zeros(number_of_remaining_scheduled_surgeries)
        if self.has_ongoing_surgery:
            start_time_ongoing_surgery = self.surgeries[self.current_surgery_index].get_realized_start_time()
            current_deterministic_finish_time = start_time_ongoing_surgery + mean_duration
            last_deterministic_finish_time = max(current_deterministic_finish_time, self.current_time)
        else:
            last_deterministic_finish_time = self.current_time
        if includes_emergency:
            emergency_start_time = last_deterministic_finish_time
            last_deterministic_finish_time = emergency_start_time + self.instance_info.emergency_duration_mean
        for i, remaining_scheduled_start_time in enumerate(remaining_scheduled_start_times):
            next_start_time = max(last_deterministic_finish_time, remaining_scheduled_start_time)
            deterministic_start_times[i] = next_start_time
            last_deterministic_finish_time = next_start_time + mean_duration
        waiting_costs = np.sum(deterministic_start_times - remaining_scheduled_start_times)* \
                        self.instance_info.waiting_costs
        if includes_emergency:
            emergency_waiting_costs = (emergency_start_time - self.current_time)* \
                                      self.instance_info.waiting_emergency_costs
        else:
            emergency_waiting_costs = 0
        over_time = max(last_deterministic_finish_time - self.instance_info.day_finish_time, 0)
        over_time_costs = over_time * self.instance_info.over_time_costs
        if self.has_ongoing_surgery:
            idle_time = self.instance_info.day_finish_time + over_time - start_time_ongoing_surgery - \
                        (number_of_remaining_scheduled_surgeries+1) * mean_duration
        else:
            idle_time = self.instance_info.day_finish_time + over_time - self.current_time - number_of_remaining_scheduled_surgeries*mean_duration
        if includes_emergency:
            idle_time -= self.instance_info.emergency_duration_mean
        idle_costs = idle_time*self.instance_info.idle_time_costs
        base_costs = waiting_costs + emergency_waiting_costs + over_time_costs + idle_costs

        return base_costs

    def get_expected_costs_exponential(self, scheduled_start_times, includes_emergency):
        number_of_scheduled_surgeries = len(scheduled_start_times) + int(self.has_ongoing_surgery)
        mean_duration = self.block.get_mean_duration()  # TODO: Do I use actual mean duration here or based on exponential?
        mean_duration = self.block.get_exp_rate()
        l = 1 / self.block.get_exp_rate()
        #l = 1 / mean_duration
        emergency_mean_duration = self.instance_info.emergency_duration_mean
        scheduled_start_times.insert(0, self.current_time)  # add current time as start time
        expected_wait_emergency = 0
        # Assume exponential for now
        if self.has_ongoing_surgery:
            if number_of_scheduled_surgeries == 1:
                expected_waiting_time = 0
                if includes_emergency:
                    expected_wait_emergency = mean_duration
                expected_finish_time = self.current_time + mean_duration + int(includes_emergency)*emergency_mean_duration
            else:
                start_expectations = self.expectation_start_times_exp(l, scheduled_start_times)
                if includes_emergency:
                    start_expectations += emergency_mean_duration
                    expected_wait_emergency = mean_duration
                expected_waiting_time = np.sum(start_expectations - scheduled_start_times[1:])
                expected_finish_time = start_expectations[-1] + mean_duration
        else: # no ongoing surgery:
            # note that the start expectations start from surgery 2 here
            if includes_emergency:
                if number_of_scheduled_surgeries == 0:
                    expected_waiting_time = 0  # emergency_mean_duration
                    expected_finish_time = self.current_time + emergency_mean_duration
                else:
                    adjusted_scheduled_start_times = scheduled_start_times.copy()
                    adjusted_scheduled_start_times[1] = max(scheduled_start_times[0]+emergency_mean_duration,
                                                            scheduled_start_times[1])
                    if number_of_scheduled_surgeries == 1:
                        expected_waiting_time = adjusted_scheduled_start_times[1] - scheduled_start_times[1]
                        expected_finish_time = self.current_time + expected_waiting_time + mean_duration
                    else:
                        start_expectations = self.expectation_start_times_exp(l, adjusted_scheduled_start_times[1:])
                        expected_waiting_time = np.sum(start_expectations - scheduled_start_times[2:]) + \
                                                adjusted_scheduled_start_times[1] - scheduled_start_times[1]
                        expected_finish_time = start_expectations[-1] + mean_duration
            else:
                if number_of_scheduled_surgeries < 2:
                    expected_waiting_time = 0
                    if number_of_scheduled_surgeries == 0:
                        expected_finish_time = self.current_time
                    else:  # number_of_scheduled_surgeries == 1
                        expected_finish_time = scheduled_start_times[1] + mean_duration
                else:
                    start_expectations = self.expectation_start_times_exp(l, scheduled_start_times[1:])
                    expected_waiting_time = np.sum(start_expectations - scheduled_start_times[2:])
                    expected_finish_time = start_expectations[-1] + mean_duration

        expected_waiting_costs = expected_waiting_time*self.instance_info.waiting_costs
        expected_waiting_emergency_costs = expected_wait_emergency*self.instance_info.waiting_emergency_costs

        expected_over_time = max(expected_finish_time - self.instance_info.day_finish_time, 0)

        expected_over_time_costs = expected_over_time*self.instance_info.over_time_costs
        expected_sum_durations = self.expected_sum_durations(number_of_scheduled_surgeries,
                                                             mean_duration,
                                                             includes_emergency=includes_emergency)
        expected_idle_time = expected_over_time + self.instance_info.day_finish_time - expected_sum_durations - \
                             self.current_time
        expected_idle_time_costs = expected_idle_time * self.instance_info.idle_time_costs
        base_costs = expected_waiting_costs + expected_over_time_costs + expected_idle_time_costs + \
                      expected_waiting_emergency_costs

        return base_costs

    def get_expected_costs_optimal(self, scheduled_start_times, includes_emergency, start_time_ongoing):
        number_of_scheduled_surgeries = len(scheduled_start_times) + int(self.has_ongoing_surgery)
        dist, param = self.block.get_distribution()
        mean_duration = self.block.get_mean_duration()
        emergency_dist = self.instance_info.emergency_duration_distribution
        emergency_param = self.instance_info.emergency_duration_parameters
        emergency_mean_duration = self.instance_info.emergency_duration_mean
        expected_wait_emergency = 0
        if self.has_ongoing_surgery:
            scheduled_start_times.insert(0, self.current_time)  # add current time as start time
            if includes_emergency:
                scheduled_start_times.insert(0, self.current_time)
            start_expectations = self.expectation_start_times(dist, param, emergency_dist, emergency_param,
                                                              scheduled_start_times,
                                                              start_time=start_time_ongoing,
                                                              includes_emergency=includes_emergency)
            if includes_emergency:
                expected_start_emergency = start_expectations[1]
                expected_wait_emergency = expected_start_emergency - self.current_time
                if number_of_scheduled_surgeries > 0:
                    expected_waiting_time = np.sum(start_expectations[2:] - scheduled_start_times[2:])
                else:
                    expected_waiting_time = 0
                expected_remaining_duration_started = expected_wait_emergency
            else:
                if number_of_scheduled_surgeries > 0:
                    expected_waiting_time = np.sum(start_expectations[1:] - scheduled_start_times[1:])
                else:
                    expected_waiting_time = 0
                if self.current_time == start_time_ongoing:
                    #print("This might be an error") This can happen
                    expected_remaining_duration_started = mean_duration
                else:
                    expected_remaining_duration_started = \
                        conditional_expectation(*param, self.current_time-start_time_ongoing) - \
                        (self.current_time-start_time_ongoing)
        else:  # no ongoing surgery:
            expected_remaining_duration_started = 0
            if includes_emergency:
                scheduled_start_times.insert(0, self.current_time)  # add current time as start time
            start_expectations = self.expectation_start_times(dist, param, emergency_dist, emergency_param,
                                                              scheduled_start_times,
                                                              start_time=start_time_ongoing,
                                                              includes_emergency=includes_emergency)
            if number_of_scheduled_surgeries > 0:
                expected_waiting_time = np.sum(start_expectations[1:] - scheduled_start_times[1:])
            else:
                expected_waiting_time = 0

        if number_of_scheduled_surgeries == 0:
            if includes_emergency:
                expected_finish_time = start_expectations[-1] + emergency_mean_duration
            else:
                expected_finish_time = self.current_time
        else:
            expected_finish_time = start_expectations[-1] + mean_duration
        expected_waiting_costs = expected_waiting_time*self.instance_info.waiting_costs
        expected_waiting_emergency_costs = expected_wait_emergency*self.instance_info.waiting_emergency_costs

        expected_over_time = max(expected_finish_time - self.instance_info.day_finish_time, 0)

        expected_over_time_costs = expected_over_time*self.instance_info.over_time_costs
        expected_sum_durations = self.expected_sum_durations(number_of_scheduled_surgeries,
                                                             mean_duration,
                                                             includes_emergency=includes_emergency,
                                                             expected_duration_started=expected_remaining_duration_started)
        expected_idle_time = expected_over_time + self.instance_info.day_finish_time - expected_sum_durations - \
                             self.current_time
        expected_idle_time_costs = expected_idle_time * self.instance_info.idle_time_costs

        base_costs = expected_waiting_costs + expected_over_time_costs + expected_idle_time_costs + \
                      expected_waiting_emergency_costs


        return base_costs

    def expectation_start_times_exp(self, l, t):  # Returns expectations of 1, ..., len(t). So not of 0!
        if len(t) > 0:
            t = np.array(t)
            expectations = np.zeros(len(t)-1)
            for i in range(1, len(t)):
                expectation = self.expectation_start_time_exp(l, t, i)
                if expectation is not None:
                    expectations[i-1] = expectation
                else:
                    expectations[i-1] = expectations[i-2] + 1/l

            return expectations
        else:
            return None

    def expectation_start_times(self, dist, param, emergency_dist, emergency_param, t, start_time, includes_emergency,
                                n=11, alpha=0.001, beta=0.8, gamma=0.8, base=exp(1), expectation_bound=True):
        # Returns expectations of 0, ..., len(t), so different from exp!
        # n, alpha, beta, gamma, base are parameters for the numerical integration:
        # n:        Number of points that are used for the integration (> more accuracy)
        # alpha:    Confidence that is used in a cdf computation for the end of the domain that is used for integration
        # beta:     Weight on the use of some extension of the domain when previous surgery is expected to end late
        # gamma:    Moves the start of the exponentially increasing spaced points on the domain
        # base:     The base for the exponentially increasing points on the domain

        if len(t) > 0:
            emergency_mean_duration = self.instance_info.emergency_duration_mean
            mean_duration = self.block.get_mean_duration()
            t = np.array(t)
            expectations = np.zeros(len(t))
            expectations[0] = start_time if self.has_ongoing_surgery else t[0]
            int_list = list()
            int_list.append(np.array([0]))

            if self.has_ongoing_surgery:
                if self.current_time == start_time:
                    expected_remaining_duration_started = mean_duration
                else:
                    expected_remaining_duration_started = \
                        conditional_expectation(*param, self.current_time-start_time) - \
                        (self.current_time-start_time)

            for i in range(1, len(t)):
                if includes_emergency and self.has_ongoing_surgery and (i == 1):
                    int_list.append(get_integration_domain(t, expectations, i, mean_duration, emergency_param,
                                                               number_of_points=n,
                                                               alpha=alpha, beta=beta, gamma=gamma, base=base))
                    min_expectation = expectations[i-1] + expected_remaining_duration_started
                elif includes_emergency and self.has_ongoing_surgery and (i == 2):
                    int_list.append(get_integration_domain(t, expectations, i, emergency_mean_duration, param,
                                                           number_of_points=n,
                                                           alpha=alpha, beta=beta, gamma=gamma, base=base))
                    min_expectation = expectations[i-1] + emergency_mean_duration
                elif includes_emergency and (i == 1):
                    int_list.append(get_integration_domain(t, expectations, i, emergency_mean_duration, param,
                                                       number_of_points=n,
                                                       alpha=alpha, beta=beta, gamma=gamma, base=base))
                    min_expectation = expectations[i-1] + emergency_mean_duration
                else:
                    int_list.append(get_integration_domain(t, expectations, i, mean_duration, param,
                                                       number_of_points=n,
                                                       alpha=alpha, beta=beta, gamma=gamma, base=base))
                    if self.has_ongoing_surgery and (i == 1):
                        min_expectation = expectations[i-1] + expected_remaining_duration_started
                    else:
                        min_expectation = expectations[i-1] + mean_duration

                expectation = self.expectation_start_time(dist, param, i, t, int_list,
                                                          has_ongoing=self.has_ongoing_surgery,
                                                          start_time=start_time,
                                                          includes_emergency=includes_emergency,
                                                          emergency_param=emergency_param)
                if expectation_bound:
                    expectations[i] = max(expectation, min_expectation)
                else:
                    expectations[i] = expectation
                # TODO: Double check if always makes sense

            return np.maximum(expectations, t)
        else:
            return None

    def expected_sum_durations(self, number_of_scheduled_surgeries, elective_surgery_mean,
                               includes_emergency=False, expected_duration_started=0):
        if self.policy == 'exponential':
            expected_duration_started = elective_surgery_mean
        if includes_emergency:
            if self.has_ongoing_surgery:
                return expected_duration_started + (number_of_scheduled_surgeries-1) * elective_surgery_mean + \
                       self.instance_info.emergency_duration_mean
            else:
                return number_of_scheduled_surgeries * elective_surgery_mean + \
                       self.instance_info.emergency_duration_mean
        else:
            if self.has_ongoing_surgery:
                return expected_duration_started + (number_of_scheduled_surgeries-1) * elective_surgery_mean
            else:
                return number_of_scheduled_surgeries * elective_surgery_mean


    @staticmethod  # TODO: This might turn none static if we decide to save expectations
    def expectation_start_time_exp(l: float, t: np.array, i: int, err_tol_rel=0.01, int_max=4*480) -> float:
        if i == 0:
            return t[0]
        elif i >= 8:
            return None
        else:
            integral_results = quad(
                lambda z: z*probability_density_finish_time_exp(l, t, i-1, z,), t[i], int_max, epsrel=err_tol_rel)[0]
            return integral_results + t[i]*cumulative_distribution_finish_time_exp(l, t, i-1, t[i])

    @staticmethod  # TODO: This might turn none static if we decide to save expectations
    def expectation_start_time(dist, param, i, t, int_list, has_ongoing=False, start_time=0,
                               includes_emergency=False, emergency_param=(0, 0), rescale=True):
        if i == 0:
            if has_ongoing:
                return start_time
            else:
                return t[0]
        else:
            z = int_list[i]
            probability_densities = [probability_density_finish_time(dist, param, i-1, z_j, t, int_list, has_ongoing=has_ongoing,
                                                     start_time=start_time,
                                                     includes_emergency=includes_emergency,
                                                     emergency_param=emergency_param)
                                     for z_j in z]
            cumulative_mass = cumulative_dist_finish_time(
                dist, param, i-1, t[i], t, int_list, has_ongoing=has_ongoing, start_time=start_time,
                includes_emergency=includes_emergency, emergency_param=emergency_param)
            if rescale:
                integrated_probability = simpson(probability_densities, z)
                adjusted_probability_densities = [probability_density / (integrated_probability + cumulative_mass)
                                                  for probability_density in probability_densities]
                cumulative_mass = cumulative_mass / (integrated_probability + cumulative_mass)
            else:
                adjusted_probability_densities = probability_densities
            y = [z_j*adjusted_probability_densities[j]
                 for j, z_j in enumerate(z)]
            integral_results = simpson(y, z)
            total_result = integral_results + t[i]*cumulative_mass

            return total_result








