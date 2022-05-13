from src.general.Classes.scenario import Scenario
from src.general.Classes.instance_info import InstanceInfo
from typing import List

import numpy as np
import pandas as pd
import os, copy


class ScenarioPool:
    def __init__(
            self,
            instance_info: InstanceInfo  # scenario pool is dependent on the instance
    ):
        self.instance_info = instance_info
        self.scenarios: List[Scenario] = list()  # when generated, these are ordered by id
        self.seed = 5

    def get_size(self):
        return len(self.scenarios)

    def get_instance_info(self):
        return self.instance_info

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)

    def set_scenarios(self, scenario_list: List[Scenario]):
        self.scenarios = scenario_list

    def get_scenario(self, scenario_index: int):
        # Note that this returns the index, not the id
        return self.scenarios[scenario_index]

    def get_scenario_ids(self):
        return [scenario.id for scenario in self.scenarios]

    def get_scenario_set(self):
        return set(self.scenarios)

    def union(self, other_pool: 'ScenarioPool'):
        this_scenario_set = self.get_scenario_set()
        other_scenario_set = other_pool.get_scenario_set()
        union_set = this_scenario_set.union(other_scenario_set)
        union_scenario_pool = ScenarioPool(self.instance_info)
        for scenario in union_set:
            union_scenario_pool.add_scenario(scenario)

        return union_scenario_pool

    def difference(self, other_pool: 'ScenarioPool'):
        this_scenario_set = self.get_scenario_set()
        other_scenario_set = other_pool.get_scenario_set()
        difference_set = this_scenario_set.difference(other_scenario_set)
        difference_scenario_pool = ScenarioPool(self.instance_info)
        for scenario in difference_set:
            difference_scenario_pool.add_scenario(scenario)

        return difference_scenario_pool

    def generate_scenarios(self, number_of_scenarios=10000, seed=5):
        self.seed = seed
        np.random.seed(self.seed)
        for i in range(number_of_scenarios):
            elective_surgery_dict = dict()
            emergency_times, emergency_durations = self.draw_emergency_surgeries(self.instance_info.day_start_time,
                                                                                    self.instance_info.day_finish_time)
            #total_surgeries = 0
            for block_type_name, block_type in self.instance_info.block_types.items():
                #print(f'{block_type_name}: {self.instance_info.instance_size*block_type.percentage_of_surgeries}')
                number_of_elective_surgeries = round(self.instance_info.instance_size*block_type.percentage_of_surgeries)
                elective_surgery_draws = getattr(np.random, block_type.distribution)(
                    *block_type.parameters, number_of_elective_surgeries)
                elective_surgery_dict[block_type_name] = list(elective_surgery_draws)
                #total_surgeries += number_of_elective_surgeries
            #print(total_surgeries)
            self.scenarios.append(Scenario(i, copy.deepcopy(elective_surgery_dict), emergency_times, emergency_durations))
        #self.save_scenario_pool()

    def save_scenario_pool(self):
        #Save the scenario pool
        result = []
        for scenario in self.scenarios:
            result.append({
                "Scenario":scenario.id,
                "Duration": scenario.emergency_surgery_durations,
                "ArrivalTime": scenario.emergency_surgery_arrival_times,
                "ElectiveDuration": scenario.elective_surgery_durations,
            })
        result = pd.DataFrame(result)
        result.to_csv("testFiles/Pools/ScenarioPool" + str(len(self.scenarios)) + ".csv", 
                        index=False, 
                        sep=";")
 

    def draw_emergency_surgeries(self, day_start_time, day_finish_time):
        time = day_start_time
        arrivals = {}
        durations = {}      
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            time = day_start_time
            arrival_times = list()
            emergency_durations = list()
            while time < day_finish_time:
                new_emergency_arrival_time = getattr(np.random, self.instance_info.emergency_arrival_distribution)(
                    *self.instance_info.emergency_arrival_parameters)
                new_emergency_duration = getattr(np.random, self.instance_info.emergency_duration_distribution)(
                    *self.instance_info.emergency_duration_parameters)
                time = time + new_emergency_arrival_time
                arrival_times.append(time)
                emergency_durations.append(new_emergency_duration)

            arrival_times.pop()  # remove the last time as it is outside the day.end_time
            emergency_durations.pop()
            arrivals[day] = arrival_times
            durations[day] = emergency_durations


        return arrivals, durations

