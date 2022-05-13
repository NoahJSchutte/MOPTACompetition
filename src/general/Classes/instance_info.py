from typing import List, Dict
import pandas as pd
from src.general.Classes.type import Type
from src.general.Functions.general_functions import get_mean


class InstanceInfo:
    def __init__(
            self,
            instance_size,
            not_scheduling_costs,
            cancellation_costs,
            waiting_costs,
            waiting_emergency_costs,
            over_time_costs,
            idle_time_costs,
            emergency_duration_distribution: str,
            emergency_duration_parameters: List[float], # see Type class for convention on distributions
            emergency_arrival_distribution: str,
            emergency_arrival_parameters: List[float],
    ):
        self.instance_size = instance_size
        self.not_scheduling_costs = not_scheduling_costs
        self.cancellation_costs = cancellation_costs
        self.waiting_costs = waiting_costs
        self.waiting_emergency_costs = waiting_emergency_costs
        self.over_time_costs = over_time_costs
        self.idle_time_costs = idle_time_costs
        self.emergency_duration_distribution = emergency_duration_distribution
        self.emergency_duration_parameters = emergency_duration_parameters
        self.emergency_duration_mean = get_mean(emergency_duration_distribution, emergency_duration_parameters)
        self.emergency_arrival_distribution = emergency_arrival_distribution
        self.emergency_arrival_parameters = emergency_arrival_parameters
        self.block_types: Dict[str: Type] = dict()
        self.day_start_time = 0
        self.day_finish_time = 8*60

    def add_type(self, block_type: Type):
        type_name = block_type.name
        self.block_types[type_name] = block_type

    def set_costs(self,csv_costs):
        costs = pd.read_csv(csv_costs, sep=";", header=None, index_col=0, squeeze=True).to_dict()
        self.not_scheduling_costs = costs["NOTSCHEDULING"]
        self.cancellation_costs = costs["CANCELLING"]
        self.waiting_costs = costs["ELECTIVEWAITINGTIME"]
        self.waiting_emergency_costs = costs["EMERGENCYWAITINGTIME"]
        self.over_time_costs = costs["OVERTIME"]
        self.idle_time_costs = costs["IDLETIME"]

    def get_block_types(self):
        return self.block_types


