from typing import List
from src.general.Functions.general_functions import get_mean


class Type:
    def __init__(
            self,
            name: str,
            distribution: str,
            parameters: List[float],  # np.random is used for draws, so: normal: [mean, sigma], lognormal: [mean, sigma],
            # exponential: [shape, scale], etc.
            percentage_of_surgeries: float,
            nr_of_surgeries: float,
            exp_rate=0
    ):
        self.name = name
        self.distribution = distribution
        self.parameters = parameters
        self.exp_rate = exp_rate
        self.mean_duration = get_mean(distribution, parameters)
        self.percentage_of_surgeries = percentage_of_surgeries
        self.nr_of_surgeries = nr_of_surgeries

    def add_distribution(self, distribution, parameters):
        self.distribution = distribution
        self.parameters = parameters

    def get_distribution(self):
        return self.distribution, self.parameters

    def get_mean_duration(self):
        return self.mean_duration

