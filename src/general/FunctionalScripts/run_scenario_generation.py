# This script generates scenarios so they can be used afterwards

from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Functions.general_functions import save_object
from src.general.Functions.instance_configuration import instance_configuration


def create_pools(instance_sizes, scenario_pool_sizes, save_directory, seed, fitted=True, emergency_rates=(2, 4)):
    costs_csv = 'data/input/Costs.csv'
    if fitted:
        dist_csv = 'data/input/Distributions_fitted.csv'
        dist_code = 'F'
    else:
        dist_csv = 'data/input/Distributions_implied.csv'
        dist_code = 'I'

    for instance_size in instance_sizes:
        for emergency_rate in emergency_rates:
            instance_info = instance_configuration(instance_size, costs_csv=costs_csv, dist_csv=dist_csv,
                                                   emergency_arrival_rate=emergency_rate)
            for scenario_pool_size in scenario_pool_sizes:
                file_name = f'n{instance_size}_s{scenario_pool_size}_e{emergency_rate}_d{dist_code}'
                scenario_pool = ScenarioPool(instance_info)
                scenario_pool.generate_scenarios(scenario_pool_size, seed=int(seed+scenario_pool_size))
                save_object(scenario_pool, save_directory, file_name)


instance_sizes = [70, 100, 140, 200]
scenario_pool_sizes = [10, 50, 100, 1000]
emergency_rates = (1, 2, 4)
save_directory = f'data/pools/Training'
create_pools(instance_sizes, scenario_pool_sizes, save_directory, seed=5, emergency_rates=emergency_rates)
save_directory = f'data/pools/Testing'
create_pools(instance_sizes, scenario_pool_sizes, save_directory, seed=6, emergency_rates=emergency_rates)
save_directory = f'data/pools/Validating'
create_pools(instance_sizes, scenario_pool_sizes, save_directory, seed=7, emergency_rates=emergency_rates)
