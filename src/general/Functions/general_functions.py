import argparse
import numpy as np
import pickle
import time
from scipy.stats import norm
import pandas as pd

def draw_from_distribution(distribution, parameters):
    return getattr(np.random, distribution)(*parameters)


def get_mean(distribution, parameters):
    if distribution == 'exponential':
        return parameters[0]
    elif distribution == 'lognormal':
        m = parameters[0]
        s = parameters[1]
        return np.exp(m + 0.5*s**2)


def get_std(distribution, parameters):
    if distribution == 'lognormal':
        m = parameters[0]
        s = parameters[1]
        return ((np.exp(s**2)-1)*np.exp(2*m+s**2))**0.5


def get_lognorm_param(m, s):
    # takes in 'regular' mean and std, returns lognormal mean and std
    std = np.sqrt(np.log(((s/m)**2)+1))
    mean = np.log(m)-0.5*std**2
    return mean, std


def lognorm_conditional_expectation_given_min(m, s, k):  # E[X | X >= k]
    k = max(k, 0.01)
    return np.exp(m+0.5*s**2) * norm.cdf((m+s**2-np.log(k))/s) / (1 - norm.cdf((np.log(k)-m)/s))


def save_object(object, directory, file_name):
    #Path(directory).mkdir(parents=True, exist_ok=True)
    with open(f'{directory}/{file_name}.pkl', 'wb') as file:
        pickle.dump(object, file)


def load_object(directory, file_name):
    with open(f'{directory}/{file_name}.pkl', 'rb') as open_file:
        return pickle.load(open_file)


def test_policy(simulator, solution, scenario_pool, policy, duration=True, print_status=False):
    now = time.time()
    simulator.reset()
    simulator.set_policy(policy)
    objective, info, criteria = simulator.evaluate(solution, scenario_pool, print_status=print_status, return_information=True)
    new_now = time.time()
    print(f'Policy: {policy}\n'
          f'Objective: {objective}\n'
          f'Costs: {info}\n'
          f'Criteria: {criteria}\n'
          f'Solve time: {new_now - now}')

    return objective, info, criteria

def create_parser():
    parser = argparse.ArgumentParser(description='Parser for run')
    parser.add_argument('--instance_sizes', '-n',
                        nargs='*',
                        type=int,
                        action='store',
                        default=[70, 100, 140, 200],
                        help='The instance size')
    parser.add_argument('--alphas', '-a',
                        nargs="*",
                        type=float,
                        default=[0.25, 0.5, 0.75],
                        help='Parameter alpha')
    parser.add_argument('--seed_start', '-r',
                        type=int,
                        default=0,
                        help='Seed start')
    parser.add_argument('--seed_end', '-q',
                        type=int,
                        default=10,
                        help='Seed end (inclusive)')
    parser.add_argument('--emergency_rates', '-e',
                        nargs="*",
                        type=int,
                        default=[1, 2, 4],
                        help='Used emergency rate')
    parser.add_argument('--costs_ids', '-c',
                        nargs="*",
                        type=int,
                        default=[1, 2, 3],
                        help='Cost id for configuration')
    return parser


def rank_hyperparameters(input_file, output_file):

    parameters = pd.read_csv(input_file, sep=";")
    #parameters_mean = parameters.groupby(['Alpha', 'Initial_temp_h', 'Instance_size', 'Initialization'], as_index=False).agg(
    #    {'Objective': "mean"})
    #parameters_std = parameters.groupby(['Alpha', 'Initial_temp_h', 'Instance_size', 'Initialization'],
      #                                   as_index=False).agg({'Objective': "std"})
    #parameters_min= parameters.groupby(['Alpha', 'Initial_temp_h', 'Instance_size', 'Initialization'],
      #                                  as_index=False).agg({'Objective': "min"})
    parameters = parameters.groupby(['Alpha', 'Initial_temp_h', 'Instance_size', 'Initialization'], as_index=False).\
        agg(Mean=('Objective', 'mean'), Std=('Objective', 'std'), Min=('Objective', 'min'))
    df70 = parameters[parameters['Instance_size'] == 70]
    df70 = df70.drop('Instance_size', 1)
    df70 = df70.rename(columns={"Mean": "Mean_I=70", "Std": "Std_I=70", "Min": "Min_I=70"})
    df100 = parameters[parameters['Instance_size'] == 100]
    df100 = df100.drop('Instance_size', 1)
    df100 = df100.rename(columns={"Mean": "Mean_I=100", "Std": "Std_I=100", "Min": "Min_I=100"})
    new_table = pd.merge(df70, df100, on=["Alpha", "Initial_temp_h", "Initialization"])
    df140 = parameters[parameters['Instance_size'] == 140]
    df140 = df140.drop('Instance_size', 1)
    df140 = df140.rename(columns={"Mean": "Mean_I=140", "Std": "Std_I=140", "Min": "Min_I=140"})
    new_table = pd.merge(new_table, df140, on=["Alpha", "Initial_temp_h", "Initialization"])
    df200 = parameters[parameters['Instance_size'] == 200]
    df200 = df200.drop('Instance_size', 1)
    df200 = df200.rename(columns={"Mean": "Mean_I=200", "Std": "Std_I=200", "Min": "Min_I=200"})
    new_table = pd.merge(new_table, df200, on=["Alpha", "Initial_temp_h", "Initialization"])

    new_table = new_table.round({'Mean_I=70': 0, 'Mean_I=100': 0,
                                 'Mean_I=140': 0, 'Mean_I=200': 0,
                                 'Std_I=70': 0, 'Std_I=100': 0,
                                 'Std_I=140': 0, 'Std_I=200': 0,
                                'Min_I=70': 0, 'Min_I=100': 0,
                                'Min_I=140': 0, 'Min_I=200': 0
                                    })

    new_table['rank_70'] = new_table['Mean_I=70'].rank(na_option='bottom')
    new_table['rank_100'] = new_table['Mean_I=100'].rank(na_option='bottom')
    new_table['rank_140'] = new_table['Mean_I=140'].rank(na_option='bottom')
    new_table['rank_200'] = new_table['Mean_I=200'].rank(na_option='bottom')

    new_table['sum_rank'] = new_table['rank_70'] + new_table['rank_100'] + new_table['rank_140'] + new_table['rank_200']
    new_table.to_csv(output_file, index=False, sep=";")

    best_config_index = new_table['sum_rank'].idxmin()
    print(f'Best configuration: alpha is {new_table.iloc[best_config_index,:]["Alpha"]} '
          f'h is {new_table.iloc[best_config_index,:]["Initial_temp_h"]} '
          f'initialization is {new_table.iloc[best_config_index,:]["Initialization"]}')
