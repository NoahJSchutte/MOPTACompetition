import argparse
import os, sys
from json.encoder import INFINITY

p = os.path.abspath('.')
sys.path.insert(1, p)

from src.optimisation_model.deterministic import Deterministic
from src.optimisation_model.stochastic import Stochastic
from src.optimisation_model.stochastic2stage import Stochastic2Stage
from src.optimisation_model.stochastic3stage import Stochastic3Stage
from src.optimisation_model.stochastic4stage import Stochastic4Stage
from src.general.Classes.type import Type
from src.general.Classes.scenario import Scenario
from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Classes.instance_info import InstanceInfo
from src.general.Functions.general_functions import get_lognorm_param, save_object, load_object, test_policy
from src.general.Functions.instance_configuration import instance_configuration
from src.general.Classes.simulator import Simulator
from src.general.Classes.evaluator import Evaluator
from src.general.Functions.solution_functions import check_feasibility,draw_solution_raw,convert_solution

from src.sim_heuristic.adaptive_large_neighborhood_search import ALNS

def runOptimisation(algType, instanceSize, scenarioCount, outputFileFolder, blocks, costs_id=1, emergencyArrivalRate=4, assignments='',
        timeout=[INFINITY], threads=0,
        node_file_start=0, node_file_dir=''):  # *inputFiles

    outputFile = f'{outputFileFolder}/n={instanceSize}_e={emergencyArrivalRate}_c={costs_id}_s={scenarioCount}' \
                 f'_z={threads}_t={"".join([f"{str(i)}_" for i in timeout])[:-1]}.csv'
    costs_csv = f'data/input/Costs_{costs_id}.csv'
    instance_info = instance_configuration(instanceSize, costs_csv=costs_csv,
                                           emergency_arrival_rate=emergencyArrivalRate)
    scenario_pool = load_object('data/pools/Training',
                                f'n{instanceSize}_s{scenarioCount}_e{emergencyArrivalRate}_dF')

    def getAlgType(algtype):
        if algtype == 'S':
            return Stochastic()
        elif algtype == 'S2':
            return Stochastic2Stage()
        elif algtype == 'S3':
            return Stochastic3Stage()
        elif algtype == 'S4':
            return Stochastic4Stage()
        elif algtype == 'D':
            return Deterministic(instanceSize)
        else:
            print("Unknown Algorithm Type")
            sys.exit()
    alg = getAlgType(algType)
    
    if algType in ['S','S2','S3','S4']:
        alg.read_input(blocks, costs_csv, instance_info, scenario_pool)
        alg.build_model(outputFile, timeout, threads, node_file_start, node_file_dir)
    elif algType == 'D':
        alg.read_input_blocks(blocks)
        alg.build_model(outputFile)
    else:
        print("Unknown Algorithm Type")
        sys.exit()

def runALNS(n, policy, init='Deterministic', h=1.01, alpha=0.25, costs_id=1, emergency_rate=4, use_evaluator=False,
            main_scenario_size=10, budget=100, representativeness_start=0.1,
            representativeness_end=0.01, train_threshold=10, number_of_pools=2, seed=0,
            output_directory="data/solutions/ALNS", load=False):
    directory = os.getcwd()
    segments = round(0.05*budget)
    iterations = round(budget/segments)
    costs = f"data/input/Costs_{costs_id}.csv"
    instance_info = instance_configuration(n, costs_csv=costs, emergency_arrival_rate=emergency_rate)
    instance_nr_surgeries = {}
    for block in instance_info.block_types:
        instance_nr_surgeries[block] = instance_info.block_types[block].nr_of_surgeries
    print("instance nr of surgeries on waitlist ", instance_nr_surgeries)
    if not load: # First run attributes were not carried over
        scenario_pool = load_object('data/pools/Training',
                                    f'n{n}_s{main_scenario_size}_e{emergency_rate}_dF')
        if init == 'Deterministic':
            initial_solution = f"{directory}/data/solutions/{init}/n={n}.csv"
        else: # init == 'Random'
            initial_solution = f"{directory}/data/solutions/{init}/Schedule_I={n}_seed={seed}.csv"

        # Run adaptive large neighborhood search
        alns = ALNS(iterations=iterations, segments=segments, h=h, alpha=alpha,
                    single_day_updates=False, track_solutions=False)
        simulator = Simulator(instance_info)
        simulator.set_policy(policy)
        alns.add_simulator(simulator)
        if use_evaluator:
            evaluator = Evaluator(simulator, scenario_pool, method='advanced_mip')
            evaluator.set_representativeness_parameters(do_representativeness_updates=True,
                                                        start_representativeness_threshold=representativeness_start,
                                                        end_representativeness_threshold=representativeness_end)
            evaluator.set_advanced_mip_parameters(flexible_number_of_sets=False,
                                                  number_of_pools=number_of_pools,
                                                  minimum_pool_size=1,
                                                  maximum_pool_size=50,
                                                  time_limit=10,
                                                  use_discrepancy_in_objective=True,
                                                  representativeness_ratio=1,
                                                  use_force_accept=True,
                                                  train_threshold=train_threshold)
            alns.add_evaluator(evaluator)
        else:
            alns.add_scenario_pool(scenario_pool)
        alns.initialize(initial_solution=initial_solution, instance_info=instance_info,
                        random_seed=seed)
        csv_name = f'{output_directory}/n={n}' \
                     f'_e={emergency_rate}_c={costs_id}' \
                     f'_i={budget}_h={h}_a={alpha}_p={policy}_init={init}_' \
                     f'seed={seed}'
        csv_output = f'{csv_name}.csv'
        attributes_csv = f'{csv_name}attr.csv'

        alns.search(csv_output=csv_output)
        best_solution = alns.best_solution
    else:
        csv_name = f'{output_directory}/n={n}' \
                   f'_e={emergency_rate}_c={costs_id}' \
                   f'_i={budget}_h={h}_a={alpha}_p={policy}_init={init}_' \
                   f'seed={seed}'
        csv_output = f'{csv_name}.csv'
        attributes_csv = f'{csv_name}attr.csv'
        best_solution = convert_solution(solution_csv=csv_output, block_type_dictionary=instance_info.block_types)

    return best_solution, attributes_csv

def get_best_solution(n,solutions, emergency_rate, nr_scenarios=1000, policy='exponential', scenario_folder='Validating'):
        best_objective = float('inf')
        best_index = 0

        for i, solution in enumerate(solutions):
            instance_info = instance_configuration(n)
            scenario_pool = load_object(f'data/pools/{scenario_folder}', f'n{n}_s{nr_scenarios}_e{emergency_rate}_dF')
            check_feasibility(solution, instance_info)
            simulator = Simulator(instance_info)
            obj, costs, criteria = test_policy(simulator, solution, scenario_pool, policy)
            if obj < best_objective:
                best_objective = obj
                best_solution = solution
                best_index = i

        return best_solution, best_index