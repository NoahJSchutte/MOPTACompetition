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
from src.general.Functions.general_functions import get_lognorm_param, save_object, load_object
from src.general.Functions.instance_configuration import instance_configuration


def getAlgType(algtype):
    if algtype == 'S':
        return Stochastic()
    elif algtype == 'S2':
        return Stochastic2Stage()
    elif algtype == 'S3':
        return Stochastic3Stage()
    elif algtype == 'S4':
        return Stochastic4Stage()
    #elif algtype == 'D':
    #    return Deterministic()
    else:
        print("Unknown Algorithm Type")
        sys.exit()


def run(algType, instanceSize, scenarioCount, outputFileFolder, blocks, costs_id=1, emergencyArrivalRate=4, assignments='',
        timeout=[INFINITY], threads=0,
        node_file_start=0, node_file_dir=''):  # *inputFiles

    outputFile = f'{outputFileFolder}/n={instanceSize}_e={emergencyArrivalRate}_c={costs_id}_s={scenarioCount}' \
                 f'_z={threads}_t={"".join([f"{str(i)}_" for i in timeout])[:-1]}.csv'
    costs_csv = f'testFiles/Input/Costs_{costs_id}.csv'
    instance_info = instance_configuration(instanceSize, costs_csv=costs_csv,
                                           emergency_arrival_rate=emergencyArrivalRate)
    scenario_pool = load_object('testFiles/Pools/Training',
                                f'n{instanceSize}_s{scenarioCount}_e{emergencyArrivalRate}_dF')

    alg = getAlgType(algType)
    
    if algType in ['S','S2','S3','S4']:
        alg.read_input(blocks, costs_csv, instance_info, scenario_pool)
        alg.build_model(outputFile, timeout, threads, node_file_start, node_file_dir)
    elif algType == 'D':
        alg.read_input(blocks, instance_info, scenario_pool)
        alg.build_model(outputFile, timeout, threads, node_file_start, node_file_dir)
    else:
        print("Unknown Algorithm Type")
        sys.exit()


def run_csv(algType, outputFile, blocks, costs, surgeriesFile, scenariosFile='xxx.csv',
            timeout=INFINITY):  # *inputFiles
    alg = getAlgType(algType)

    if algType == 'SO':
        alg.read_input_csv(blocks, costs, surgeriesFile, scenariosFile)
        alg.build_model(outputFile, timeout)
    elif algType == 'D':
        alg.read_input_csv(blocks, surgeriesFile)
        alg.build_model(outputFile, timeout)
    else:
        print("Unknown Algorithm Type to run from CSV")
        sys.exit()


def createParser():
    parser = argparse.ArgumentParser(description='Run Optimisation Model')
    parser.add_argument('--algType', '-a',
                        type=str,
                        action='store',
                        default="D",
                        help='The model type, S for Stochastic, D for Deterministic')
    parser.add_argument('--instanceSize', '-i',
                        type=int,
                        action='store',
                        default=70,
                        help='Number of Elective Surgeries')
    parser.add_argument('--scenarioCount', '-s',
                        type=int,
                        action='store',
                        default=1,
                        help='Number of Scenarios')
    parser.add_argument('--timeOut', '-t',
                        type=int,
                        default=INFINITY,
                        action='store',
                        help='Time Out Value stage 1')
    parser.add_argument('--timeOut2', '-u',
                        type=int,
                        default=INFINITY,
                        action='store',
                        help='Time Out Value stage 2')
    parser.add_argument('--timeOut3', '-v',
                        type=int,
                        default=INFINITY,
                        action='store',
                        help='Time Out Value stage 3')
    parser.add_argument('--timeOut4', '-w',
                        type=int,
                        default=INFINITY,
                        action='store',
                        help='Time Out Value stage 4')
    parser.add_argument('--threads', '-z',
                        type=int,
                        default=0,
                        action='store',
                        help='Number of Threads Used')
    parser.add_argument('--nodeFileStart', '-n',
                        type=float,
                        default=0,
                        action='store',
                        help='Node File Start Parameter')
    parser.add_argument('--nodeFileDir', '-d',
                        type=str,
                        default='',
                        action='store',
                        help='Node File Dir Parameter')
    parser.add_argument('--emergencyRate', '-e',
                        type=int,
                        default='',
                        action='store',
                        help='Emergency Rate Parameter')
    parser.add_argument('--costsId', '-c',
                        type=int,
                        default='',
                        action='store',
                        help='Costs Id')
    return parser


def main():
    parser = createParser()
    args = parser.parse_args()
    print(args)

    directory = os.getcwd()
    subfolder = {"S": "Stochastic/Stochastic1Stage",
                "S2": "Stochastic/Stochastic2Stage",
                "S3": "Stochastic/Stochastic3Stage",
                "S4": "Stochastic/Stochastic4Stage",
                "D": "Deterministic"}
    run(args.algType,
        args.instanceSize,
        args.scenarioCount,
        directory + "/testFiles/Solutions/" + subfolder[args.algType],
        directory + '/testFiles/Input/Blocks.csv',
        costs_id=args.costsId,
        emergencyArrivalRate=args.emergencyRate,
        timeout=[args.timeOut, args.timeOut2, args.timeOut3, args.timeOut4],
        threads=args.threads,
        node_file_start=args.nodeFileStart,
        node_file_dir=args.nodeFileDir, )


main()
#directory = os.getcwd()
# run_csv("S",
#         directory + "/testFiles/Solutions/Schedule_I=testS.csv",
#         directory +'/testFiles/Input/Blocks.csv',
#         directory +'/testFiles/Input/Costs.csv',
#         directory + "/testFiles/Input/Surgeries_Uncertain_I=10(2).csv",
#         directory + "/testFiles/Input/Emergency_Scenarios(2).csv", )

#for n in [70, 100, 140, 200]:
#    run_csv("D",
#            directory + "/testFiles/Solutions/Deterministic/Schedule_I=" + str(n) + ".csv",
#            directory + '/testFiles/Input/Blocks.csv',
#            directory + '/testFiles/Input/Costs.csv',
#            directory + "/testFiles/Input/Surgeries_I=" + str(n) + ".csv",
#             )
