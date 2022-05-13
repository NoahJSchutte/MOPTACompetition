import argparse
import sys, os
import shutil
from json.encoder import INFINITY
from src.run import runOptimisation, runALNS, get_best_solution
from src.gui.app import startGUI



class upperParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Runs a chosen solution type for the given problem instance',
                                        usage='<solution> [<args>] \n accepted solution entries are "optimise" and "alns"')

        parser.add_argument("solution",
                            type=str,
                            action='store',
                            help='choose which solution type to run')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self,args.solution):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self,args.solution)()

    def optimise(self):
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
                            default=50,
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
                            default=4,
                            choices=[1,2,4],
                            action='store',
                            help='Emergency Rate Parameter')
        parser.add_argument('--costsId', '-c',
                            type=int,
                            default=1,
                            choices=[1,2,3],
                            action='store',
                            help='Costs Id')
        args = parser.parse_args(sys.argv[2:])

        directory = os.getcwd()
        subfolder = {"S": "Stochastic/Stochastic1Stage",
                    "S2": "Stochastic/Stochastic2Stage",
                    "S3": "Stochastic/Stochastic3Stage",
                    "S4": "Stochastic/Stochastic4Stage",
                    "D": "Deterministic"}
        runOptimisation(args.algType,
            args.instanceSize,
            args.scenarioCount,
            directory + "/data/solutions/" + subfolder[args.algType],
            directory + '/data/input/Blocks.csv',
            costs_id=args.costsId,
            emergencyArrivalRate=args.emergencyRate,
            timeout=[args.timeOut, args.timeOut2, args.timeOut3, args.timeOut4],
            threads=args.threads,
            node_file_start=args.nodeFileStart,
            node_file_dir=args.nodeFileDir, )


    def alns(self):
            parser = argparse.ArgumentParser(description='Run Sim-Heuristic Solution')
            parser.add_argument('--instance_size', '-i',
                        type=int,
                        action='store',
                        default=70,
                        choices=[70, 100, 140, 200],
                        help='The instance size')
            parser.add_argument('--scenarioCount', '-s',
                            type=int,
                            action='store',
                            default=50,
                            choices=[10,50,100,1000],
                            help='Number of Scenarios')
            parser.add_argument('--alpha', '-a',
                                type=float,
                                default= 0.25,
                                help='Parameter alpha')
            parser.add_argument('--seed_start', '-r',
                                type=int,
                                default=0,
                                help='Seed start')
            parser.add_argument('--seed_end', '-q',
                                type=int,
                                default=10,
                                help='Seed end (inclusive)')
            parser.add_argument('--emergency_rate', '-e',
                                type=int,
                                default=4,
                                choices=[1,2,4],
                                help='Used emergency rate')
            parser.add_argument('--costs_id', '-c',
                                type=int,
                                default=1,
                                choices=[1,2,3],
                                help='Cost id for configuration')
            parser.add_argument('--init', '-in',
                                type=str,
                                default="Deterministic",
                                choices=['Deterministic', 'Random'],
                                help='Initialisation solution for sim-heuristic')
            parser.add_argument('--budget', '-b',
                                type=int,
                                default=100,
                                help='Number of iterations for sim heuristic')
            parser.add_argument('--useScenarioSelection', '-ss',
                                type=bool,
                                default=False,
                                choices = [True,False],
                                help='Use scenario selection [True] of [False]')



            args = parser.parse_args(sys.argv[2:])
            input_directory = 'data/solutions/ALNS'
            output_directory = f'data/solutions/ALNS'
            output_name = f'n={args.instance_size}_e={args.emergency_rate}_c={args.costs_id}'
            output_path = f'{output_directory}/{output_name}.csv'
            output_path_attributes = f'{output_directory}/{output_name}attr.csv'
            solutions = list()
            attributes_csvs = list()
  
            solution, attributes_csv = runALNS(args.instance_size,
                                            policy='exponential', 
                                            costs_id=args.costs_id,
                                            emergency_rate=args.emergency_rate, 
                                            init=args.init, 
                                            budget=args.budget,
                                            seed=11,
                                            main_scenario_size=args.scenarioCount,
                                            train_threshold=10, 
                                            number_of_pools=2,
                                            output_directory=input_directory,
                                            use_evaluator=args.useScenarioSelection,
                                            load=False)
            solutions.append(solution)
            attributes_csvs.append(attributes_csv)

            best_solution, best_index = get_best_solution(args.instance_size,solutions,emergency_rate=args.emergency_rate)
            best_solution.save_solution(output_path)
            best_attributes = attributes_csvs[best_index]
            shutil.copyfile(best_attributes, output_path_attributes)



    def gui(self):
        startGUI()

if __name__ == '__main__':
    upperParser()