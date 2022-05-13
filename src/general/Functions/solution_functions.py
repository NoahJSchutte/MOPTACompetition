from re import I
from typing import overload
from src.general.Classes.instance_info import InstanceInfo
from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Classes.solution import Solution
from src.general.Classes.block import Block
from src.general.Classes.sim_block import SimBlock
from src.general.Classes.day import Day
from src.general.Functions.instance_configuration import instance_configuration
import matplotlib.pyplot as plt
import pandas as pd
import random


def check_feasibility(solution, instance_info):
    feasible = True
    for block in instance_info.block_types:
        if solution.nr_surgeries[block] > instance_info.block_types[block].nr_of_surgeries:
            feasible = False
    return feasible


def create_deterministic_solution(block_assignment,  instance_info, durations):
    """
    :param block_assignment: output of deterministic MILP
    :param instance_info: instance info
    :param durations: durations per surgery type
    :return: Solution

    Converts output of deterministic MILP into Solution() where the start times
    are starting from 0 and use mean durations
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    blocks = pd.read_csv("data/input/Blocks.csv", delimiter=";")
    nr_surgeries = {"CARD": 0, "GASTRO": 0, "GYN": 0, "MED": 0, "ORTH": 0, "URO": 0}
    solution = Solution(nr_surgeries)
    day_dict = {}
    block_type_dictionary = instance_info.block_types
    for d in days:
        day_blocks = blocks.loc[blocks['DAY'] == d]
        day = Day(name=d, start_time=0, end_time=8 * 60)
        for index, row in day_blocks.iterrows():
            b = row['BLOCK']
            specialty = row['TYPE']
            block_type = block_type_dictionary[specialty]
            block = Block(block_type=block_type, key=b, end_time=8 * 60)
            surgeries = block_assignment[b]
            start = 0
            for s in range(int(surgeries)):
                block.add_surgery(round(float(start)))
                start += durations[specialty]
            nr_surgeries[specialty] += len(block.start_times_surgeries)
            day.add_block(block)
        day_dict[d] = day
    solution.days = day_dict

    return solution


def convert_solution(solution_csv, block_type_dictionary):
    # this function now correctly reads csvs
    # to do: check effect on results
    """
    :param solution_csv: str   , location of file containing csv of solution schedule
    :return: solution: Solution, converted version of solution
    """

    blocks = pd.read_csv("data/input/Blocks.csv", delimiter=";")
    df = pd.read_csv(solution_csv)
    df = df[df['Surgery'].astype(str).str[0] != 'E']
    df['Block'] = pd.to_numeric(df['Block'], downcast='integer')
    nr_surgeries = {"CARD": 0, "GASTRO": 0, "GYN": 0, "MED": 0, "ORTH": 0, "URO": 0}
    solution = Solution(nr_surgeries)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    day_dict = {}
    for d in days:
        day_schedule = df.loc[df['Day'] == d]
        day_blocks = blocks.loc[blocks['DAY'] == d]
        day = Day(name=d, start_time=0, end_time=8 * 60)

        for index, row in day_blocks.iterrows():
            b = row['BLOCK']
            specialty = row['TYPE']
            block_schedule = day_schedule.loc[day_schedule['Block'] == b]
            block_type = block_type_dictionary[specialty]
            block = Block(block_type=block_type, key=b, end_time=8 * 60)
            surgery_start_times = block_schedule.Start.sort_values().tolist()
            for start_time in surgery_start_times:
                block.add_surgery(round(float(start_time)))
            nr_surgeries[specialty] += len(block.start_times_surgeries)
            day.add_block(block)
        day_dict[d] = day
    solution.days = day_dict

    return solution


def analyse_solution(solution, instance_info, scenario_pool,index, directory):
    """
    :param solution: Solution, solution object
    dumps out solution analysis to csv
    """
    outputfile = directory + "/testFiles/SolutionAnalysis/SolutionAnalysisI" + str(instance_info.instance_size) + str(len(scenario_pool.scenarios)) + ".csv"
    with open(outputfile,'w') as f:
        f.write("\n Full Schedule \n")
        f.write("Day ; Block ; Type ; Start \n")
        for day in solution.days:
            for b in solution.days[day].blocks:
                block = solution.days[day].blocks[b]
                f.write(f'{day} ; {b} ; {block.block_type.name} ; {[surgery for surgery in block.start_times_surgeries]}\n')

        f.write("\n Block Level Schedule Summary \n")
        f.write("Day ; Block ; Surgeries ; AvgDiff ; Difference \n")
        for day in solution.days:
            for b in solution.days[day].blocks:
                block = solution.days[day].blocks[b]
                difference_list = []
                average_difference = 0
                if len(block.start_times_surgeries)>1:
                    difference_list = block.get_difference_list()
                    average_difference = block.get_average_difference() 
                f.write(f'{day} ; {b} ; {len(block.start_times_surgeries)} ; {average_difference} ; {difference_list} \n')

        f.write("\n Day Level Schedule Summary \n")
        f.write("Day ; Surgeries ; AvgDiff \n")
        for day in solution.days:
            difference = []
            surgeries = 0
            for b in solution.days[day].blocks:
                block = solution.days[day].blocks[b]
                difference.extend(block.get_difference_list())
                surgeries = surgeries + len(block.start_times_surgeries)
            average_difference = sum(difference)/len(difference)
            f.write(f'{day} ; {surgeries} ; {average_difference} \n')
    draw_solution_scenario(solution, instance_info, scenario_pool,index, directory)


def draw_solution_raw(solution,fig,ax):
    fig, ax = plt.subplots()
    fig.clear()
    ax.clear()
    yticks = []
    yticklabels = []
    i=0
    print("drawing")
    for day in solution.days:
        j=1 #to offset plot
        for b in solution.days[day].blocks:
            ax.broken_barh([(start_time, 2) for k,start_time in enumerate(solution.days[day].blocks[b].start_times_surgeries)], ((i*100)+(10*j), 3), facecolors=('blue','green'))
            yticks.append((i*100)+(10*j))
            yticklabels.append(solution.days[day].blocks[b].key)
            j+=1
        i+=1

    ax.set_xlabel('Blocks')
    yticks.extend([i*100 for i in range(len(solution.days))])
    yticklabels.extend([solution.days[day].name for day in solution.days])
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=6)
    ax.grid(True)
    return fig,ax



def draw_solution_scenario(solution, instance_info, scenario_pool, index, directory, no_scenarios=False,
                           expected_durations=False):
    outputfile = directory + "/testFiles/SolutionAnalysis/SolutionPlotI" + str(instance_info.instance_size) + "Scenario" + str(index) + "of" + str(len(scenario_pool.scenarios)) + ".png"
    block_length = 10
    scenario = scenario_pool.scenarios[index]
    fig, ax = plt.subplots()
    yticks = []
    yticklabels = []
    i=0
    for day in solution.days:
        j=1 #to offset plot
        for b in solution.days[day].blocks:
            sim_block = SimBlock(solution.days[day].blocks[b], instance_info)
            sim_block.set_surgery_durations(scenario)
            print("scenario ",index,b,[(start_time, sim_block.surgeries[k].realized_duration) for k,start_time in enumerate(solution.days[day].blocks[b].start_times_surgeries)])
            hbars = list()
            hbars_start = list()
            overlap_hbars = list()
            previous_block_finish = 0
            start_times = solution.days[day].blocks[b].start_times_surgeries
            print(start_times)
            for k,start_time in enumerate(solution.days[day].blocks[b].start_times_surgeries):

                if not no_scenarios and not expected_durations:
                    block_length = sim_block.surgeries[k].realized_duration
                elif expected_durations: # expected_durations
                    block_length = sim_block.block.get_mean_duration()

                if not no_scenarios or expected_durations:
                    if previous_block_finish > start_time:
                        overlap_hbars.append((start_time, previous_block_finish - start_time))

                previous_block_finish = start_time + block_length
                hbars.append((start_time, block_length))
                hbars_start.append((start_time, 5))
            ax.broken_barh(hbars, ((i*100)+(10*j), 3), facecolors=('blue'))
            ax.broken_barh(hbars_start, ((i*100)+(10*j), 3), facecolors=('orange'))
            if not no_scenarios or expected_durations:
                ax.broken_barh(overlap_hbars, ((i*100)+(10*j), 3), facecolors=('red'))

            yticks.append((i*100)+(10*j))
            yticklabels.append(f'{solution.days[day].blocks[b].block_type.name}{solution.days[day].blocks[b].key}')
            j+=1
        #plot emergency arrivals
        if not no_scenarios:
            ax.broken_barh([(arrival, 1) for arrival in scenario.emergency_surgery_arrival_times[day]], ((i*100), (j*10)), facecolors=('red'))
        i+=1

    ax.set_xlabel('Blocks')
    yticks.extend([i*100 for i in range(len(solution.days))])
    yticklabels.extend([solution.days[day].name for day in solution.days])
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=6)
    ax.grid(True)
    plt.savefig(outputfile)



def generate_random_solution(n):
    """
    :param n: instance size: Int
    :return: solution: random initialized solution: Solution
    """

    # Read input blocks
    blocks = pd.read_csv("data/input/Blocks.csv", delimiter=";")

    # Initialize empty solution
    nr_surgeries = {"CARD": 0, "GASTRO": 0, "GYN": 0, "MED": 0, "ORTH": 0, "URO": 0}
    solution = Solution(nr_surgeries)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_dict = {}
    instance_nr_patients = {}
    for d in days:
        day = Day(name=d, start_time=0, end_time=8 * 60)
        day_dict[d] = day

    # Instance configuration to know number of patients on waitlist
    instance_info = instance_configuration(n, "testFiles/Input/Costs.csv")

    # Iterate over different block types [CARD, GASTRO, GYN, MED, ORTH, URO]
    for block in instance_info.block_types:
        type_class = instance_info.block_types[block]
        nr_patients = instance_info.block_types[block].nr_of_surgeries  # nr of patients
        instance_nr_patients[block] = nr_patients
        blocks_of_type = blocks.loc[blocks['TYPE'] == block]["BLOCK"].to_list()  # blocks with that type

        # Initialize assignment dictionary
        assignment_dict = {}
        for b in blocks_of_type:
            assignment_dict[b] = 0
        assignment_dict["not assigned"] = 0

        # Divide total number of patients of certain types between different
        # blocks of that type + not assigned possibility
        for i in range(nr_patients):
            assignment = random.choice(blocks_of_type + ["not assigned"])
            assignment_dict[assignment] += 1

        # Add surgery times to block when surgeries are assigned
        # Add block classes to days
        for b in blocks_of_type:
            day = blocks.loc[blocks['BLOCK'] == b]["DAY"].values[0]
            block_class = Block(block_type=type_class, key=b, end_time=8 * 60)
            nr_surgeries = assignment_dict[b]
            surgery_start_times = sorted(random.sample(range(0, 8 * 60), nr_surgeries))
            block_class.start_times_surgeries = [float(start_time) for start_time in surgery_start_times]
            day_dict[day].add_block(block_class)

    # Add day dictionary to solution
    solution.days = day_dict

    # Recount nr of surgeries
    solution.count_surgeries()
    print("patients on waitlist ", instance_nr_patients)
    print("solution random ", solution.nr_surgeries)

    return solution