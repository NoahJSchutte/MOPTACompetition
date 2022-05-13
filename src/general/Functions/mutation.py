import copy
import random
from src.general.Functions.solution_functions import check_feasibility
#random.seed(5)


def shift(old_list, direction):
    """
    :param list:
    :param direction: left or right
    :return: list with one surgery start time 15 minutes shifted to left or right
    """
    new_list = copy.deepcopy(old_list)
    surgery = random.randint(0, len(new_list)-1)
    if direction == "left":
        new_list[surgery] = max(0, new_list[surgery] - 15)

    elif direction == "right":
        new_list[surgery] = min(8*60, new_list[surgery] + 15)

    return new_list


def add_slack(old_list, base_shift=30, slack_begin=False, slack_end=False):
    new_list = copy.deepcopy(old_list)
    number_of_surgeries = len(old_list)
    if number_of_surgeries > 0:
        if slack_begin:
            shift = base_shift / number_of_surgeries
            compounding_shift = shift
            for i in range(number_of_surgeries).__reversed__():
                new_list[i] = round(old_list[i] + compounding_shift)
                compounding_shift += shift

        if slack_end:
            if number_of_surgeries == 1:
                new_list[0] -= base_shift
            else:
                shift = round(base_shift / (number_of_surgeries-1))
                compounding_shift = shift
                for i in range(1, number_of_surgeries):
                    new_list[i] = round(old_list[i] - compounding_shift)
                    compounding_shift += shift

    return new_list


def spread_equally(list_input, slack_begin=False, slack_end=False):
    if list_input is not None:

        start = 0 if slack_begin is False else 30
        end = 8*60 if slack_end is False else 8*60-30

        new_list = [round(((end-start) / len(list_input)) * i + start) for i in range(len(list_input))]
    else:
        new_list = copy.deepcopy(list_input)
    return new_list


def spread_increasing(list_input, total_spread=60):
    number_of_surgeries = len(list_input)
    new_list = copy.deepcopy(list_input)
    if number_of_surgeries > 2:
        spread_step = total_spread / (number_of_surgeries-2)
        base_shift = -total_spread / 2
        gap_change = base_shift
        for i in range(1, number_of_surgeries):
            new_list[i] = new_list[i-1] + (list_input[i] - list_input[i-1]) + gap_change
            gap_change += spread_step

    return new_list


def remove_one_random(solution, spread=None, last=False, slack_begin=False, slack_end=True):
    """
    :param solution: Solution
    :param spread: Boolean
    :param last: Boolean is 1 if you want to remove last surgery of block
    :return: Solution
    """
    mutation = False
    while not mutation:
        random_day = random.choice(list(solution.days.items()))
        random_block = random.choice(list(random_day[1].blocks.items()))
        if len(random_block[1].start_times_surgeries) > 0:
            new_solution = copy.deepcopy(solution)
            old_times = random_block[1].start_times_surgeries
            new_times = copy.deepcopy(old_times)
            if last:
                new_times = new_times[:-1]
            else:
                random_surgery = random.choice(old_times)
                new_times.remove(random_surgery)
            new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = new_times
            mutation = True

    if spread == "spread_equally":
        new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
            spread_equally(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries,
                           slack_begin=slack_begin, slack_end=slack_end)
    elif spread == "spread_increasing":
        new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
            spread_increasing(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries)
    else:
        new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
            add_slack(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries,
                              slack_begin=slack_begin, slack_end=slack_end)

    new_solution.set_adjusted_day(random_day[1])
    return new_solution


def only_spread(solution, heuristic):
    """
    Heuristic that doesn't add/remove surgery but only spreads
    """
    mutation = False
    while not mutation:
        random_day = random.choice(list(solution.days.items()))
        random_block = random.choice(list(random_day[1].blocks.items()))
        if len(random_block[1].start_times_surgeries) > 0:
            new_solution = copy.deepcopy(solution)

            if heuristic == "spread_increasing":
                new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
                    spread_increasing(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries)
            elif heuristic == "slack_begin":
                new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
                    add_slack(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries,
                              slack_begin=True, slack_end=False)

            elif heuristic == "slack_end":
                new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
                    add_slack(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries,
                              slack_begin=False, slack_end=True)

            elif heuristic == "shift_left":
                """
                To do implement shift to the left
                """
                new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
                    shift(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries, "left")

            elif heuristic == "shift_right":
                """
                To do implement shift to the left
                """
                new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
                    shift(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries, "right")
            mutation = True

    new_solution.set_adjusted_day(random_day[1])
    return new_solution


def add_one_random(solution, instance_info, spread=None, slack_begin=False, slack_end=False):
    """
    :param solution: Solution
    :param instance_info: InstanceInfo
    :param spread: Boolean
    :return: Solution
    """
    mutation = False
    while not mutation:
        random_day = random.choice(list(solution.days.items()))
        random_block = random.choice(list(random_day[1].blocks.items()))
        random_start_time = random.randint(0, 8 * 60)
        if not (random_start_time in solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries):
            new_solution = copy.deepcopy(solution)
            new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries.append(random_start_time)
            new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries.sort()
            new_solution.count_surgeries()
            if check_feasibility(new_solution, instance_info):
                mutation = True

    if spread == "spread_equally":
        new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
            spread_equally(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries,
                           slack_begin=slack_begin, slack_end=slack_end)
    elif "spread_increasing":
        new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
            spread_increasing(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries)
    else:
        new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries = \
            add_slack(new_solution.days[random_day[0]].blocks[random_block[0]].start_times_surgeries,
                              slack_begin=slack_begin, slack_end=slack_end)

    new_solution.set_adjusted_day(random_day[1])
    return new_solution


def remove_one(solution, day, block, surgery):
    """
    :param day: String
    :param block: Integer
    :param surgery: Integer (index)
    :return: Solution
    """
    new_solution = copy.deepcopy(solution)
    old_times = new_solution.days[day].blocks[block].start_times_surgeries
    new_times = copy.deepcopy(old_times)
    new_times.pop(surgery)
    new_solution.days[day].blocks[block].start_times_surgeries = new_times
    new_solution.count_surgeries()

    new_solution.set_adjusted_day(new_solution.days[day])
    return new_solution


def add_one(solution, day, block, instance_info):
    """
    :param day: String
    :param block: Integer
    :return: Solution
    """
    mutation = False
    while not mutation:
        random_start_time = random.randint(0, 8 * 60)
        if not (random_start_time in solution.days[day].blocks[block].start_times_surgeries):
            new_solution = copy.deepcopy(solution)
            new_solution.days[day].blocks[block].start_times_surgeries.append(random_start_time)
            new_solution.days[day].blocks[block].start_times_surgeries.sort()
            new_solution.count_surgeries()
            mutation = True
    if check_feasibility(new_solution, instance_info):
        new_solution.count_surgeries()
        new_solution.set_adjusted_day(new_solution.days[day])
        return new_solution
    else:
        return None

def swap_blocks(solution, instance_info):
    """
    :param solution: Solution
    :param instance_info: InstanceInfo
    :return: new_solution: Solution

    Swaps to blocks within one solution
    """
    mutation = False
    while not mutation:
        random_day1 = random.choice(list(solution.days.items()))
        random_block1 = random.choice(list(random_day1[1].blocks.items()))
        random_day2 = random.choice(list(solution.days.items()))
        random_block2 = random.choice(list(random_day2[1].blocks.items()))
        if not random_day1 == random_day2:
            new_solution = copy.deepcopy(solution)
            print("candidate before swap")
            print("1", solution.days[random_day1[0]].blocks[random_block1[0]].start_times_surgeries)
            print("2", solution.days[random_day2[0]].blocks[random_block2[0]].start_times_surgeries)
            new_solution.days[random_day1[0]].blocks[random_block1[0]].start_times_surgeries = copy.deepcopy(solution.days[random_day2[0]].blocks[random_block2[0]].start_times_surgeries)
            new_solution.days[random_day2[0]].blocks[random_block2[0]].start_times_surgeries = copy.deepcopy(solution.days[random_day1[0]].blocks[random_block1[0]].start_times_surgeries)
            new_solution.count_surgeries()
            if check_feasibility(new_solution, instance_info):
                print("after swap")
                print("1", new_solution.days[random_day1[0]].blocks[random_block1[0]].start_times_surgeries)
                print("2", new_solution.days[random_day2[0]].blocks[random_block2[0]].start_times_surgeries)
                mutation = True
            else:
                print("swap was infeasible")
    return new_solution