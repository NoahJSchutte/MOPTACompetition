import copy
import random

def cross_over(parent1, parent2, mix):
    """
    :param parent1: Solution
    :param parent2: Solution
    :param mix: float
    :return child: Solution
    """
    days = list(parent1.days.items())
    # Initialize child by copying parent1
    child = copy.deepcopy(parent1)
    for day in days:
        blocks = list(day[1].blocks.items())
        for block in blocks:
            if random.random() > mix:
                # Change block assignment with parent2 assignment
                child.days[day[0]].blocks[block[0]] = copy.deepcopy(parent2.days[day[0]].blocks[block[0]])
    return child
