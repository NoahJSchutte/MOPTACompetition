from json.encoder import INFINITY
from gurobipy import *
import pandas as pd
from src.general.Functions.instance_configuration import instance_configuration
from src.general.Functions.solution_functions import create_deterministic_solution

class Deterministic():
    def __init__(
            self,
            instance_size
    ):
        self.BLOCKS = []
        self.TYPE_PER_BLOCK = []
        self.DAY_PER_BLOCK = []
        self.ROOM_PER_BLOCK = []
        self.TYPES = ["CARD", "ORTH", "GASTRO", "URO", "MED", "GYN"]
        self.NR_SURGERIES = {}
        self.model = Model("DeterministicScheduling")
        self.instance_size = instance_size
        self.instance_info = instance_configuration(instance_size, costs_csv="data/input/Costs_1.csv",
                                               emergency_arrival_rate=4)
        self.DURATIONS = dict()
        for name, block_type in self.instance_info.get_block_types().items():
            self.DURATIONS[name] = block_type.get_mean_duration()

    def read_input_blocks(self, blocks_csv):
        blocks_input = pd.read_csv(blocks_csv, sep=";")
        self.BLOCKS = [int(b) for b in blocks_input["BLOCK"].tolist()]
        self.TYPE_PER_BLOCK = blocks_input["TYPE"].tolist()
        self.DAY_PER_BLOCK = blocks_input["DAY"].tolist()
        self.ROOM_PER_BLOCK = blocks_input["ROOM"].tolist()
        for block in self.instance_info.block_types:
            self.NR_SURGERIES[block] = self.instance_info.block_types[block].nr_of_surgeries

    def build_model(self, csv_output):

        # Sets
        BT = set([(b, t) for b in self.BLOCKS for t in self.TYPES])

        # Parameters
        # b[i,block] = 1 if surgery i can be scheduled in block (same type)
        a = {}
        for t in self.TYPES:
            for b in self.BLOCKS:
                if self.TYPE_PER_BLOCK[b] == t:
                    a[b, t] = 1
                else:
                    a[b, t] = 0

        x = self.model.addVars(BT, vtype=GRB.INTEGER, name='x')
        z = self.model.addVar(obj=1, vtype=GRB.INTEGER, name='z', lb=0)

        self.model.addConstrs((x[b, t] <= a[b, t] * 500 for (b, t) in BT), 'blocksameType')

        self.model.addConstrs((quicksum(x[b, t] for b in self.BLOCKS) <= self.NR_SURGERIES[t] for t in self.TYPES), 'nrSurgeries')
        self.model.addConstrs((x[b, t] * self.DURATIONS[t] <= 480 for (b, t) in BT), 'capacity')

        self.model.addConstr((self.instance_size - quicksum([x[b, t] for (b, t) in BT]) == z), 'objectiveConstraint')

        # Optimize
        self.model.update()
        # self.model.write('SimpleDeterministic.lp')
        self.model.optimize()

        self.write_results(x, csv_output)

    def write_results(self, x, csv_output):
        x_m = dict(self.model.getAttr('X', x))
        assigned = 0

        block_assignment = {}
        for block in self.BLOCKS:
            block_assignment[block] = 0

        for block in self.BLOCKS:
            for u in x_m.keys():
                if u[0] == block:
                    if x_m[u] > 0.5:
                        print(f'Block {u[0]} of type {u[1]} has {x_m[u]} surgeries assigned')
                        block_assignment[u[0]] = x_m[u]
                        assigned += x_m[u]

        print(f'There are in total {assigned} surgeries assigned')

        solution = create_deterministic_solution(block_assignment, instance_info=self.instance_info, durations=self.DURATIONS)
        solution.save_solution(csv_output)

        resultAttributes = {
            "SolvedOptimally": False,
            "SolutionCount": 0,
            "ObjectiveBound": INFINITY,
            "SolveTime": INFINITY,
            "TimeOut": False
        }
        if self.model.status == GRB.OPTIMAL:
            resultAttributes["SolvedOptimally"] = True
        if self.model.status == GRB.TIME_LIMIT:
            resultAttributes["TimeOut"] = True
        resultAttributes["SolutionCount"] = self.model.SolCount
        resultAttributes["ObjectiveBound"] = self.model.ObjBound
        resultAttributes["SolveTime"] = self.model.RunTime

        attr_output = csv_output.split(".csv", 1)[0] + "attr.csv"
        attributes = pd.DataFrame([resultAttributes])
        attributes.to_csv(attr_output, index=False)
