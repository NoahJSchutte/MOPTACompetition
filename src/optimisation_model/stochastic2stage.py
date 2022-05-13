from json.encoder import INFINITY
from tarfile import BLOCKSIZE
from gurobipy import *
import pandas as pd
from src.optimisation_model.algorithm import Algorithm
from src.general.Classes.scenario_pool import ScenarioPool
from src.general.Classes.scenario import Scenario
import ast,sys,csv


class Stochastic2Stage(Algorithm):
    def __init__(
            self,
    ):
        super().__init__()
        stage_models = {}
        self.SCENARIOS = []
        self.COSTS = {}
        self.SURGERY_DURATIONS = [] #nested for scenarios
        self.EMERGENCY_CAPACITY = []
        self.EMERGENCY_SURGERIES = [] #assume large emergency surgeries 100
        self.EMERGENCY_SURGERY_DURATIONS = [] #nested for scenarios
        self.EMERGENCY_SURGERY_ARRIVALS = [] #nested for scenarios
        self.emergency_arrival_map = []
        self.emergency_duration_map = []
        # self.surgery_assignment = {}
        self.node_file_start = 0
        self.node_dir = ''
        self.threads = 0
        self.symmetry_breaking = True



    def read_input(self,csv_blocks,csv_costs,surgery_instance,scenarios,):
        self.read_input_blocks(csv_blocks)
        self.convert_instance_info(surgery_instance,scenarios)
        self.convert_scenario_pool(scenarios)
        self.read_costs(csv_costs)

    def read_input_csv(self,csv_blocks,csv_costs,csv_surgeries,csv_emergency_scenarios,):
        self.read_input_blocks(csv_blocks)
        self.read_input_surgeries(csv_surgeries)
        self.read_scenarios(csv_emergency_scenarios)
        self.read_costs(csv_costs)

    def read_input_surgeries(self, csv_surgeries):
        #TO DO match handling of emergency scenarios
        surgeries_input = pd.read_csv(csv_surgeries, sep=";")
        self.SURGERIES = set([int(i) for i in surgeries_input["SURGERY"].tolist()])
        self.SURGERY_TYPES = surgeries_input["TYPE"].tolist()
        self.SURGERY_DURATIONS = [ast.literal_eval(str(j)) for j in surgeries_input["DURATION"].tolist()] #tuple of all scenarios

    def read_scenarios(self, csv_emergency_scenarios):
        emergency_input = pd.read_csv(csv_emergency_scenarios, sep=";")   
        
        self.EMERGENCY_SURGERIES = [f'E{i}' for i in range(50)]
        DURATIONS = [ast.literal_eval(value) for value in emergency_input["Duration"]]
        ARRIVALS = [ast.literal_eval(value) for value in emergency_input["ArrivalTime"]]
        self.SCENARIOS = [i for i in range(len(DURATIONS))] 

        for index in self.SCENARIOS:
            start = 0
            self.EMERGENCY_SURGERY_DURATIONS.append({})
            self.EMERGENCY_SURGERY_ARRIVALS.append({})
            for day in DURATIONS[index]:
                self.EMERGENCY_SURGERY_DURATIONS[index][day] = [0 for i in range(len(self.EMERGENCY_SURGERIES))]
                self.EMERGENCY_SURGERY_ARRIVALS[index][day] = [0 for i in range(len(self.EMERGENCY_SURGERIES))]

                for j,value in enumerate(DURATIONS[index][day]):
                    self.EMERGENCY_SURGERY_DURATIONS[index][day][j+start] = value
                for j,value in enumerate(ARRIVALS[index][day]):
                    self.EMERGENCY_SURGERY_ARRIVALS[index][day][j+start] = value

                start = start + len(DURATIONS[index][day])
        #TO DO:add sanity check that all scenarios filled equally     

    def convert_instance_info(self,instance_info,scenario_pool):
        i = 0 #surgery index
        self.SURGERIES = []
        self.SURGERY_TYPES = []
        self.SURGERY_DURATIONS = []
        for type in instance_info.block_types:
            j = 0
            for surgery in range(instance_info.block_types[type].nr_of_surgeries):
                self.SURGERIES.append(i)
                self.SURGERY_TYPES.append(type)
                self.SURGERY_DURATIONS.append([scenario.elective_surgery_durations[type][j] for scenario in scenario_pool.scenarios])
                i += 1
                j += 1
        
    def convert_scenario_pool(self, scenario_pool):
        self.SCENARIOS = [i for i in range(len(scenario_pool.scenarios))]
        self.EMERGENCY_SURGERIES = [f'E{i}' for i in range(50)]
        self.EMERGENCY_CAPACITY = [[0 for b in self.BLOCKS] for scenario in scenario_pool.scenarios]
        for index,scenario in enumerate(scenario_pool.scenarios):
            total = 0
            for day in scenario.emergency_surgery_durations:
                total = sum(scenario.emergency_surgery_durations[day])
                for i in range(len(self.BLOCKS)):
                    if self.DAY_PER_BLOCK[i]==day:
                        self.EMERGENCY_CAPACITY[index][i] = total/len([i for i in range(len(self.BLOCKS)) if self.DAY_PER_BLOCK[i]==day])
        
        for index,scenario in enumerate(scenario_pool.scenarios):
            start = 0
            self.EMERGENCY_SURGERY_DURATIONS.append({})
            self.EMERGENCY_SURGERY_ARRIVALS.append({})

            for day in scenario.emergency_surgery_durations:
                self.EMERGENCY_SURGERY_DURATIONS[index][day] = [0 for i in range(len(self.EMERGENCY_SURGERIES))]
                self.EMERGENCY_SURGERY_ARRIVALS[index][day] = [0 for i in range(len(self.EMERGENCY_SURGERIES))]

                for j,value in enumerate(scenario.emergency_surgery_durations[day]):
                    self.EMERGENCY_SURGERY_DURATIONS[index][day][j+start] = value
                for j,value in enumerate(scenario.emergency_surgery_arrival_times[day]):
                    self.EMERGENCY_SURGERY_ARRIVALS[index][day][j+start] = value

                start = start + len(scenario.emergency_surgery_durations[day])   

    def read_costs(self, csv_costs):
        self.COSTS = pd.read_csv(csv_costs, sep=";", header=None, index_col=0, squeeze=True).to_dict()


    def build_stage1(self,timeout=INFINITY, ):
        stage_model = Model("FirstStage")
        stage_model.params.NonConvex = 1 #decide how to handle quadratic constraints
        stage_model.params.timeLimit = timeout

        if self.node_file_start > 0:
            stage_model.params.NodefileStart = self.node_file_start
        if len(self.node_dir) > 0:
            stage_model.params.NodefileDir = self.node_dir
        if self.threads > 0:
            stage_model.params.Threads = self.threads  # this is because of memory issue

        # Sets - to serve as variable indices
        B = self.BLOCKS + ["dummy"] #add dummy block
        SB = set([(i, b) for i in self.SURGERIES for b in B]) #elective surgery block indices
 
        #Decision Variables
        x = stage_model.addVars(SB, vtype=GRB.BINARY, name='x')  # x[i,b]=1 if surgery j is assigned to block b

        #Cost Variables
        CN = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CN') #scheduling
        CI = stage_model.addVars(self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS, lb=0,name='CI')
        CI_ = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CI_') #idle
        CO = stage_model.addVars(self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS,lb=0, name='CO')
        CO_ = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CO_') #overtime

        #Parameters
        b = {}  # block is of certain type
        for i in self.SURGERIES:
            b[i,"dummy"] = 1
            for block in self.BLOCKS:
                if self.TYPE_PER_BLOCK[block] == self.SURGERY_TYPES[i]:
                    b[i, block] = 1
                else:
                    b[i, block] = 0

        #Constraints (Stage 1)
        stage_model.addConstrs((quicksum(x[i, b] for b in B) == 1 for i in self.SURGERIES), '3a') # constraint maybe not necessary due to idle time, is needed because of cancellation costs
        stage_model.addConstrs((x[i, block] <= b[i, block] for i in self.SURGERIES for block in self.BLOCKS), '3b')  # adjust this constraint enforce 0s for wrong block, didn't work

        if self.symmetry_breaking:
            for type in list(set(self.SURGERY_TYPES)):
                ST = [i for i in self.SURGERIES if self.SURGERY_TYPES[i] == type]
                BT = [b for b in self.BLOCKS if self.TYPE_PER_BLOCK[b] == type]
                for si,i in enumerate(ST[:-1]):
                    for bi,b in enumerate(BT):
                        stage_model.addConstr((x[ST[si],b] - quicksum(x[ST[si+1],k] for k in BT[bi:]) - x[ST[si+1],"dummy"] <= 0), f"sym{type}{i}{b}") #symmetry breaking

        #average_capacity = quicksum(self.EMERGENCY_CAPACITY[w][b] for b in self.BLOCKS for w in self.SCENARIOS) / (len(self.BLOCKS)*len(self.SCENARIOS))
        stage_model.addConstrs((CO[b,w] - CI[b,w]
                        == quicksum(self.SURGERY_DURATIONS[i][w]*x[i,b] for i in self.SURGERIES) 
                                + self.EMERGENCY_CAPACITY[w][b] - self.LENGTH_PER_BLOCK
                                for b in self.BLOCKS for w in self.SCENARIOS),'4')
        # removing the following 3 constraints don't help
        stage_model.addConstr(CO_ == quicksum(CO[b,w] for b in self.BLOCKS for w in self.SCENARIOS) * 1/len(self.SCENARIOS))
        stage_model.addConstr(CI_ == quicksum(CI[b,w] for b in self.BLOCKS for w in self.SCENARIOS) * 1/len(self.SCENARIOS))

        #Objective
        #not scheduled cost
        stage_model.addConstr((CN == quicksum([x[i, "dummy"] for i in self.SURGERIES])), 'notscheduled')

        #overtime and idletime cost        
        stage_model.setObjective(self.COSTS["NOTSCHEDULING"]*CN
                            + self.COSTS["IDLETIME"]*CI_
                            + self.COSTS["OVERTIME"]*CO_
                            , GRB.MINIMIZE)

        # Optimize
        stage_model.update()
        stage_model.optimize()

        return {"model":stage_model,"x":x}

    def build_stage2(self,x,timeout=INFINITY):
        stage_model = Model("SecondStage")
        stage_model.params.NonConvex = 1 #decide how to handle quadratic constraints
        stage_model.params.timeLimit = timeout
        stage_model.NumStart = 1
        #stage_model.params.PreSparsify = 1  # haven't tried this yet but want to

        if self.node_file_start > 0:
            stage_model.params.NodefileStart = self.node_file_start
        if len(self.node_dir) > 0:
            stage_model.params.NodefileDir = self.node_dir
        if self.threads > 0:
            stage_model.params.Threads = self.threads  # this is because of memory issue

        # Sets - to serve as variable indices
        SB = set([(i, b) for i in self.SURGERIES for b in self.BLOCKS]) #elective surgery block indices
        SS = set([(i, j) for i in self.SURGERIES for j in self.SURGERIES]) #elective surgery pairs
        S = set([j for j in self.SURGERIES]) #elective surgeries
        E = set([j for j in self.EMERGENCY_SURGERIES]) #emergency surgeries
        W = set([j for j in self.SCENARIOS]) #all scenarios

        SA = S.union(set([j for j in self.EMERGENCY_SURGERIES]))#all surgeries
        
        SW = set([(i,j) for i in S for j in W])
        SAW = set([(i,j) for i in SA for j in W])
        SEB = set([(i, b) for i in self.EMERGENCY_SURGERIES for b in self.BLOCKS])
        SE = set([(i, j) for i in self.EMERGENCY_SURGERIES for j in self.EMERGENCY_SURGERIES])
        SAE = set([(i, j) for i in self.EMERGENCY_SURGERIES for j in self.EMERGENCY_SURGERIES]).union(
            set([(i, j) for i in self.SURGERIES for j in self.EMERGENCY_SURGERIES]),
            set([(i, j) for i in self.EMERGENCY_SURGERIES for j in self.SURGERIES])
        )

        #Decision Variables
        p = stage_model.addVars(SS, vtype=GRB.BINARY, name='p')  # p[i,j]=1 if surgery i precedes surgery j in the same block
        s = stage_model.addVars(S, vtype=GRB.CONTINUOUS, name='s')  # Start times for elective surgeries

        T = stage_model.addVars(SAW, vtype=GRB.CONTINUOUS, name='T') # Actual start times for all surgeries
        y = stage_model.addVars(self.EMERGENCY_SURGERIES, self.BLOCKS, self.SCENARIOS, vtype=GRB.BINARY, name='y') # Block assignemnt variable for emergencies
        z = stage_model.addVars(SW, vtype=GRB.BINARY, name='z') # Elective cancelled or not {0,1}
        q = stage_model.addVars(SAE,self.SCENARIOS, vtype=GRB.BINARY, name='q') #precedence constraint between surgery pairs of which one or more is an emergency
        V = stage_model.addVars(self.SURGERIES,self.EMERGENCY_SURGERIES,self.BLOCKS,self.SCENARIOS,vtype=GRB.BINARY,name='V') #second stage helpers
        VV = stage_model.addVars(self.SURGERIES,self.SURGERIES,self.SCENARIOS,vtype=GRB.BINARY,name='VV') #second stage helpers

        #Cost Variables
        CN = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CN') #not scheduling
        CW = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CW') #elective waiting
        CC = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CC') #cancelling
        CEW = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CEW') #emergency waiting
        CI = stage_model.addVars(self.SCENARIOS,vtype=GRB.CONTINUOUS, name='CI')
        CI_ = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CI_') #idle
        CO = stage_model.addVars(self.SCENARIOS,vtype=GRB.CONTINUOUS, name='CO')
        CO_ = stage_model.addVar(vtype=GRB.CONTINUOUS, name='CO_') #overtime
        coh1 = stage_model.addVars(self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS, name='coh1')
        coh2 = stage_model.addVars(self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS, name='coh2')
        coh3 = stage_model.addVars(self.SURGERIES,self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS, name='coh3')
        coh4 = stage_model.addVars(self.EMERGENCY_SURGERIES,self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS, name='coh4')
        coh5 = stage_model.addVars(self.BLOCKS,self.SCENARIOS,vtype=GRB.CONTINUOUS, name='coh5')
        cih = stage_model.addVars(self.BLOCKS,self.SCENARIOS, vtype=GRB.CONTINUOUS, name='cih')

        #Parameters
        b = {}
        for i in self.SURGERIES:
            for block in self.BLOCKS:
                if self.TYPE_PER_BLOCK[block] == self.SURGERY_TYPES[i]:
                    b[i, block] = 1
                else:
                    b[i, block] = 0

        self.emergency_arrival_map = {"TIME":{i:[0 for w in self.SCENARIOS] for i in self.EMERGENCY_SURGERIES }, "DAY":{i:["Monday" for w in self.SCENARIOS] for i in self.EMERGENCY_SURGERIES}} 
        self.emergency_duration_map = {i:[0 for w in self.SCENARIOS] for i in self.EMERGENCY_SURGERIES}
        for w in self.SCENARIOS:
            for day in self.EMERGENCY_SURGERY_DURATIONS[w]:
                for index,i in enumerate(self.EMERGENCY_SURGERIES):
                    if self.EMERGENCY_SURGERY_DURATIONS[w][day][index] > 0:
                        self.emergency_arrival_map["TIME"][i][w] = int(self.EMERGENCY_SURGERY_ARRIVALS[w][day][index])
                        self.emergency_arrival_map["DAY"][i][w] = day
                        self.emergency_duration_map[i][w] = int(self.EMERGENCY_SURGERY_DURATIONS[w][day][index])

        h = {}
        for w in self.SCENARIOS:
            for i in self.EMERGENCY_SURGERIES:
                for block in self.BLOCKS:
                    if self.DAY_PER_BLOCK[block] == self.emergency_arrival_map["DAY"][i][w]:
                        h[i,block,w] = 1
                    else:
                        h[i,block,w] = 0

        #set some start variables - all electives cancelled
        for var in z.values():
            var.Start = 1


        stage_model.addConstrs(
            (p[i, j] + p[j, i] == quicksum([x[i, b] * x[j, b] for b in self.BLOCKS]) for i in self.SURGERIES for j in self.SURGERIES if
             i != j),'1.1d')

        stage_model.addConstrs((s[j] >= p[i, j] * s[i] for i in self.SURGERIES for j in self.SURGERIES), '1.1e')

        #Constraints (Stage 2)
        stage_model.addConstrs((T[i,w] >= s[i] for i in self.SURGERIES for w in self.SCENARIOS), '1.2a')

        stage_model.addConstrs((T[i,w] >= self.emergency_arrival_map["TIME"][i][w] for i in self.EMERGENCY_SURGERIES for w in self.SCENARIOS), '1.2b')

        
        stage_model.addConstrs((VV[i,j,w]==p[i,j]*z[i,w] for i in self.SURGERIES 
                for j in self.SURGERIES for w in self.SCENARIOS),'1.2chelper')
        stage_model.addConstrs((T[j,w] >= (T[i,w] + self.SURGERY_DURATIONS[i][w])*(p[i,j] - VV[i,j,w])
                for i in self.SURGERIES for j in self.SURGERIES for w in self.SCENARIOS if i!=j), '1.2c') 

        stage_model.addConstrs((T[j,w] >= (T[i,w] + self.emergency_duration_map[i][w])*q[i,j,w]
        for i in self.EMERGENCY_SURGERIES for j in self.EMERGENCY_SURGERIES for w in self.SCENARIOS if i!=j), '1.2da')

        stage_model.addConstrs((T[j,w] >= (T[i,w] + self.emergency_duration_map[i][w])*q[i,j,w]
        for i in self.EMERGENCY_SURGERIES for j in self.SURGERIES for w in self.SCENARIOS if i!=j), '1.2db')

        stage_model.addConstrs((T[j,w] >= (T[i,w] + self.SURGERY_DURATIONS[i][w])*q[i,j,w]
        for i in self.SURGERIES for j in self.EMERGENCY_SURGERIES for w in self.SCENARIOS if i!=j), '1.2dc')

        stage_model.addConstrs((quicksum(y[i,b,w] for b in self.BLOCKS) == 1 for i in self.EMERGENCY_SURGERIES for w in self.SCENARIOS), '1.2ea')
        stage_model.addConstrs((y[i,b,w] <= h[i,b,w] for i in self.EMERGENCY_SURGERIES for b in self.BLOCKS for w in self.SCENARIOS), '1.2eb')


        stage_model.addConstrs((V[i,j,b,w]==x[i,b]*y[j,b,w] for i in self.SURGERIES 
                for j in self.EMERGENCY_SURGERIES for b in self.BLOCKS for w in self.SCENARIOS),'1.2fhelper')
        stage_model.addConstrs((q[i,j,w]+q[j,i,w] == V[i,j,b,w]*(1-z[i,w]) for i in self.SURGERIES 
                for j in self.EMERGENCY_SURGERIES for b in self.BLOCKS for w in self.SCENARIOS), '1.2f')

        #1.2g unimplemented
        #1.2h unimplemented

        #Objective
        #not scheduled cost
        stage_model.addConstr((CN == quicksum([1 - quicksum([x[i, b] for b in self.BLOCKS]) for i in self.SURGERIES])), 'notscheduled') 

        #elective waiting cost
        stage_model.addConstr((CW == quicksum(T[i,w] - s[i] for i in self.SURGERIES for w in self.SCENARIOS) * 1/(len(self.SCENARIOS))), 'waiting') 

        #cancellation cost
        stage_model.addConstr((CC == quicksum(z[i,w] for i in self.SURGERIES for w in self.SCENARIOS) * 1/(len(self.SCENARIOS))), 'cancellation') 

        #emergency waiting cost
        stage_model.addConstr((CEW == quicksum(T[i,w] - self.emergency_arrival_map["TIME"][i][w] for i in self.EMERGENCY_SURGERIES for w in self.SCENARIOS)
        * 1/(len(self.SCENARIOS))), 'emergencywaiting') 

        #overtime cost
        stage_model.addConstrs((coh3[i,b,w] == x[i,b]*(T[i,w]+self.SURGERY_DURATIONS[i][w]) for i in self.SURGERIES
                    for w in self.SCENARIOS for b in self.BLOCKS), "overTimeHelper1")
        stage_model.addConstrs((coh1[b,w] == max_(coh3[i,b,w] for i in self.SURGERIES) 
                    for w in self.SCENARIOS for b in self.BLOCKS), 
                                "overTimeHelper2")
                                
        stage_model.addConstrs((coh4[j,b,w] == y[j,b,w]*(T[j,w]+self.emergency_duration_map[j][w])  for j in self.EMERGENCY_SURGERIES
                    for w in self.SCENARIOS for b in self.BLOCKS), "overTimeHelper3")     

        stage_model.addConstrs((coh2[b,w] == max_(coh4[j,b,w] for j in self.EMERGENCY_SURGERIES) 
                    for w in self.SCENARIOS for b in self.BLOCKS), 
                                "overTimeHelper4")

        stage_model.addConstrs((coh5[b,w] == max_(
                    coh1[b,w],
                    coh2[b,w],
                    self.LENGTH_PER_BLOCK
                    ) for b in self.BLOCKS for w in self.SCENARIOS)
            ,"overTimeHelper5"
        )
        
        stage_model.addConstrs((CO[w] == quicksum(coh5[b,w] - self.LENGTH_PER_BLOCK for b in self.BLOCKS) 
                        for w in self.SCENARIOS) ,"overtime")

        stage_model.addConstr(CO_ == quicksum(CO[w] for w in self.SCENARIOS) * 1/(len(self.SCENARIOS)))            
        
        #idletime cost
        stage_model.addConstrs((cih[b,w] == coh5[b,w] - 
                        quicksum([x[i,b]*self.SURGERY_DURATIONS[i][w] for i in self.SURGERIES]) -
                        quicksum([y[j,b,w]*self.emergency_duration_map[j][w] for j in self.EMERGENCY_SURGERIES])
                        for b in self.BLOCKS for w in self.SCENARIOS),"idletimehelper")

        stage_model.addConstrs((CI[w]  == quicksum(cih[b,w] for b in self.BLOCKS) for w in self.SCENARIOS),"idletime")

        stage_model.addConstr(CI_ == quicksum(CI[w] for w in self.SCENARIOS) * 1/(len(self.SCENARIOS)))
        
        stage_model.setObjective(self.COSTS["NOTSCHEDULING"]*CN 
                            + self.COSTS["ELECTIVEWAITINGTIME"]*CW 
                            + self.COSTS["CANCELLING"]*CC 
                            + self.COSTS["EMERGENCYWAITINGTIME"]*CEW  
                            + self.COSTS["IDLETIME"]*CI_ 
                            + self.COSTS["OVERTIME"]*CO_ 
                            , GRB.MINIMIZE)

        # Optimize
        stage_model.update()
        stage_model.write('Scheduling.lp')
        stage_model.optimize()

        if stage_model.status == GRB.INFEASIBLE:
            print("Infeasible model. Review!")
            sys.exit()

        return {"model":stage_model,"y":y,"q":q,"s":s,"z":z,"T":T}
            

    def build_model(self, csv_output, timeout=[INFINITY], threads=0, node_file_start=0, node_dir=''):
        self.threads = threads
        self.node_file_start = node_file_start
        self.node_dir = node_dir

        stage_results = {}
        stage_results[1] = self.build_stage1(timeout[0])
        x = dict(stage_results[1]["model"].getAttr('X',stage_results[1]["x"]))
        x = {key:1 if value>=0.5 else 0 for key,value in x.items()}
        
        stage_results[2]  = self.build_stage2(x,timeout[1])

        s = {(i):"-" for i in self.SURGERIES}
        T = {(i,w):"-" for i in (set(self.SURGERIES).union(set(self.EMERGENCY_SURGERIES))) for w in self.SCENARIOS}
        y = {(i,b,w):0 for i in self.EMERGENCY_SURGERIES for b in self.BLOCKS for w in self.SCENARIOS}

        if stage_results[2]["model"].SolCount > 0:
            y = dict(stage_results[2]["model"].getAttr('X',stage_results[2]["y"]))
            y = {key:1 if value>=0.5 else 0 for key,value in y.items()}

            z = dict(stage_results[2]["model"].getAttr('X',stage_results[2]["z"]))
            z = {key:1 if value>=0.5 else 0 for key,value in z.items()}

            for i in self.SURGERIES:
                for b in self.BLOCKS:
                    if x[i,b]:
                        s[i] = stage_results[2]["s"][i].X
                        for w in self.SCENARIOS:
                            T[i,w] = stage_results[2]["T"][i,w].X

            for i in self.EMERGENCY_SURGERIES:
                for w in self.SCENARIOS:
                    for b in self.BLOCKS:
                        if y[i,b,w] and any([self.emergency_duration_map[i][v]>0 for v in self.SCENARIOS]):
                            T[i,w] = stage_results[2]["T"][i,w].X
        self.write_results(stage_results,x,s,y,z,T,csv_output)


    def write_results(self,stage_results,x,s,y,z,T,csv_output):            
        results = []
        assigned = 0
        for i in self.SURGERIES:
            block = [b for b in self.BLOCKS if x[i,b]]
            if len(block) > 1:
                print("Model assigned surgery to multiple blocks")
                sys.exit()
            elif len(block) == 0:
                pass
            else:
                block = block[0]
                assigned += 1
                results.append({"Surgery": i, "Block": block, "Start": s[i], "Real Start": [T[(i,w)] for w in self.SCENARIOS], 
                                        "Status": [z[(i,w)] for w in self.SCENARIOS],
                                        "Specialty": self.SURGERY_TYPES[i], 
                                        "BlockType": self.TYPE_PER_BLOCK[block], 
                                        "Arrival Day": "-",
                                        "Arrival Time": "-",
                                        "Day": self.DAY_PER_BLOCK[block], 
                                        "Room": self.ROOM_PER_BLOCK[block]})
        for i in self.EMERGENCY_SURGERIES:
            if not all(self.emergency_duration_map[i][w] == 0 for w in self.SCENARIOS):
                block = [[b for b in self.BLOCKS if y[i,b,w]>=0.5] for w in self.SCENARIOS]
                for w in self.SCENARIOS:
                    if len(block[w]) != 1:
                        print("Model assigned surgery to multiple/no blocks")
                        sys.exit()
            
                results.append({"Surgery": i, "Block": [block[w][0] for w in self.SCENARIOS], "Start": "-", "Real Start": [T[(i,w)] for w in self.SCENARIOS], "Status": "-",
                                            "Specialty": "-", 
                                            "BlockType": [self.TYPE_PER_BLOCK[block[w][0]] for w in self.SCENARIOS], 
                                            "Arrival Day": [self.emergency_arrival_map["DAY"][i][w] for w in self.SCENARIOS],
                                            "Day": [self.DAY_PER_BLOCK[block[w][0]] for w in self.SCENARIOS],
                                            "Arrival Time": [self.emergency_arrival_map["TIME"][i][w] for w in self.SCENARIOS],
                                            "Room": [self.ROOM_PER_BLOCK[block[w][0]] for w in self.SCENARIOS]})
        schedule = pd.DataFrame(results)
        schedule.to_csv(csv_output, index=False)

        resultAttributes = []
        for m in stage_results:
            model = stage_results[m]["model"]
            resultAttributes.append({"Model": model.ModelName,
            "SolvedOptimally": True if model.status == GRB.OPTIMAL else False,
            "SolutionCount": model.SolCount,
            "ObjectiveBound": model.ObjBound,
            "SolveTime": model.RunTime,
            "TimeOut": True if model.status == GRB.TIME_LIMIT else False,
            "OptimalityGap": model.MIPGap,
            "BestObjective":model.ObjVal,
            })           

        attributes = pd.DataFrame(resultAttributes)
        attr_output = csv_output.split(".csv",1)[0] + "attr.csv"
        attributes.to_csv(attr_output, index=False)


        


