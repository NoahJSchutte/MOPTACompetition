import os
import sys, random
p = os.path.abspath('.')
sys.path.insert(1, p)

from json import tool
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * #QIcon, QFileDialog
from PyQt5 import QtCore 

from src.general.Functions.solution_functions import convert_solution, draw_solution_raw
from src.general.Classes.simulator import Simulator
from src.general.Functions.instance_configuration import instance_configuration
from src.general.Functions.general_functions import load_object
from src.run import runALNS
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import threading

class EmergencySurgery:
    def __init__(
        self,
        arrivalDay,
        arrivalTime,
        Duration
    ):
        self.arrivalTime = arrivalTime
        self.arrivalDay = arrivalDay
        self.Duration = Duration

class ScheduleWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Schedule")
        self.solution = []
        self.scenario = []
        self.simulator = []
        self.setLayout(layout)
        self.tableWidget = QTableWidget()
        self.saveButton = QPushButton("Save Schedule")
        self.saveButton.clicked.connect(self.onSave)
        self.is_simulation = False

        layout.addWidget(self.saveButton)
        layout.addWidget(self.tableWidget)
        
        
    def onShow(self):
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(1000)
        self.tableWidget.setColumnCount(10)

        self.tableWidget.setItem(0, 0 , QTableWidgetItem("Surgery ID"))
        self.tableWidget.setItem(0, 1 , QTableWidgetItem("Block Type"))
        self.tableWidget.setItem(0, 2 , QTableWidgetItem("Block ID"))
        self.tableWidget.setItem(0, 3 , QTableWidgetItem("Day"))
        self.tableWidget.setItem(0, 4 , QTableWidgetItem("Scheduled start time"))
        self.tableWidget.setItem(0, 5 , QTableWidgetItem("Realized start time"))
        self.tableWidget.setItem(0, 6 , QTableWidgetItem("Emergency"))

        if self.is_simulation:
            i = 1
            for day in self.simulator.sim_blocks:
                for b in self.simulator.sim_blocks[day]:
                    for surgery in [surgery for surgery in b.surgeries]:
                        self.tableWidget.setItem(i, 0 , QTableWidgetItem(str(i)))
                        self.tableWidget.setItem(i, 1 , QTableWidgetItem(str(b.block.block_type.name)))
                        self.tableWidget.setItem(i, 2 , QTableWidgetItem(str(b.block.key)))
                        self.tableWidget.setItem(i, 3 , QTableWidgetItem(str(day)))
                        self.tableWidget.setItem(i, 4 , QTableWidgetItem(str(surgery.scheduled_start_time)))
                        self.tableWidget.setItem(i, 5 , QTableWidgetItem(str(surgery.realized_start_time)))
                        self.tableWidget.setItem(i, 6 , QTableWidgetItem(str(surgery.is_emergency)))
                        i+=1
        else:
            i = 1
            for day in self.solution.days:
                for b in self.solution.days[day].blocks:
                    for k,start_time in enumerate(self.solution.days[day].blocks[b].start_times_surgeries):
                        self.tableWidget.setItem(i, 0 , QTableWidgetItem(str(i)))
                        self.tableWidget.setItem(i, 1 , QTableWidgetItem(str(self.solution.days[day].blocks[b].block_type.name)))
                        self.tableWidget.setItem(i, 2 , QTableWidgetItem(str(b)))
                        self.tableWidget.setItem(i, 3 , QTableWidgetItem(str(day)))
                        self.tableWidget.setItem(i, 4 , QTableWidgetItem(str(start_time)))
                        i+=1
        self.show()

    def onSave(self):
        name = QFileDialog.getSaveFileName(self, 'Save File',directory = 'schedule.csv')
        self.solution.save_solution(name[0])

class ScheduleAttributeWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Schedule Attributes")
        self.solution = []
        self.scenario = []
        self.simulator = []
        self.setLayout(layout)
        self.tableWidget = QTableWidget()
        self.saveButton = QPushButton("Save Attributes")
        self.saveButton.clicked.connect(self.onSave)
        self.is_simulation = False
        self.objective = 0
        self.info = {}

        layout.addWidget(self.saveButton)
        layout.addWidget(self.tableWidget)
        
        
    def onShow(self):
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(1000)
        self.tableWidget.setColumnCount(10)
        i = 0
        for cost in self.info:
            self.tableWidget.setItem(i, 0 , QTableWidgetItem(str(cost)))
            self.tableWidget.setItem(i, 1 , QTableWidgetItem(str(self.info[cost])))
            i+=1    
        self.show()

    def onSave(self):
        name = QFileDialog.getSaveFileName(self, 'Save File',directory = 'schedule.csv')
        self.solution.save_solution(name[0])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Team Uncertainity Delft Surgery Scheduler")
        self.resize(500,500)

        self.runningLabel = QLabel("Running...")
        self.runningLabel.hide()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)
        run_button = QAction(QIcon("src/gui/play.png"),"Run Algorithm", self)
        run_button.setStatusTip("Running algorithm please wait...")
        run_button.triggered.connect(self.runningLabel.show)
        run_button.triggered.connect(self.onRun)
        reset_button = QAction(QIcon("src/gui/reset.png"),"Reset", self)
        reset_button.setStatusTip("Resetting window")
        reset_button.triggered.connect(self.onReset)
        load_button = QAction(QIcon("src/gui/upload.png"),"Load Existing Solution", self)
        load_button.setStatusTip("Loading saved solution")
        load_button.triggered.connect(self.onLoad)

        toolbar.addAction(run_button)
        toolbar.addAction(load_button)
        toolbar.addAction(reset_button)
        toolbar.setIconSize(QtCore.QSize(100, 50))
        

        self.figure = Figure()
        self.ax = self.figure.subplots()
        self.schedulePlot = FigureCanvasQTAgg(self.figure)
        self.schedulePlot.hide()

        self.day_order = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4}
        self.instance_info = None

        inputForm = QFormLayout()  

        self.instance = Input('Instance Size',['70','100','140','200'])
        instanceLabel = QLabel('Instance Size')
        inputForm.addRow(instanceLabel, self.instance)

        self.policy = Input('Emergency Policy',['log-normal', 'exponential', 'deterministic', 'greedy','random'])
        policyLabel = QLabel('Emergency Policy')
        inputForm.addRow(policyLabel, self.policy)

        self.nr_scenarios = 100
        self.scenario_pool = load_object('data/pools/Training', f'n{self.instance.currentText()}_s{self.nr_scenarios}_e4_dF')
        self.scenario = self.scenario_pool.scenarios[random.randint(0,self.nr_scenarios-1)]
        self.emergencies = []
        for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
            self.scenario.emergency_surgery_arrival_times[day].sort()
            self.scenario.emergency_surgery_durations[day].sort()
            for k,val in enumerate(self.scenario.emergency_surgery_arrival_times[day]):
                self.emergencies.append(EmergencySurgery(day,
                                        self.scenario.emergency_surgery_arrival_times[day][k],
                                        self.scenario.emergency_surgery_durations[day][k]))

        # sort emergencies
        self.currentEmergency = 0

        self.simulateButton = QPushButton("Simulate Next Emergency")
        self.simulateButton.clicked.connect(self.onSimulate)
        self.simulateButton.hide()

        self.scheduleWindow = ScheduleWindow()
        self.showSchedule = QPushButton("Show Schedule")
        self.showSchedule.clicked.connect(self.scheduleWindow.onShow)
        self.showSchedule.hide()

        self.scheduleAttributeWindow = ScheduleAttributeWindow()
        self.showScheduleAttribute = QPushButton("Show Schedule Attributes")
        self.showScheduleAttribute.clicked.connect(self.scheduleAttributeWindow.onShow)
        self.showScheduleAttribute.hide()

        layoutSched = QHBoxLayout()
        layoutSched.addWidget(self.showSchedule)
        layoutSched.addWidget(self.showScheduleAttribute)

        layout = QVBoxLayout()
        layout.addLayout(inputForm)
        layout.addWidget(self.runningLabel)
        layout.addWidget(self.simulateButton)
        layout.addWidget(self.schedulePlot)
        layout.addLayout(layoutSched)
          
        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def onRun(self, s):
        self.instance_info = instance_configuration(int(self.instance.currentText()))
        self.scenario_pool = load_object('data/pools/Training', f'n{self.instance.currentText()}_s{self.nr_scenarios}_e4_dF')
        self.scenario = self.scenario_pool.scenarios[random.randint(0,self.nr_scenarios-1)]
        solution, attributes = runALNS(int(self.instance.currentText()),"exponential")
        self.plotSolution(solution)
        self.scheduleWindow.solution = solution
        self.scheduleAttributeWindow.solution = solution
        self.showSchedule.show()
        self.showScheduleAttribute.show()

        self.simulateButton.show()
        self.runningLabel.hide()

    def onLoad(self, s):
        self.instance_info = instance_configuration(int(self.instance.currentText()))
        self.scenario_pool = load_object('data/pools/Training', f'n{self.instance.currentText()}_s{self.nr_scenarios}_e4_dF')
        self.scenario = self.scenario_pool.scenarios[random.randint(0,self.nr_scenarios-1)]
        solution = convert_solution(
            f"data/solutions/ALNS/n={self.instance.currentText()}_e=4_c=1.csv",
            self.instance_info.block_types)
        self.plotSolution(solution)
        self.scheduleWindow.solution = solution
        self.scheduleAttributeWindow.solution = solution
        self.showSchedule.show()
        self.showScheduleAttribute.show()

        self.simulateButton.show()

    def onReset(self):
        self.scheduleWindow.is_simulation = False
        self.scheduleAttributeWindow.is_simulation = False
        self.scheduleWindow.simulator = []
        self.scheduleAttributeWindow.simulator = []
        self.scheduleWindow.solution = []
        self.scheduleAttributeWindow.solution = []
        self.scheduleAttributeWindow.info = {}


        self.scenario = self.scenario_pool.scenarios[random.randint(0,self.nr_scenarios-1)]
        self.currentEmergency = 0
        self.simulateButton.hide()
        self.showSchedule.hide()
        self.showScheduleAttribute.hide()
        self.ax.clear()
        self.schedulePlot.hide()


    def onSimulate(self,s):     
        self.scenario.reset_emergencies()
        self.currentEmergency+=1

        for emergency in self.emergencies[0:self.currentEmergency]:
            self.scenario.emergency_surgery_arrival_times[emergency.arrivalDay].append(emergency.arrivalTime)
            self.scenario.emergency_surgery_durations[emergency.arrivalDay].append(emergency.Duration)

        self.instance_info = instance_configuration(int(self.instance.currentText()))

        simulator = Simulator(self.instance_info)
        simulator.set_policy(self.policy.currentText())
        self.scheduleAttributeWindow.objective, self.scheduleAttributeWindow.info, criteria = simulator.simulate_full_solution(self.scheduleWindow.solution,self.scenario)

        self.scheduleWindow.simulator = simulator
        self.scheduleAttributeWindow.simulator = simulator
        self.scheduleWindow.is_simulation = True
        self.scheduleAttributeWindow.is_simulation = True
        self.scheduleAttributeWindow.objective, self.scheduleAttributeWindow.info

        self.plotSimulation(simulator)
 
    def plotSimulation(self,simulator):
        yticks = []
        yticklabels = []
        i=0
        for day in simulator.sim_blocks:
            j=1 #to offset plot
            for b in simulator.sim_blocks[day]:
                self.ax.broken_barh([(surgery.realized_start_time, surgery.realized_duration) for k,surgery in enumerate([surgery for surgery in b.surgeries 
                                                                                                                            if not surgery.is_emergency 
                                                                                                                            and ( (surgery.realized_start_time+surgery.realized_duration <= int(self.emergencies[self.currentEmergency-1].arrivalTime)
                                                                                                                            and self.day_order[day] ==  self.day_order[self.emergencies[self.currentEmergency-1].arrivalDay])
                                                                                                                            or (self.day_order[day] <  self.day_order[self.emergencies[self.currentEmergency-1].arrivalDay]) ) ]) ], ((i*100)+(10*j), 3), facecolors=('green'))
                self.ax.broken_barh([(surgery.realized_start_time, 5) for k,surgery in enumerate([surgery for surgery in b.surgeries if not surgery.is_emergency]) ], ((i*100)+(10*j), 3), facecolors=('blue'))
                yticks.append((i*100)+(10*j))
                yticklabels.append(f"{b.block.block_type.name}  {b.block.key}")
                j+=1
            i+=1

        i=0
        for day in simulator.sim_blocks:
            j=1 #to offset plot
            for b in simulator.sim_blocks[day]:
                self.ax.broken_barh([(surgery.realized_start_time, surgery.realized_duration) for k,surgery in enumerate([surgery for surgery in b.surgeries if surgery.is_emergency
                                                                                                                                                            and ( (surgery.realized_start_time+surgery.realized_duration <= int(self.emergencies[self.currentEmergency-1].arrivalTime)
                                                                                                                                                            and self.day_order[day] ==  self.day_order[self.emergencies[self.currentEmergency-1].arrivalDay])
                                                                                                                                                            or (self.day_order[day] <  self.day_order[self.emergencies[self.currentEmergency-1].arrivalDay]) ) ]) ], ((i*100)+(10*j), 3), facecolors=('red'))
                self.ax.broken_barh([(surgery.realized_start_time, 5) for k,surgery in enumerate([surgery for surgery in b.surgeries if surgery.is_emergency])], ((i*100)+(10*j), 3), facecolors=('black'))                                                                                                                          
                j+=1
            i+=1

        self.ax.set_xlabel('Blocks')
        yticks.extend([i*100 for i in range(len(simulator.sim_blocks))])
        yticklabels.extend([day for day in simulator.sim_blocks])
        
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(yticklabels, fontsize=6)
        self.ax.grid(True)

        self.schedulePlot.update()
        self.schedulePlot.draw()
        self.schedulePlot.show()

    def plotSolution(self,solution):
        yticks = []
        yticklabels = []
        i=0
        for day in solution.days:
            j=1 #to offset plot
            for b in solution.days[day].blocks:
                self.ax.broken_barh([(start_time, 5) for k,start_time in enumerate(solution.days[day].blocks[b].start_times_surgeries)], ((i*100)+(10*j), 3), facecolors=('blue'))
                yticks.append((i*100)+(10*j))
                yticklabels.append(f"{solution.days[day].blocks[b].block_type.name}  {solution.days[day].blocks[b].key}")
                j+=1
            i+=1

        self.ax.set_xlabel('Blocks')
        yticks.extend([i*100 for i in range(len(solution.days))])
        yticklabels.extend([solution.days[day].name for day in solution.days])
        
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(yticklabels, fontsize=6)
        self.ax.grid(True)

        self.schedulePlot.update()
        self.schedulePlot.draw()
        self.schedulePlot.show()

class Input(QComboBox):
    def __init__(self, name,items):
        super(Input, self).__init__()
        self.name = name
        self.items = items
        self.addItems(items)


def startGUI():

    app = QApplication(sys.argv)

    window =  MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.

    # Start the event loop.
    app.exec()

if __name__ == "__main__":
    startGUI()

