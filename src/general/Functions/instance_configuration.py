from src.general.Classes.instance_info import InstanceInfo
from src.general.Classes.type import Type
from src.general.Functions.general_functions import get_lognorm_param, save_object, load_object, test_policy
import pandas as pd


def instance_configuration(n, costs_csv="data/input/Costs_1.csv",
                           dist_csv="data/input/Distributions_fitted.csv",
                           percentages_csv="data/input/Percentages_waitlist.csv",
                           emergency_arrival_rate=4):
    df_dist = pd.read_csv(dist_csv)
    df_percentages = pd.read_csv(percentages_csv, sep=';')
    df_costs = pd.read_csv(costs_csv, sep=";", header=None, index_col=0).to_dict()[1]
    
    emergency_duration_distribution = df_dist.loc[df_dist['Type'] == 'EMERGENCY', 'Optimal distribution'].values[0]
    emergency_duration_parameters = [*eval(df_dist.loc[df_dist['Type'] == 'EMERGENCY', 'Parameters'].values[0])]
    
    instance_info = InstanceInfo(instance_size=n,
                                 not_scheduling_costs=df_costs["NOTSCHEDULING"],
                                 cancellation_costs=df_costs["CANCELLING"],
                                 waiting_costs=df_costs["ELECTIVEWAITINGTIME"],
                                 waiting_emergency_costs=df_costs["EMERGENCYWAITINGTIME"],
                                 over_time_costs=df_costs["OVERTIME"],
                                 idle_time_costs=df_costs["IDLETIME"],
                                 emergency_duration_distribution=emergency_duration_distribution,
                                 emergency_duration_parameters=emergency_duration_parameters,
                                 emergency_arrival_distribution='exponential',
                                 emergency_arrival_parameters=[int(480/emergency_arrival_rate)])
    for idx in df_dist.index:
        block_type = df_dist.loc[idx, 'Type']
        if block_type != 'EMERGENCY':
            percentage = df_percentages.loc[df_percentages['Type'] == block_type, "Percentage"]
            percentage_as_float = float(percentage.values[0].replace(',', '.'))
            instance_info.add_type(Type(block_type, df_dist.loc[idx, 'Optimal distribution'],
                                        [*eval(df_dist.loc[idx, 'Parameters'])],
                                        percentage_as_float,
                                        round(percentage_as_float*instance_info.instance_size),
                                        exp_rate=df_dist.loc[idx, 'Exponential rate']))

    return instance_info
