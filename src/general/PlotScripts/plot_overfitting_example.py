import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

from src.general.Functions.general_functions import load_object


def uniform_edits(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(f'iteration')
    ax.set_yticks([16, 18, 20, 22])
    ax.set_ylim((15.5, 24))
    ax.set_xticks([0, 250, 500, 750, 1000])

file_name_2 = 'n=140_e=4_c=1_i=1000_h=1.01_a=0.25_s=1000_p=exponential_init=Deterministic_seed=4x=3_y=1_z=5_f=0_sf=Validating_ss=100__m=advanced_mip_tt=5_sr=0.1_se0.01_mp=1_pn=3'
file_name_1 = 'n=140_e=4_c=1_i=1000_h=1.01_a=0.25_s=1000_p=exponential_init=Deterministic_seed=4x=3_y=1_z=5_f=0_sf=Validating_ss=100_'
file_name_1 = 'n=140_e=4_c=1_i=1000_h=1.01_a=0.25_s=100_p=exponential_init=Deterministic_seed=4x=3_y=1_z=5_f=0_sf=Validating_ss=100_'
file_name_2 = 'n=140_e=4_c=1_i=1000_h=1.01_a=0.25_s=1000_p=exponential_init=Deterministic_seed=4x=3_y=1_z=5_f=0_sf=Validating_ss=100__m=advanced_mip_tt=5_sr=0.1_se0.01_mp=1_pn=3'

track_dict = load_object('testFiles/TrackingResults', file_name_1)
second_track_dict = load_object('testFiles/TrackingResults', file_name_2)

pool_size = 1000
points_of_discrepancy = list()
stop_appending = False
for i, j in enumerate(second_track_dict['evaluator']['evaluations per iteration']):
    if j > 100 and not stop_appending:
        points_of_discrepancy.append(i)
        stop_appending = True
    elif j < 1000 and stop_appending:
        stop_appending = False

per_evaluation = False
has_evaluator = True
width = 1

fig, ax = plt.subplots(1, 1)
#ids = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
ids = [0, 1]

if per_evaluation and has_evaluator:
    eval_per_it = track_dict['evaluator']['evaluations per iteration']
    eval_per_it_cum = eval_per_it.cumsum()
    transformation = np.concatenate(([0], eval_per_it_cum))
elif per_evaluation:
    eval_per_it = np.ones(len(track_dict['objective x'])) * pool_size
    eval_per_it_cum = eval_per_it.cumsum()
    transformation = np.concatenate(([0], eval_per_it_cum))
else:
    transformation = np.array(track_dict['objective x'])

sns.lineplot(ax=ax, x=transformation[track_dict['objective x']],
             y=[i/1000 for i in track_dict['objective y']], label='candidate', linestyle='solid',
             color=sns.color_palette("Blues")[2],
             linewidth=width)
sns.lineplot(ax=ax, x=transformation[track_dict['acceptance x']],
             y=[i/1000 for i in track_dict['acceptance y']], label='current', linestyle='solid',
             color=sns.color_palette("Blues")[5],
             linewidth=width)
sns.lineplot(ax=ax, x=transformation[track_dict['validation x']],
             y=[i/1000 for i in track_dict['validation y']], label='validation', linestyle='solid',
             color=sns.color_palette("YlOrBr")[3],
             linewidth=width)
sns.lineplot(ax=ax, x=transformation[track_dict['best x']],
             y=[i/1000 for i in track_dict['best y']], label='best', linestyle='', marker='+',
             color='black', markerfacecolor=sns.color_palette("Blues")[4],
             markeredgecolor='black',
             markersize=3.5)

uniform_edits(ax)
ax.set_ylabel('objective (in thousands)')
fig.set_size_inches(3.3, 2.5)
fig.subplots_adjust(bottom=0.15, left=0.15, right=0.97, top=0.97)
plt.savefig(f'testFiles/Plots/Final/Scenario_selection_1.png', format='png', dpi=1200)



# Subplot number 2
fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax, x=transformation[second_track_dict['objective x']],
             y=[i/1000 for i in second_track_dict['objective y']], label='candidate', linestyle='solid',
             color=sns.color_palette("Blues")[2],
             linewidth=width)
sns.lineplot(ax=ax, x=transformation[second_track_dict['acceptance x']],
             y=[i/1000 for i in second_track_dict['acceptance y']], label='current', linestyle='solid',
             color=sns.color_palette("Blues")[4],
             linewidth=width)
sns.lineplot(ax=ax, x=transformation[second_track_dict['validation x']],
             y=[i/1000 for i in second_track_dict['validation y']], label='current validation', linestyle='solid',
             color=sns.color_palette("YlOrBr")[3],
             linewidth=width)
sns.lineplot(ax=ax, x=transformation[second_track_dict['best x']],
             y=[i/1000 for i in second_track_dict['best y']], label='best', linestyle='', marker='+',
             color=sns.color_palette("Blues")[4], markerfacecolor=sns.color_palette("YlOrBr")[4],
             markeredgecolor='black',
             markersize=3.5)
for vline_value in points_of_discrepancy:
    ax.axvline(x=vline_value, linewidth=0.5, color='0.7')

legend = ax.legend()
legend.remove()
uniform_edits(ax)
ax.set_yticklabels([])
fig.set_size_inches(3.1, 2.5)
fig.subplots_adjust(bottom=0.15, left=0.05, right=0.97, top=0.97)
plt.savefig(f'testFiles/Plots/Final/Scenario_selection_2.png', format='png', dpi=1200)



