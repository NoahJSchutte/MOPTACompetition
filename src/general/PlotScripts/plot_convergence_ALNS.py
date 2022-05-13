import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    #"font.size": 6
})


from src.general.Functions.general_functions import load_object
from src.general.Functions.plot_functions import plot_multiple_alns_search

seed = 8
alpha = 0.25
h = 1.01
e = 4
c = 1
instance_sizes = [70, 140, 200]
first_dict_list = list()
second_dict_list = list()
two_rows = False
has_evaluator=False
has_validation_results=False
per_evaluation=False
pool_size=100 
cond_dict_list=None
two_rows=False
width = 1
markersize = 2

for n in instance_sizes:
    file_name = f'n={n}_e={e}_c={c}_i=1000_h={h}_a={alpha}_s=100_p=exponential_init=Deterministic_seed={seed}'
    tracking_results = load_object('testFiles/TrackingResults', file_name)
    first_dict_list.append(tracking_results)
    file_name_2 = f'n={n}_e={e}_c={c}_i=1000_h={h}_a={alpha}_s=100_p=exponential_init=Random_seed={seed}'
    tracking_results = load_object('testFiles/TrackingResults', file_name_2)
    second_dict_list.append(tracking_results)

for i, track_dict in enumerate(first_dict_list):
    fig, ax = plt.subplots(1, 1)
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
                 y=[i/1000 for i in track_dict['objective y']], label='candidate det', linestyle='solid',
                 color=sns.color_palette("Blues")[2],
                 linewidth=width)
    sns.lineplot(ax=ax, x=transformation[track_dict['acceptance x']],
                 y=[i/1000 for i in track_dict['acceptance y']], label='current det', linestyle='solid',
                 color=sns.color_palette("Blues")[5],
                 linewidth=width)
    sns.lineplot(ax=ax, x=transformation[track_dict['best x']],
                 y=[i/1000 for i in track_dict['best y']], label='best det', linestyle='', marker='+',
                 color='black', markerfacecolor=sns.color_palette("Blues")[-1],
                 markeredgecolor='black',
                 markersize=markersize)
    if second_dict_list:
        second_track_dict = second_dict_list[i]
        sns.lineplot(ax=ax, x=transformation[second_track_dict['objective x']],
                     y=[i/1000 for i in second_track_dict['objective y']], label='candidate ran', linestyle='solid',
                     color=sns.color_palette("YlOrBr")[2],
                     linewidth=width)
        sns.lineplot(ax=ax, x=transformation[second_track_dict['acceptance x']],
                     y=[i/1000 for i in second_track_dict['acceptance y']], label='current ran', linestyle='solid',
                     color=sns.color_palette("YlOrBr")[4],
                     linewidth=width)
        sns.lineplot(ax=ax, x=transformation[second_track_dict['best x']],
                     y=[i/1000 for i in second_track_dict['best y']], label='best ran', linestyle='', marker='+',
                     color=sns.color_palette("YlOrBr")[4], markerfacecolor=sns.color_palette("YlOrBr")[4],
                     markeredgecolor='black',
                     markersize=markersize)


    if has_validation_results:
        sns.lineplot(ax=ax, x=transformation[track_dict['validation x']], y=track_dict['validation y'], label='validation')
    if i == 0:
        ax.set_ylabel('objective (in thousands)')
    if i < 3:
        legend = ax.legend()
        legend.remove()


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(f'iteration')
    if two_rows:
        x = 500
    else:
        x = 200
    #ax.text(x=x, y=max([i/1000 for i in second_track_dict['objective y']]),
    #                s=f'I={instance_sizes[i]}', verticalalignment='top', horizontalalignment='center',
    #                bbox=dict(facecolor='white'))



    if i == 0:
        fig.set_size_inches(2.2, 2.3)
        ax.set_yticks([15, 20])
        ax.set_ylabel('objective (in thousands)')
        fig.subplots_adjust(bottom=0.18, left=0.22, right=0.97, top=0.97)
    elif i == 1:
        fig.set_size_inches(2, 2.3)
        ax.set_yticks([15, 25, 35])
        fig.subplots_adjust(bottom=0.18, left=0.13, right=0.97, top=0.97)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0:3], labels[0:3])
    else:
        fig.set_size_inches(2.3, 2.3)
        ax.set_yticks([20, 30, 40, 50])
        ax.legend(loc='center right')
        fig.subplots_adjust(bottom=0.18, left=0.13, right=0.84, top=0.97)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[3:6], labels[3:6], bbox_to_anchor=(0.3,1))
        #plt.legend(bbox_to_anchor=(0.35, 1))

    plt.savefig(f'testFiles/Plots/Final/Convergence_{i}.png', format='png', dpi=1200)

