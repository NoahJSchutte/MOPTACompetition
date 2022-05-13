# This script plots analysis results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.general.Functions.plot_functions import plot_costs_type_plots, plot_approach_results, change_width

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    #"font.size": 6
})


df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
costs_id = 1
emergency_rate = 4
data_type = 'generated'
policy = 'optimal'
df = df.loc[((df['Costs_id'] == costs_id) &
            (df['Policy'] == policy)) &
            (((df['Emergency_rate'] == 4) &
            (df['Data_type'] == 'generated')) |
            ((df['Emergency_rate'] == 2) &
             (df['Data_type'] == 'real')))
            , :]


#df_det = pd.read_csv()
#plot_approach_results(df)
#plot_costs_type_plots(df)


df['Algorithm'] = df['Algorithm'].replace({'ALNS': 'SH', 'Stochastic': 'SOM', 'Deterministic': 'DOM'})
df['Objective'] = df['Objective'] / 1000
#grouped = df[['objective', 'instance size', 'policy']].groupby(['instance size', 'policy']).mean().reset_index()
#grouped = grouped.reset_index(drop=True)
instance_sizes = [70, 100, 140, 200]
data_types = ['generated', 'real']
full_df = df.copy()
j = -4
for k, data_type in enumerate(data_types):
    fig, ax = plt.subplots(1, 4)
    df = full_df.loc[full_df['Data_type'] == data_type, :]
    j =0
    for i, instance_size in enumerate(instance_sizes):
        if data_type == 'generated':
            palette = list(sns.color_palette("Blues")[i] for i in [1, 3, 5])
        else:
            palette = list(sns.color_palette("YlOrBr")[i] for i in [1, 3, 5])
        sns.barplot(ax=ax[i+j], data=df.loc[df['Instance_size'] == instance_size], y='Objective', x='Algorithm',
                    ci=None, palette=palette,
                    order=['SH', 'SOM', 'DOM'])
        change_width(ax[i+j], 1)
        ax[i+j].spines['top'].set_visible(False)
        ax[i+j].spines['right'].set_visible(False)
        ax[i+j].spines['left'].set_visible(False)
        ax[i+j].set_xlabel(f'$I = {instance_size}$')
        #ax[i+j].margins(x=0.05)
        ax[i+j].tick_params(bottom=False)
        if i > 0 or data_type == 'real':
            ax[i+j].set_yticklabels([])
            ax[i+j].set_ylabel('')
        elif data_type == 'generated':
            ax[i+j].set_ylabel('objective (in thousands)')
        ax[i+j].set_yticks([0, 5, 10, 15, 20, 25])
        ax[i+j].set_xticklabels(['SH', 'SOM', 'DOM'], fontsize=9, rotation=90, va='center_baseline')
        ax[i+j].set_ylim((0, 30))

    ylim_last_plot = ax[3].get_ylim()
    for i, instance_size in enumerate(instance_sizes):
        ax[i+j].set_ylim(ylim_last_plot)

    if data_type == 'generated':
        extra = 0.4
        fig.subplots_adjust(bottom=0.3, left=0.15, right=0.97, top=0.97)
    else:
        extra = 0
        fig.subplots_adjust(bottom=0.3, left=0.05, right=0.97, top=0.97)
    fig.set_size_inches(3+extra, 2)
    #fig.tight_layout()

    plt.savefig(f'testFiles/Plots/Final/Approach_{data_type}.png', format='png', dpi=1200)
