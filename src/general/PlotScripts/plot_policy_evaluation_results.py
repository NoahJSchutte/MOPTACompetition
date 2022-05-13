# This script evaluates different policies
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



from src.general.Functions.general_functions import get_lognorm_param, save_object, load_object
from src.general.Functions.plot_functions import change_width

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    #"font.size": 6
})

p = os.path.abspath('..')
sys.path.insert(1, p)
directory = os.getcwd()

pool_size = 1000
instance_sizes = [70, 100, 140, 200]
costs_id = 1
emergency_rate = 4
solution_types = ['Deterministic']
test_folder = 'Testing'
get_stats = False
plot_file_name = 'policy_evaluation_plot'

# Load dataframe
# Making plots ------------------------------------------------------------------------------------------------------- #
#solution_name = 'alns'

for solution_type in solution_types:
    df = pd.DataFrame(columns=['objective', 'policy', 'instance size', 'emergency rate', 'costs id', 'solution'])
    for instance_size in instance_sizes:
        file_name = f'n={instance_size}_e={emergency_rate}_c={costs_id}_s={pool_size}' \
                    f'_solution={solution_type}'
        new_df = load_object(directory='testFiles/PolicyEvaluationResults',
                             file_name=file_name)
        df = pd.concat([df, new_df])
    #file_name = f'Boxplot_{pool_size}_{used_policy}'
    save_directory = f'testFiles/Plots/{plot_file_name}'
    fig, ax = plt.subplots(1, 4)
    df.loc[df['policy'] == 'optimal', 'policy'] = 'log-normal'
    df['policy'] = df['policy'].apply(lambda x: x[0:3])
    df['objective'] = df['objective'] / 1000
    grouped = df[['objective', 'instance size', 'policy']].groupby(['instance size', 'policy']).mean().reset_index()
    #grouped = grouped.reset_index(drop=True)
    instance_sizes = [70, 100, 140, 200]
    for i, instance_size in enumerate(instance_sizes):
        sns.barplot(ax=ax[i], data=grouped.loc[grouped['instance size'] == instance_size], y='objective', x='policy',
                    ci=None, palette=sns.color_palette("Blues")[1:],
                    order=['log', 'exp', 'det', 'gre', 'ran'])
        change_width(ax[i], 1)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_xlabel(f'$I = {instance_size}$')
        ax[i].tick_params(bottom=False)
        #ax[i].margins(x=0.05)
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('objective (in thousands)')
        ax[i].set_yticks([0, 10, 20, 30])

    ylim_last_plot = ax[3].get_ylim()
    for i, instance_size in enumerate(instance_sizes):
        ax[i].set_ylim(ylim_last_plot)

    fig.set_size_inches(6.4, 2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.97)
    plt.savefig(f'testFiles/Plots/Final/Policy_plot.png', format='png', dpi=1200)