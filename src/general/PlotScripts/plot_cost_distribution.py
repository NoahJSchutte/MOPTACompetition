import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})
import seaborn as sns
import pandas as pd

from src.general.Functions.plot_functions import change_width

instance_size = 200
costs_id = 1
model = 'ALNS'
policy = 'optimal'

columns = ['Objective',
           'Emergency_waiting_costs', 'Waiting_costs', 'Over_costs', 'Idle_costs',
            'Not_scheduling_costs',
           'Cancellation_costs']
rename_columns = {'Waiting_costs': '$c^w$',
                  'Emergency_waiting_costs': '$c^e$',
                  'Over_costs': '$c^o$',
                  'Idle_costs': '$c^i$',
                  'Cancellation_costs': '$c^c$',
                  'Not_scheduling_costs': '$c^s$'}
columns_plus = columns.copy()
columns_plus.append('Emergency_rate')
columns_plus.append('Algorithm')
df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
df['CANCELLING'] = df['CANCELLING'] # change to per week!
df = df.loc[(df['Costs_id'] == costs_id) &
            #(df['Algorithm'] == model) &
            (df['Policy'] == policy) &
            #(df['Instance_size'] == instance_size) &
            (df['Data_type'] == 'generated'),
            columns_plus]
df = df.groupby(['Algorithm', 'Emergency_rate']).mean().reset_index()
df = df.rename(rename_columns, axis=1)
df['Objective'] = df['Objective'] / 6
df = pd.melt(df, id_vars=['Emergency_rate', 'Algorithm'])
df['value'] = df['value'] / 1000
for i in range(0, 5):
    clone_part = df.iloc[0:9, :].copy()
    clone_part['variable'] = clone_part['variable'].apply(lambda x: f'{x}_{i}')
    df = pd.concat([df, clone_part])

full_df = df.copy()
number_of_figures = 6
half_num_fig = round(number_of_figures/2)

j = -half_num_fig
for model in ['ALNS', 'Stochastic']:
    fig, ax = plt.subplots(1, half_num_fig)
    j += half_num_fig
    j = 0
    df = full_df.loc[full_df['Algorithm'] == model, :]
    for i, e in enumerate([1, 2, 4]):
        sns.barplot(ax=ax[i+j], data=df.loc[(df['Emergency_rate'] == e) & (df['variable'].apply(lambda x: x[0:3]) == 'Obj'), :],
                    x='variable', y='value', ci=None,
                    palette=[sns.color_palette("YlOrBr")[3]])
        sns.barplot(ax=ax[i+j], data=df.loc[(df['Emergency_rate'] == e) & (df['variable'].apply(lambda x: x[0:3]) != 'Obj'), :],
                    x='variable', y='value', ci=None,
                    palette=sns.color_palette("Blues")) # adjust order
        change_width(ax[i+j], 1)
        ax[i+j].spines['top'].set_visible(False)
        ax[i+j].spines['right'].set_visible(False)
        ax[i+j].spines['left'].set_visible(False)
        if i > 0 or model == 'Stochastic':
            ax[i+j].set_yticklabels([])
            ax[i+j].set_ylabel('')
        elif model == 'ALNS':
            ax[i+j].set_ylabel('costs (in thousands)')
        ax[i+j].set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax[i+j].set_xlabel(f'$e = {e}$')
        ax[i+j].set_ylim(0, 6)
        ax[i+j].tick_params(bottom=False)
        #ax[i+j].margins(x=0.05)

    fig.subplots_adjust(bottom=0.2, left=0.05, right=0.97, top=0.97)
    if model == 'ALNS':
        extra = 0.4
        fig.subplots_adjust(left=0.15)
    else:
        extra = 0
    fig.set_size_inches(3+extra, 2)
    #fig.tight_layout()

    plt.savefig(f'testFiles/Plots/Final/Costs_distribution_{model}.png', format='png', dpi=1200)