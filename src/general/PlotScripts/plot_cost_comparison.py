import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    #"font.size": 6
})

y = 'Emergency_rate'
y = 'Costs_id'
emergency_rate = 2
if y == 'Emergency_rate':
    df = pd.read_csv('testFiles/Tables/table_emergency_rate_comparison.csv')
    palette = list(sns.color_palette("YlOrBr")[j] for j in [1, 3, 5])
    label = 'emergency rate'
else:
    df = pd.read_csv(f'testFiles/Tables/table_costs_comparison_e={emergency_rate}.csv')
    palette = list(sns.color_palette("Blues")[j] for j in [1, 3, 5])
    label = 'cost structure'
instance = 140
df = df.drop(columns=['Unnamed: 0'])
df = df.groupby(y).mean().reset_index()
#df = df.loc[df['Instance_size'] == instance, :]
units = ['Unscheduled Elective', 'Cancelled Elective', 'Idle time', 'Over time',
         'Elective waiting', 'Emergency waiting']
map_units = {'Unscheduled Elective': 'unscheduled\n electives',
             'Cancelled Elective': 'cancelled\n electives',
             'Idle time': 'idle time',
             'Over time': 'over time',
             'Elective waiting': 'elective\n waiting time',
             'Emergency waiting': 'emergency\n waiting time'}
df = df.rename(map_units, axis=1)

fig, ax = plt.subplots(2, 3)
f = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0, 2), 5: (1, 2)}

mapped_units = [map_units[i] for i in units]
for i, unit in enumerate(mapped_units):
    sns.barplot(ax=ax[f[i]], data=df, y=y, x=unit, ci=None, orient="h",
                palette=palette)
    ax[f[i]].spines['top'].set_visible(False)
    ax[f[i]].spines['right'].set_visible(False)
    if i > 1:
        ax[f[i]].set_ylabel('')
        ax[f[i]].set_yticks([])
    else:
        if y == 'Costs_id':
            ax[f[i]].set_yticks([0, 1, 2], ['$C_1$', '$C_2$', '$C_3$'])
        ax[f[i]].set_ylabel('')

    ax[f[i]].tick_params(left=False)
    ax[f[i]].set_xlabel(unit, ha='left', position=(0, 0))
    if i == 1:
        ax[f[i]].set_xticks([0, 0.1], ['0', '0.1'])
    if i == 5 and y == 'Costs_id':
        ax[f[i]].set_xticks([0, 2], ['0', '2'])



    #ax[i].spines['left'].set_visible(False)
#fig.subplots_adjust(hspace=.4)
fig.set_size_inches(3.2, 2.5)
fig.tight_layout()
if y == 'Costs_id':
    fig.subplots_adjust(bottom=0.23, left=0.08, right=0.97, top=0.97, hspace=0.9)
else:
    fig.subplots_adjust(bottom=0.23, left=0.06, right=0.97, top=0.97, hspace=0.9)
#for i, unit in enumerate(mapped_units):
#    xlabel = ax[f[i]].get_xaxis().get_label()
#    trans = xlabel.get_transform()
#    xpos, ypos = trans.transform(xlabel.get_position())  # convert to display coordinates
#    xpos, ypos = fig.transFigure.inverted().transform([xpos, ypos])
#    ax[f[i]].annotate(unit, (xpos, ypos+0.15), xycoords='figure fraction', va='top')
#    ax[f[i]].set_xlabel('')

#fig.subplots_adjust(wspace=0.4)
plt.savefig(f'testFiles/Plots/Final/Comparison_{y}.png', format='png', dpi=1200)