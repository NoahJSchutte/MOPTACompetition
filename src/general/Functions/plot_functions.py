import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_policy_performance(df, title='Plot policies', base_policy='optimal', sort=True, plot_average=False,
                            pool_size=100, save_plot=False, save_directory=''):
    if sort:
        df = df.sort_values(base_policy).reset_index(drop=True)

    plt.plot(df)
    if plot_average:
        averages = df.mean()
        standard_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.hlines(averages, 0, pool_size, colors=standard_colors[:len(averages)])

    plt.legend(df.columns)
    plt.title(title)
    if save_plot:
        plt.savefig(save_directory)


def plot_policy_boxplots(df, pool_size, solution_name, save_plot=True, save_directory=''):
    flierprops = dict(markersize=3, marker='.')
    sns.boxplot(x='instance size', y='objective', hue='policy', data=df, flierprops=flierprops)
    title = f'Boxplots of evaluated objectives tested on {pool_size} scenarios \non the {solution_name} solution'
    plt.title(title)

    if save_plot:
        plt.savefig(save_directory)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def plot_policy_results(df):
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
        ax[i].set_xlabel(f'{instance_size}')
        ax[i].margins(x=0.05)
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('objective (in thousands)')
            ax[i].set_yticks([5, 10, 15, 20, 25, 30, 35, 40])

    ylim_last_plot = ax[3].get_ylim()
    for i, instance_size in enumerate(instance_sizes):
        ax[i].set_ylim(ylim_last_plot)

def plot_approach_results(df):
    fig, ax = plt.subplots(1, 4)
    df['Algorithm'] = df['Algorithm'].replace({'ALNS': 'SH', 'Stochastic': 'SOM', 'Deterministic': 'DOM'})
    df['Objective'] = df['Objective'] / 1000
    #grouped = df[['objective', 'instance size', 'policy']].groupby(['instance size', 'policy']).mean().reset_index()
    #grouped = grouped.reset_index(drop=True)
    instance_sizes = [70, 100, 140, 200]
    data_types = ['generated', 'real']
    for j, data_type in enumerate(data_types):
        for i, instance_size in enumerate(instance_sizes):
            sns.barplot(ax=ax[i], data=df.loc[df['Instance_size'] == instance_size], y='Objective', x='Algorithm',
                        ci=None, palette=list(sns.color_palette("Blues")[i] for i in [1, 3, 5]),
                        order=['SH', 'SOM', 'DOM'])
            change_width(ax[i], 1)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].set_xlabel(f'{instance_size}')
            ax[i].margins(x=0.05)
            if i > 0:
                ax[i].set_yticklabels([])
                ax[i].set_ylabel('')
            else:
                ax[i].set_ylabel('objective (in thousands)')
                ax[i].set_yticks([5, 10, 15, 20, 25, 30, 35, 40])

        ylim_last_plot = ax[3].get_ylim()
        for i, instance_size in enumerate(instance_sizes):
            ax[i].set_ylim(ylim_last_plot)



def plot_policy_subplots(df, pool_size, solution_name='...', save_plot=True, save_directory='',
                         optimal_relative=False, kind='box', kwargs=dict()):
    if optimal_relative:
        grouped = df[['objective', 'instance size', 'policy']].groupby(['instance size', 'policy']).mean()
        optimal_means = grouped.groupby(['instance size']).min()
        for instance_size in [70, 100, 140, 200]:
            grouped.loc[instance_size, 'optimal objective'] = optimal_means.loc[instance_size, 'objective']
            grouped.loc[instance_size, 'relative objective'] = grouped.loc[instance_size, 'objective'] / \
                                                               optimal_means.loc[instance_size, 'objective']
            df.loc[df['instance size'] == instance_size, 'optimal mean'] = optimal_means.loc[instance_size, 'objective']
        grouped['relative objective'] = (grouped['objective'] / grouped['optimal objective'] - 1) * 100
        df['relative objective'] = df['objective'] / df['optimal mean']
        objective = 'relative objective'
    else:
        objective = 'objective'
    # This function should plot the boxplot in 4 subplots
    if kind == 'box':
        flierprops = dict(markersize=3, marker='.')
        kwargs['flierprops'] = flierprops
    #elif kind == 'bar':
        #kwargs['ci'] = None
    elif kind == 'violin':
        df = pd.DataFrame(df.to_dict())

    sns.catplot(x='policy', y=objective, col='instance size', col_wrap=4, kind=kind, data=df, size=5,
                **kwargs)
    title = f'Boxplots of evaluated objectives tested on {pool_size} scenarios \non the {solution_name} solution'
    #plt.title(title)

    if save_plot:
        plt.savefig(save_directory)


def plot_policy_line_plots(df):
    sns.relplot(x='index', y='objective', data=df, hue='policy', col='instance size', kind='line', col_wrap=2)


def plot_costs_type_plots(df):
    df_unpivot = pd.melt(df, id_vars=['Algorithm', 'Instance_size'], var_name='Costs type', value_name='Costs',
                         value_vars=['Waiting_costs', 'Emergency_waiting_costs', 'Over_costs', 'Idle_costs',
                                     'Cancellation_costs', 'Not_scheduling_costs'])
    sns.catplot(data=df_unpivot, y='Costs', x='Algorithm', hue='Costs type', kind='bar',
                col='Instance_size', col_wrap=2, ci=None)


def plot_objective_comparison(df):
    sns.catplot(data=df, y='Objective', x='Instance_size', hue='Algorithm', kind='bar', ci=None)


def plot_alns_search(track_dict, title='', has_evaluator=False, has_validation_results=False, per_evaluation=False,
                     pool_size=100):
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

    sns.lineplot(x=transformation[track_dict['objective x']], y=track_dict['objective y'], label='Candidate')
    sns.lineplot(x=transformation[track_dict['acceptance x']], y=track_dict['acceptance y'], label='Current')
    sns.lineplot(x=transformation[track_dict['best x']], y=track_dict['best y'], label='Best')
    if has_validation_results:
        sns.lineplot(x=transformation[track_dict['validation x']], y=track_dict['validation y'], label='Validation')
    plt.title(title)


def plot_multiple_alns_search(track_dict_list, title='', has_evaluator=False, has_validation_results=False, per_evaluation=False,
                     pool_size=100, second_track_dict_list=None, two_rows=False, width=1.6):
    if two_rows:
        fig, ax = plt.subplots(2, 2)
        ids = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    else:
        number_of_plots = len(track_dict_list)
        fig, ax = plt.subplots(1, number_of_plots)
        ids = [0, 1, 2, 3]

    instance_sizes = [70, 100, 140, 200]
    for i, track_dict in enumerate(track_dict_list):
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

        sns.lineplot(ax=ax[ids[i]], x=transformation[track_dict['objective x']],
                     y=[i/1000 for i in track_dict['objective y']], label='candidate det', linestyle='solid',
                     color=sns.color_palette("Blues")[3],
                     linewidth=width)
        sns.lineplot(ax=ax[ids[i]], x=transformation[track_dict['acceptance x']],
                     y=[i/1000 for i in track_dict['acceptance y']], label='current det', linestyle='solid',
                     color=sns.color_palette("Blues")[-1],
                     linewidth=width)
        sns.lineplot(ax=ax[ids[i]], x=transformation[track_dict['best x']],
                     y=[i/1000 for i in track_dict['best y']], label='best det', linestyle='', marker='+',
                     color='black', markerfacecolor=sns.color_palette("Blues")[-1],
                     markeredgecolor='black',
                     markersize=3)
        if second_track_dict_list:
            second_track_dict = second_track_dict_list[i]
            sns.lineplot(ax=ax[ids[i]], x=transformation[second_track_dict['objective x']],
                         y=[i/1000 for i in second_track_dict['objective y']], label='candidate ran', linestyle='solid',
                         color=sns.color_palette("YlOrBr")[2],
                         linewidth=width)
            sns.lineplot(ax=ax[ids[i]], x=transformation[second_track_dict['acceptance x']],
                         y=[i/1000 for i in second_track_dict['acceptance y']], label='current ran', linestyle='solid',
                         color=sns.color_palette("YlOrBr")[4],
                         linewidth=width)
            sns.lineplot(ax=ax[ids[i]], x=transformation[second_track_dict['best x']],
                         y=[i/1000 for i in second_track_dict['best y']], label='best ran', linestyle='', marker='+',
                         color=sns.color_palette("YlOrBr")[4], markerfacecolor=sns.color_palette("YlOrBr")[4],
                         markeredgecolor='black',
                         markersize=3)


        if has_validation_results:
            sns.lineplot(ax=ax[ids[i]], x=transformation[track_dict['validation x']], y=track_dict['validation y'], label='validation')
        if i == 0:
            ax[ids[i]].set_ylabel('objective (in thousands)')
        if i < 3:
            legend = ax[ids[i]].legend()
            legend.remove()


        ax[ids[i]].spines['top'].set_visible(False)
        ax[ids[i]].spines['right'].set_visible(False)
        ax[ids[i]].set_xlabel(f'iteration')
        if two_rows:
            x = 500
        else:
            x = 200
        ax[ids[i]].text(x=x, y=max([i/1000 for i in second_track_dict['objective y']]),
                        s=f'I={instance_sizes[i]}', verticalalignment='top', horizontalalignment='center',
                        bbox=dict(facecolor='white'))
    ax[ids[0]].set_yticks([15, 20])
    ax[ids[1]].set_yticks([15, 20, 25])
    ax[ids[2]].set_yticks([15, 25, 35])
    ax[ids[3]].set_yticks([20, 30, 40, 50])
    if two_rows:
        ax[1, 0].set_ylabel('objective (in thousands)')
        ax[0, 0].set_xlabel('')
        ax[0, 1].set_xlabel('')
    else:
        plt.subplots_adjust(bottom=0.15)
        ax[3].legend(loc='center right')
        plt.legend(bbox_to_anchor=(1.1, 0.8))



def plot_alns_tuning(track_dict, title='', has_evaluator=False, has_validation_results=False, per_evaluation=False,
                     pool_size=100, save_plot=False, save_directory='', config=1):
    """
    Plot two different configurations in one plot
    """
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

    sns.lineplot(x=transformation[track_dict['objective x']], y=track_dict['objective y'], label=f'Candidate configuration {config}')
    sns.lineplot(x=transformation[track_dict['acceptance x']], y=track_dict['acceptance y'], label=f'Current configuration {config}')
    sns.lineplot(x=transformation[track_dict['best x']], y=track_dict['best y'], label=f'Best configuration {config}')
    if has_validation_results:
        sns.lineplot(x=transformation[track_dict['validation x']], y=track_dict['validation y'], label=f'Validation {config}')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")

    if save_plot:
        plt.savefig(save_directory)
        plt.close()



#fig, ax = plt.subplots()
#sns.barplot(x='policy', y=objective, data=df, ci=None, ax=ax, linewidth=2.5, facecolor=(1, 1, 1, 0), edgecolor='.2')
#sns.boxplot(x='policy', y=objective, data=df, width=0.5, ax=ax)