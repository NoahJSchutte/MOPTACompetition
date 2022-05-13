import pandas as pd

df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
columns = ['Algorithm', 'Instance_size', 'Costs_id', 'Emergency_rate', 'Objective', 'Data_type']
df = df.loc[(df['Policy'] == 'optimal') &
            #(df['Emergency_rate'] == 4) &
            (df['Data_type'] == 'generated'), columns]
df['Algorithm'] = df['Algorithm'].replace({'ALNS': 'SH', 'Stochastic': 'SOM', 'Deterministic': 'DOM'})
som = df.loc[df['Algorithm'] == 'SOM', :].reset_index()
sh = df.loc[df['Algorithm'] == 'SH', :].reset_index()
dom = df.loc[df['Algorithm'] == 'DOM', :].reset_index()
sh_vs_som = ((sh['Objective'] - som['Objective'] )/ som['Objective'])
sh_vs_dom = ((sh['Objective'] - dom['Objective'] )/ dom['Objective'])
som_vs_dom = ((som['Objective'] - dom['Objective'] )/ dom['Objective'])
print(f'SH performs {sh_vs_som.mean()}% better than SOM and {sh_vs_dom.mean()}% better than DOM on average')
print(f'SH performs up to {sh_vs_som.min()}% better than SOM and {sh_vs_dom.min()}% better than DOM on average')
print(f'SH performs {(sh_vs_som < 0).sum() / sh_vs_som.shape[0] *100}% of the time better than SOM and '
      f'{(sh_vs_dom < 0).sum() / sh_vs_dom.shape[0]*100}% of the time better than DOM')
print(f'SOM performs {som_vs_dom.mean()}% better than DOM')

df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
df['IDLETIMEmin'] = df['IDLETIME'] / 60
id_df = df.loc[(df['Policy'] == 'optimal') &
                    #(df['Emergency_rate'] == 4) &
                    (df['Algorithm'] == 'ALNS') &
                    (df['Data_type'] == 'generated'), :]
idle_per_total = (id_df.loc[id_df['Idle_costs'] > 0, 'Idle_costs'] / id_df.loc[id_df['Idle_costs'] > 0, 'Objective']).mean()
print(f'Idle costs (when positive) are an average of {idle_per_total}% of total costs')

stds = id_df.groupby(['Instance_size', 'Costs_id']).std()['IDLETIMEmin']
means = id_df.groupby(['Instance_size', 'Costs_id']).mean()['IDLETIMEmin']
average_std_in_minutes = (stds / means).mean()
print(f'Idle time has as std of {average_std_in_minutes} on average in the same configuration')
