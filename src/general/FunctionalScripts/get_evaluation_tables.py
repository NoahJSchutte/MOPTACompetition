import pandas as pd

# Get table for costs evaluation
emergency_rate = 4
model = 'ALNS'
policy = 'optimal'
columns = ['Instance_size',
           'Costs_id',
           'Objective',
           'NOTSCHEDULING',
           'CANCELLING',
           'IDLETIME',
           'OVERTIME',
           'ELECTIVEWAITINGTIME',
           'EMERGENCYWAITINGTIME']
rename_columns = {'NOTSCHEDULING' : 'Unscheduled Elective',
                  'CANCELLING' : 'Cancelled Elective',
                  'IDLETIME': 'Idle time',
                  'OVERTIME': 'Over time',
                  'ELECTIVEWAITINGTIME': 'Elective waiting',
                  'EMERGENCYWAITINGTIME': 'Emergency waiting'}

df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
df['CANCELLING'] = df['CANCELLING'] # change to per week!
table = df.loc[(df['Emergency_rate'] == emergency_rate) &
               (df['Algorithm'] == model) &
               (df['Policy'] == policy),
                columns]
for i in ['IDLETIME',
          'OVERTIME',
          'ELECTIVEWAITINGTIME',
          'EMERGENCYWAITINGTIME']:
    table[i] = round(table[i] / 60, 1)

table = table.rename(rename_columns, axis=1)
table.to_csv(f'testFiles/Tables/table_costs_comparison_e={emergency_rate}.csv')

# Get some other table?
id = 1
model = 'ALNS'
policy = 'optimal'
columns = ['Instance_size',
           'Emergency_rate',
           'Objective',
           'NOTSCHEDULING',
           'CANCELLING',
           'IDLETIME',
           'OVERTIME',
           'ELECTIVEWAITINGTIME',
           'EMERGENCYWAITINGTIME']

rename_columns = {'NOTSCHEDULING' : 'Unscheduled Elective',
                  'CANCELLING' : 'Cancelled Elective',
                  'IDLETIME': 'Idle time',
                  'OVERTIME': 'Over time',
                  'ELECTIVEWAITINGTIME': 'Elective waiting',
                  'EMERGENCYWAITINGTIME': 'Emergency waiting'}

df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
df['CANCELLING'] = df['CANCELLING'] # change to per week!
table = df.loc[(df['Costs_id'] == id) &
               (df['Algorithm'] == model) &
               (df['Policy'] == policy) &
               (df['Data_type'] == "generated"),
                columns]
for i in ['IDLETIME',
          'OVERTIME',
          'ELECTIVEWAITINGTIME',
          'EMERGENCYWAITINGTIME']:
    table[i] = round(table[i] / 60, 1)

table = table.rename(rename_columns, axis=1)
table.to_csv('testFiles/Tables/table_emergency_rate_comparison.csv')


# Get table with all emergency/costs combinations
#emergency_rate = 4
model = 'ALNS'
policy = 'optimal'
columns = ['Instance_size',
           'Costs_id',
           'Emergency_rate',
           'Objective',
           'NOTSCHEDULING',
           'CANCELLING',
           'IDLETIME',
           'OVERTIME',
           'ELECTIVEWAITINGTIME',
           'EMERGENCYWAITINGTIME']
rename_columns = {'NOTSCHEDULING' : 'Unscheduled Elective',
                  'CANCELLING' : 'Cancelled Elective',
                  'IDLETIME': 'Idle time',
                  'OVERTIME': 'Over time',
                  'ELECTIVEWAITINGTIME': 'Elective waiting',
                  'EMERGENCYWAITINGTIME': 'Emergency waiting'}

df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
df['CANCELLING'] = df['CANCELLING'] # change to per week!
table = df.loc[(df['Algorithm'] == model) &
               (df['Policy'] == policy),
               columns]
for i in ['IDLETIME',
          'OVERTIME',
          'ELECTIVEWAITINGTIME',
          'EMERGENCYWAITINGTIME']:
    table[i] = round(table[i] / 60, 1)

table = table.rename(rename_columns, axis=1)
table.to_csv(f'testFiles/Tables/table_costs_and_emergency_comparison.csv')



# Full evaluation table ---------------------------------------------------------------------------------------------- #
df = pd.read_csv('testFiles/EvaluationResults/evaluation_full_final.csv')
columns = ['Algorithm', 'Instance_size', 'Costs_id', 'Emergency_rate', 'Objective', 'Data_type']
df = df.loc[df['Policy'] == 'optimal', columns]
df['Algorithm'] = df['Algorithm'].replace({'ALNS': 'SH', 'Stochastic': 'SOM', 'Deterministic': 'DOM'})
table = df.pivot(index=['Instance_size', 'Algorithm'], columns=['Costs_id', 'Emergency_rate', 'Data_type'],
                 values='Objective')
table = table.reindex([(70,  'SH'),
               (70, 'SOM'),
               (70, 'DOM'),
               (100,  'SH'),
               (100, 'SOM'),
               (100, 'DOM'),
               (140,  'SH'),
               (140, 'SOM'),
               (140, 'DOM'),
               (200,  'SH'),
               (200, 'SOM'),
               (200, 'DOM')])
table.to_csv('testFiles/Tables/table_full_comparison.csv')