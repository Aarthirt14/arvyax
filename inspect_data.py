import pandas as pd

train = pd.read_excel('Sample__reflective_dataset.xlsx')
test = pd.read_excel('_test_inputs_120.xlsx')

print('=' * 80)
print('TRAINING DATA')
print('=' * 80)
print(f'Shape: {train.shape}')
print(f'\nColumns: {train.columns.tolist()}')
print(f'\nData types:\n{train.dtypes}')
print(f'\nFirst 3 rows:')
print(train.head(3))
print(f'\nMissing values:\n{train.isnull().sum()}')

print('\n\n' + '=' * 80)
print('TEST DATA')
print('=' * 80)
print(f'Shape: {test.shape}')
print(f'\nColumns: {test.columns.tolist()}')
print(f'\nData types:\n{test.dtypes}')
print(f'\nFirst 3 rows:')
print(test.head(3))
print(f'\nMissing values:\n{test.isnull().sum()}')
