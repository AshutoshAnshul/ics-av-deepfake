# code to split the main file into train val and test and save as json metadata. 
# Also, it saves another column showing the name of file where the extracted feature would be saved

import pandas as pd
from sklearn.model_selection import train_test_split
import json

fraction = 0.8

data = pd.read_csv('meta_data.csv', header=0)
data = data.drop(data.columns.to_list()[-2:], axis=1)

data['feature_file'] = data['path'].str.split('/').apply(lambda x: '_'.join(x[1:])).str.replace(' ', '') + '_' + data['filename'].str.replace('.mp4', '.json')

train_val_data = data.groupby(["type"]).sample(frac=fraction, random_state=2)
test_data = data.drop(train_val_data.index)

train_val_data = train_val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_data = train_val_data.groupby(["type"]).sample(frac=fraction, random_state=2)
val_data = train_val_data.drop(train_data.index)

train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

print(len(train_data))
print(len(val_data))
print(len(test_data))

int_data = pd.merge(train_data, test_data, how='inner', on=['filename', 'path'])
print(len(int_data))
int_data = pd.merge(train_data, val_data, how='inner', on=['filename', 'path'])
print(len(int_data))
int_data = pd.merge(test_data, val_data, how='inner', on=['filename', 'path'])
print(len(int_data))

data.to_json('meta_data.json', orient='records')
train_data.to_json('train_data.json', orient='records')
val_data.to_json('val_data.json', orient='records')
test_data.to_json('test_data.json', orient='records')

with open('test_data.json', 'r') as f:
    data = json.load(f)

print(len(data))
print(data[0]['path'])