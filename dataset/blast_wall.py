from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch



def make_dataset(data_root, txt, input_size, output_size, portion_train, portion_val):
    global train_split
    global test_split
    global val_split
    global num_input
    global num_output
    num_input = input_size
    num_output = output_size
    train_split = 'data_splits/' + txt.split('.')[0] + '_train.txt'
    test_split = 'data_splits/' + txt.split('.')[0] + '_test.txt'
    val_split = 'data_splits/' + txt.split('.')[0] + '_val.txt'


    if os.path.isfile(train_split) and os.path.isfile(test_split) \
            and os.path.isfile(val_split):

        print("Use Splitted Data")

        return split_data(portion_train, portion_val, None, None, new_split=False)

    else:
        with open(os.path.join(data_root, txt), 'r') as d:
            lines = d.readlines()

        target = []
        input = []
        for line in lines:
            line = eval(line.strip())
            line =  list(map(float, line))
            input.append(line[:num_input])
            target.append(line[num_input:(num_input + num_output)])

        target = np.array(target).astype(np.float32)
        input = np.array(input).astype(np.float32)

        return split_data(portion_train, portion_val, input, target, new_split=True)


def split_data(portion_train, portion_val, input, target, new_split):


    if new_split:
        num_sequences = input.shape[0]
        break_train = int(num_sequences * portion_train)
        break_val = int(num_sequences * (portion_train + portion_val))
        splits = np.random.permutation(np.arange(num_sequences))
        splits = np.split(splits, [break_train, break_val])
        input_train, input_val, input_test = input[splits[0]], input[splits[1]], input[splits[2]]
        target_train, target_val, target_test = target[splits[0]], target[splits[1]], target[splits[2]]

        with open(train_split, 'w') as tr:
            for i, (inp, tar) in enumerate(zip(input_train, target_train)):
                point = np.append(inp, tar)
                tr.write(str(point.tolist()) + '\n')

        with open(val_split, 'w') as val:
            for i, (inp, tar) in enumerate(zip(input_val, target_val)):
                point = np.append(inp, tar)
                val.write(str(point.tolist()) + '\n')

        with open(test_split, 'w') as te:
            for i, (inp, tar) in enumerate(zip(input_test, target_test)):
                point = np.append(inp, tar)
                te.write(str(point.tolist()) + '\n')
    else:
        with open(train_split, 'r') as tr, open(test_split, 'r') as te, open(val_split, 'r') as val:
            strlines = tr.readlines()
            stelines = te.readlines()
            vallines = val.readlines()

        target_train = []
        target_test = []
        target_val = []
        input_train = []
        input_test = []
        input_val = []

        for line in strlines:
            line = eval(line.strip())
            line =  list(map(float, line))
            input_train.append(line[:num_input])
            target_train.append(line[num_input:(num_input + num_output)])
        for line in stelines:
            line = eval(line.strip())
            line =  list(map(float, line))
            input_test.append(line[:num_input])
            target_test.append(line[num_input:(num_input + num_output)])
        for line in vallines:
            line = eval(line.strip())
            line =  list(map(float, line))
            input_val.append(line[:num_input])
            target_val.append(line[num_input:(num_input + num_output)])



        target_train = np.array(target_train).astype(np.float32)
        input_train = np.array(input_train).astype(np.float32)

        target_test = np.array(target_test).astype(np.float32)
        input_test = np.array(input_test).astype(np.float32)

        target_val = np.array(target_val).astype(np.float32)
        input_val = np.array(input_val).astype(np.float32)

    target_scaler = StandardScaler()
    target_train = torch.from_numpy(target_scaler.fit_transform(target_train).astype(np.float32))
    target_test = torch.from_numpy(target_scaler.transform(target_test).astype(np.float32))
    target_val = torch.from_numpy(target_scaler.transform(target_val).astype(np.float32))

    input_scaler = StandardScaler()
    input_train = torch.from_numpy(input_scaler.fit_transform(input_train).astype(np.float32))
    input_test = torch.from_numpy(input_scaler.transform(input_test).astype(np.float32))
    input_val = torch.from_numpy(input_scaler.transform(input_val).astype(np.float32))

    return BlastWall(input_train, target_train), BlastWall(input_val, target_val), BlastWall(input_test, target_test)


class BlastWall(Dataset):

    def __init__(self, input, target):
        self.input = input
        self.target = target


    def __len__(self):
        return self.input.size()[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]
