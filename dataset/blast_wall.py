from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch



def make_dataset(data_root, txt, input_size, output_size, portion_train, portion_val):
    ######################################################################
    # split the data into train, val and test and do normalization on it #
    ######################################################################

    # make those variables visible globally
    global train_split
    global test_split
    global val_split
    global num_input
    global num_output

    # the dim of input
    num_input = input_size
    # the dim of output
    num_output = output_size
    # specify the file of train , val, test set.
    train_split = 'data_splits/' + txt.split('.')[0] + '_train.txt'
    test_split = 'data_splits/' + txt.split('.')[0] + '_test.txt'
    val_split = 'data_splits/' + txt.split('.')[0] + '_val.txt'

    # if the data is already splittd, use it! You dont need to split it once more
    if os.path.isfile(train_split) and os.path.isfile(test_split) \
            and os.path.isfile(val_split):

        print("Use Splitted Data")
        # remember to set the argument "new_split" to False， since we dont need to split it anymore
        # we also dont need to pass inputs and targets to the function. Set them to None
        return split_data(portion_train, portion_val, None, None, new_split=False)

    else:
    # otherwise, split the data
        # open the raw data file and read each line
        with open(os.path.join(data_root, txt), 'r') as d:
            lines = d.readlines()

        target = [] # an empty list for storing the targets
        input = [] # an empty list for storing the inputs
        for line in lines:
            line = eval(line.strip())
            line =  list(map(float, line))
            # split the line accoring the dim of inputs and outputs and append them to the corresponding lists.
            input.append(line[:num_input]) 
            target.append(line[num_input:(num_input + num_output)])
        # convert the data type to float
        target = np.array(target).astype(np.float32)
        input = np.array(input).astype(np.float32)
        # split the data now
        return split_data(portion_train, portion_val, input, target, new_split=True)


def split_data(portion_train, portion_val, input, target, new_split):
    # portion_train: the rate of the training set
    # portion_val: the rate of the validation set
    # input: input data
    # target: label data
    # new_split: whether splitting the data 

    if new_split:
        # number of totoal data
        num_sequences = input.shape[0]
        # the breakpoint/index of training set
        break_train = int(num_sequences * portion_train)
        # the breakpoint/index of validation set
        break_val = int(num_sequences * (portion_train + portion_val))
        # randomly permute the dataset before splitting
        splits = np.random.permutation(np.arange(num_sequences))
        # split the data; only two breakpoints are need to generate 3 datasets, i.e. train, val, test
        splits = np.split(splits, [break_train, break_val])
        # splits[0], splits[1], splits[2] now contain the indices of train, val, test set
        # map those indices to actual data by input[splits[0]], input[splits[1]], input[splits[2]]; similarly for the targets
        input_train, input_val, input_test = input[splits[0]], input[splits[1]], input[splits[2]]
        target_train, target_val, target_test = target[splits[0]], target[splits[1]], target[splits[2]]
        
        # open the train data file and write each data element
        with open(train_split, 'w') as tr:
            for i, (inp, tar) in enumerate(zip(input_train, target_train)):
                point = np.append(inp, tar)
                tr.write(str(point.tolist()) + '\n')
        # open the validation data file and write each data element
        with open(val_split, 'w') as val:
            for i, (inp, tar) in enumerate(zip(input_val, target_val)):
                point = np.append(inp, tar)
                val.write(str(point.tolist()) + '\n')
        # open the test data file and write each data element
        with open(test_split, 'w') as te:
            for i, (inp, tar) in enumerate(zip(input_test, target_test)):
                point = np.append(inp, tar)
                te.write(str(point.tolist()) + '\n')
    else:

        # if not splitting data / data is already splitted. Read them directly!
        with open(train_split, 'r') as tr, open(test_split, 'r') as te, open(val_split, 'r') as val:
            strlines = tr.readlines()
            stelines = te.readlines()
            vallines = val.readlines()
        # empty lists for storing inputs and targets for each type of dataset
        target_train = []
        target_test = []
        target_val = []
        input_train = []
        input_test = []
        input_val = []

        for line in strlines:
            # convert line from string to python list
            line = eval(line.strip())
            # map element in the list to float type
            line =  list(map(float, line))
            # append data to lists according to the dim of inputs and outputs
            input_train.append(line[:num_input])
            target_train.append(line[num_input:(num_input + num_output)])
        # similarly for the test set and validation set
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

        # convert all data to the type of np.float32, becaue we need to normalize data with packages accepting np.float32 data

        target_train = np.array(target_train).astype(np.float32)
        input_train = np.array(input_train).astype(np.float32)

        target_test = np.array(target_test).astype(np.float32)
        input_test = np.array(input_test).astype(np.float32)

        target_val = np.array(target_val).astype(np.float32)
        input_val = np.array(input_val).astype(np.float32)

    # data normalization

    target_scaler = StandardScaler()
    # fit the mean/std from the training set. you can only touch trainning set
    target_train = torch.from_numpy(target_scaler.fit_transform(target_train).astype(np.float32))
    # transform the test and val set
    target_test = torch.from_numpy(target_scaler.transform(target_test).astype(np.float32))
    target_val = torch.from_numpy(target_scaler.transform(target_val).astype(np.float32))

    # same for the targets
    input_scaler = StandardScaler()
    input_train = torch.from_numpy(input_scaler.fit_transform(input_train).astype(np.float32))
    input_test = torch.from_numpy(input_scaler.transform(input_test).astype(np.float32))
    input_val = torch.from_numpy(input_scaler.transform(input_val).astype(np.float32))
     
    # wrap the data with pytorch dataset
    return BlastWall(input_train, target_train), BlastWall(input_val, target_val), BlastWall(input_test, target_test)


# the pytorch dataset
class BlastWall(Dataset):

    def __init__(self, input, target):
        self.input = input
        self.target = target


    def __len__(self):
        return self.input.size()[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]
