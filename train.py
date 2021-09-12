import sys
import os
import numpy as np
import random
import math
from scipy import stats
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import time
from args import Config
from dataset.blast_wall import make_dataset
from models.model import RegressionModel
from utils.summary import TensorboardSummary
from utils.utils import str2bool
from utils.utils import transfer_optim_state
from utils.utils import time_str
from utils.utils import save_ckpt, load_ckpt
from utils.utils import load_state_dict
from utils.utils import ReDirectSTD
from utils.utils import set_devices
from utils.utils import AverageMeter
from utils.utils import to_scalar
from utils.utils import may_set_mode
from utils.utils import may_mkdir
from utils.utils import set_seed
from sklearn.metrics import r2_score

# parsed arguments; see args.py
cfg = Config()
# log
if cfg.log_to_file:
    # redirect the standard outputs/error to local txt files.
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

summary = TensorboardSummary(cfg.exp_dir)
writer = summary.create_summary()

# dump the configuration to log.
import pprint
print(('-' * 60))
print('cfg.__dict__')
pprint.pprint(cfg.__dict__)
print(('-' * 60))

# set the random seed
if cfg.set_seed:
    set_seed(cfg.rand_seed)

# init devices
set_devices(cfg.sys_device_ids) # gpu runnable?

# Dataset
## split the rawdata to train, val and test accoring to rates set in arguments
train_set, val_set, test_set = make_dataset(cfg.data_root, cfg.raw_data, cfg.input_size, \
                                   cfg.output_size, cfg.portion_train, cfg.portion_val)

                            # 0.4/0.3/0.3 ; 0.5/0.25/0.25 10 settings

# pytoch dataloader. you can get data in batches
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=cfg.batch_size,
    shuffle=True, # whether shuffle the dataset
    num_workers=cfg.workers,
    drop_last=False) # drop last test
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=cfg.batch_size,
    num_workers=cfg.workers,
    drop_last=False)
val_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=cfg.batch_size,
    num_workers=cfg.workers,
    drop_last=False)


### model ###
model = RegressionModel(**cfg.model_kwargs)
# Loss function
criterion = nn.MSELoss()
# Optimizer
optimizer = optim.Adamax(model.parameters(), lr=cfg.lr)
# Learning rate scheduler; change the learning rate dynamically
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = cfg.epochs_per_decay, \
                                            gamma=cfg.staircase_decay_multiple_factor)

# bind the model and optimizer; can be moved to cpu or gpu togethre
modules_optims = [model, optimizer]

# Load existing model weight or not
if cfg.load_model_weight:
    # load the model weights, optimizer, start epoch from the saved model files
    start_epoch, scores = load_ckpt(modules_optims, cfg.ckpt_file)
else:
    start_epoch = 0

# transfer model to cpu or gpu; -1 for cpu; >=0 for gpu
transfer_optim_state(state=optimizer.state, device_id=-1) 

LowestLoss = math.inf # the lowest mean squared error
HighestR2 = 0 # the highest r^2 value
best_epoch = 0 # the index of the epoch achieving the best performance


# iterate over all epochs
for epoch in range(start_epoch, cfg.total_epochs):
    # set the model to "train" for training
    may_set_mode(modules_optims, 'train')
    # recording loss, time
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    dataset_L = len(train_loader)
    ep_st = time.time()

    # iterate over all all batches in one batch
    for step, (sample_train, target_train) in enumerate(train_loader):
        # current time
        step_st = time.time()

        # forward pas
        score = model(sample_train)


        # calculate loss
        loss = criterion(score, target_train)
        # zero the gradient buffer before calculating the gradients in the current step.
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update weights; a gradient descent step
        optimizer.step()

        ############
        # step log #
        ############
        # log loss for this batch
        loss_meter.update(to_scalar(loss))
        # write to tensorboard; used for visualization
        writer.add_scalar('train/total_loss_iter', to_scalar(loss), step + 1 + dataset_L * epoch)
        # print the log for every "steps_per_log" batches or the final batch
        if (step+1) % cfg.steps_per_log == 0 or (step+1)%len(train_loader) == 0:
            log = '{}, Step {}/{} in Ep {}, {:.2f}s, loss:{:.4f}'.format( \
            time_str(), step+1, dataset_L, epoch+1, time.time()-step_st, loss_meter.val)
            print(log)

    # update the learning rate
    scheduler.step()
    ##############
    # epoch log  #
    ##############
    # add tensorboard log for this epoch
    writer.add_scalar('train/total_avgloss_epoch', loss_meter.avg, epoch + 1)
    # print the log for this epoch
    log = 'Ep{}, {:.2f}s, loss {:.4f}'.format(
        epoch+1, time.time() - ep_st, loss_meter.avg)
    print(log)

    # Average epoch time updates
    time_meter.update(time.time() - ep_st)
    # save model weights for every "epochs_per_save" epochs
    if (epoch + 1) % cfg.epochs_per_save == 0 or epoch+1 == cfg.total_epochs:
        ckpt_file = os.path.join(cfg.exp_dir, 'model', 'ckpt_epoch%d.pth'%(epoch+1))
        save_ckpt(modules_optims, epoch+1, 0, ckpt_file)

    ##########################
    # test on val set #
    ##########################
    if (epoch + 1) % cfg.epochs_per_val == 0 or (epoch+1) == cfg.total_epochs:
        print('test on valset')
        # set the model to "eval" for testing
        may_set_mode(modules_optims, 'eval')
        testloss_meter = AverageMeter()
        pred_list = [] # a list for storing predictions for the whole validation set; initialized with an empty list.
        target_list = [] # a list for storing targets for the whole validation set; initialized with an empty list
        # iterate over all batches in validation set
        for i, (sample_test, target_test) in enumerate(val_loader):
            # use torch.no_grad(): because we dont need to calculate gradients(backpropagate) for testing; 
            with torch.no_grad():
                output = model(sample_test)
            test_loss = criterion(output, target_test)
            testloss_meter.update(to_scalar(test_loss))
            # append predictions of current batch to the list
            pred_list.extend(output.tolist())
            # append targets of the current batch to the list
            target_list.extend(target_test.tolist())
        # calculate r^2 of validation set
        r2 = r2_score(target_list, pred_list)
        # the average loss of validation set
        avgloss = testloss_meter.avg
        # decide the best epoch
        if avgloss < LowestLoss:
            LowestLoss = avgloss
            HighestR2 = r2
            best_epoch = epoch + 1
        print('-' * 60)
        print('Mean Squared Errors: %.4f\n'%(avgloss))
        print('R squared: %.4f\n'%(r2))
        print('-' * 60)
        # write to tensorboard; for visualization
        writer.add_scalar('val/r2', r2, epoch+1)
        writer.add_scalar('val/total_avgloss_epoch', avgloss, epoch+1)

##########################
# Best Accuracy #
##########################
log = 'Lowest Loss: {:.3f} with R2 : {:.3f} on epoch {}'.format(LowestLoss, HighestR2, best_epoch)
print(log)


##########################
# Test on final test set #
##########################

print(('-' * 60))
# load the model from the best epoch during training phase
e, s = load_ckpt(modules_optims, os.path.join(cfg.exp_dir, 'model', 'ckpt_epoch{}.pth'.format(best_epoch)))
print('test on testset')
# set the model to "eval" mode for testing
may_set_mode(modules_optims, 'eval')
# for tracking the loss on test set
testloss_meter = AverageMeter()
pred_list = [] # a list for storing predictions for the whole test set; initialized with an empty list.
target_list = [] # a list for storing targets for the whole test set; initialized with an empty list
# Iterating the whole test loader
for i, (sample_test, target_test) in enumerate(test_loader):
    with torch.no_grad():
        output = model(sample_test)
    # calculate the loss for current batch
    test_loss = criterion(output, target_test)
    testloss_meter.update(to_scalar(test_loss))
    # append predictions of current batch to the list
    pred_list.extend(output.tolist())
    # append targets of the current batch to the list
    target_list.extend(target_test.tolist())
# calculate r^2 on the whole test set
r2 = r2_score(target_list, pred_list)
avgloss = testloss_meter.avg
print('-' * 60)
print('Mean Epoch Time: %.4f\n'%(time_meter.avg))
print('Converge Time: %.4f\n'%(time_meter.avg * best_epoch))
print('Mean Squared Erros: %.4f\n'%(avgloss))
print('R squared: %.4f\n'%(r2))
print('-' * 60)

