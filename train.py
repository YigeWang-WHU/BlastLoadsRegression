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
cfg = Config()
# log
if cfg.log_to_file:
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
set_devices(cfg.sys_device_ids)

# Dataset

train_set, val_set, test_set = make_dataset(cfg.data_root, cfg.raw_data, cfg.input_size, \
                                   cfg.output_size, cfg.portion_train, cfg.portion_val)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.workers,
    drop_last=False)
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
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = cfg.epochs_per_decay, \
                                            gamma=cfg.staircase_decay_multiple_factor)

# bind the model and optimizer
modules_optims = [model, optimizer]

### Load existing model or not ###
if cfg.load_model_weight:
    # store the model, optimizer, epoch
    start_epoch, scores = load_ckpt(modules_optims, cfg.ckpt_file)
else:
    start_epoch = 0

transfer_optim_state(state=optimizer.state, device_id=-1)

LowestLoss = math.inf
HighestR2 = 0
best_epoch = 0


for epoch in range(start_epoch, cfg.total_epochs):
    # adjust the learning rate

    may_set_mode(modules_optims, 'train')
    # recording loss
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    dataset_L = len(train_loader)
    ep_st = time.time()

    for step, (sample_train, target_train) in enumerate(train_loader):

        step_st = time.time()

        score = model(sample_train)


        # loss for regression
        loss = criterion(score, target_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        ############
        # step log #
        ############

        loss_meter.update(to_scalar(loss))
        writer.add_scalar('train/total_loss_iter', to_scalar(loss), step + 1 + dataset_L * epoch)
        if (step+1) % cfg.steps_per_log == 0 or (step+1)%len(train_loader) == 0:
            log = '{}, Step {}/{} in Ep {}, {:.2f}s, loss:{:.4f}'.format( \
            time_str(), step+1, dataset_L, epoch+1, time.time()-step_st, loss_meter.val)
            print(log)
    scheduler.step()
    ##############
    # epoch log  #
    ##############
    writer.add_scalar('train/total_avgloss_epoch', loss_meter.avg, epoch + 1)
    log = 'Ep{}, {:.2f}s, loss {:.4f}'.format(
        epoch+1, time.time() - ep_st, loss_meter.avg)
    print(log)

    # Average epoch time updates
    time_meter.update(time.time() - ep_st)
    # model ckpt
    if (epoch + 1) % cfg.epochs_per_save == 0 or epoch+1 == cfg.total_epochs:
        ckpt_file = os.path.join(cfg.exp_dir, 'model', 'ckpt_epoch%d.pth'%(epoch+1))
        save_ckpt(modules_optims, epoch+1, 0, ckpt_file)

    ##########################
    # test on test set #
    ##########################
    if (epoch + 1) % cfg.epochs_per_val == 0 or (epoch+1) == cfg.total_epochs:
        print('test on valset')
        may_set_mode(modules_optims, 'eval')
        testloss_meter = AverageMeter()
        pred_list = []
        target_list = []
        for i, (sample_test, target_test) in enumerate(val_loader):

            with torch.no_grad():
                output = model(sample_test)
            test_loss = criterion(output, target_test)
            testloss_meter.update(to_scalar(test_loss))
            pred_list.extend(output.tolist())
            target_list.extend(target_test.tolist())
        r2 = r2_score(target_list, pred_list)
        avgloss = testloss_meter.avg
        if avgloss < LowestLoss:
            LowestLoss = avgloss
            HighestR2 = r2
            best_epoch = epoch + 1
        print('-' * 60)
        print('Mean Squared Erros: %.4f\n'%(avgloss))
        print('R squared: %.4f\n'%(r2))
        print('-' * 60)
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
e, s = load_ckpt(modules_optims, os.path.join(cfg.exp_dir, 'model', 'ckpt_epoch{}.pth'.format(best_epoch)))
print('test on testset')
may_set_mode(modules_optims, 'eval')
testloss_meter = AverageMeter()
pred_list = []
target_list = []
for i, (sample_test, target_test) in enumerate(test_loader):
    with torch.no_grad():
        output = model(sample_test)
    test_loss = criterion(output, target_test)
    testloss_meter.update(to_scalar(test_loss))
    pred_list.extend(output.tolist())
    target_list.extend(target_test.tolist())
r2 = r2_score(target_list, pred_list)
avgloss = testloss_meter.avg
print('-' * 60)
print('Mean Epoch Time: %.4f\n'%(time_meter.avg))
print('Converge Time: %.4f\n'%(time_meter.avg * best_epoch))
print('Mean Squared Erros: %.4f\n'%(avgloss))
print('R squared: %.4f\n'%(r2))
print('-' * 60)

