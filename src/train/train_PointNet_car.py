import argparse
import os, sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import gitpath
HOME_PATH = gitpath.root()
sys.path.append(HOME_PATH)

# Import pytorch dependencies
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

from pytorch3d.loss import chamfer_distance

# Import toolkits
from utils.visualization_3D_objects import *
from Hyperparameters import *
from pointnetae import *

def train(params: Hyperparameters, silent = False):
    #load dataset
    f1 = open(os.path.join(params.DatasetHyperparams.DATA_PATH,'train.txt'),'rb')
    X_train = pickle.load(f1)
    original_size = len(X_train)
    X_train = X_train[:(int)(params.DatasetHyperparams.N_SAMPLE_RATE * len(X_train))]
    f2 = open(os.path.join(params.DatasetHyperparams.DATA_PATH,'test.txt'),'rb')
    X_test = pickle.load(f2)
    print("Training set size: %d, Original Size: %d"%(len(X_train), original_size))
    print("Testing set size: ", len(X_test))

    # construct dataloader
    train_loader = DataLoader(
        X_train, 
        batch_size=params.DatasetHyperparams.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        X_test, 
        batch_size=params.DatasetHyperparams.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )

    # GPU check                
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    # Model Definition  
    model = PointNet_AE(params.StructureHyperparams.TYPE, params.StructureHyperparams.STN3D)
    model = model.to(device)
    summary(model, (3,2048))


    # Check if on GPU
    assert(next(model.parameters()).is_cuda)

    # create loss function: Chamfer distance
    criterion = lambda recon_x, x: chamfer_distance(recon_x, x)

    # Add optimizer
    step = params.TrainingHyperparams.DECAY_STEP
    epochs = params.TrainingHyperparams.EPOCHS
    init_lr = params.TrainingHyperparams.INIT_LR
    decay_rate = params.TrainingHyperparams.DECAY_RATE
    decay_epochs = np.arange(step, step * (np.round(epochs/step)), step)

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_rate, verbose=True)

    # alias for logging params
    CHECKPOINT_PATH = params.LoggingHyperparams.CHECKPOINT_PATH
    MODEL_NAME = params.LoggingHyperparams.MODEL_NAME
    RESUME = params.LoggingHyperparams.RESUME
    LOG_EPOCHS = params.LoggingHyperparams.LOG_EPOCHS

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(CHECKPOINT_PATH + '/runs')

    best_loss = 1e20
    START_EPOCH = 0

    # store loss learning curve
    train_loss_lst = []
    valid_loss_lst = []

    # resume from last checkpoint
    if RESUME:
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, MODEL_NAME + '.pth'))

        model.load_state_dict(checkpoint['state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']) 
        
        START_EPOCH = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

        train_loss_lst = checkpoint['train_loss_lst']
        valid_loss_lst = checkpoint['train_loss_lst']

        for i in range(len(train_loss_lst)): 
            writer.add_scalar('training loss', train_loss_lst[i], i)
            writer.add_scalar('validation loss', valid_loss_lst[i], i)
        
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print("==> Resuming from epoch: %d, learning rate %.4f, best loss: %.4f"%(START_EPOCH, current_learning_rate, best_loss))
        print("="*50)


    # start the training/validation process
    start = time.time()
    print("==> Training starts!")
    print("="*50)                

    for i in range(START_EPOCH, epochs):    
        # switch to train mode
        model.train()
        
        print("Epoch %d:" %i)

        train_loss = 0 # track training loss if you want
        
        # Train the model for 1 epoch.
        for batch_idx, (inputs) in enumerate(train_loader):
            # copy inputs to device
            inputs = inputs.float().to(device)

            # compute the output and loss
            outputs, _ = model(inputs)

            inputs = inputs.transpose(1, 2)
            outputs = outputs.transpose(1, 2)
            dist1, _ = criterion(outputs, inputs)

            loss = (torch.mean(dist1))
            train_loss += loss.to('cpu').detach().numpy()

            # zero the gradient
            optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # apply gradient and update the weights
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_lst.append(avg_train_loss)
        writer.add_scalar('training loss', avg_train_loss, i)
        # switch to eval mode
        model.eval()

        # this help you compute the validation accuracy
        total_examples = 0
        correct_examples = 0
        
        val_loss = 0 # again, track the validation loss if you want
        
        # disable gradient during validation, which can save GPU memory
        with torch.no_grad():
            for batch_idx, (inputs) in enumerate(val_loader):
                
                inputs = inputs.float().to(device)
                outputs, _ = model(inputs)

                inputs = inputs.transpose(1, 2)
                outputs = outputs.transpose(1, 2)

                dist1, _ = criterion(outputs, inputs)

                loss = (torch.mean(dist1))
                val_loss += loss
                
                # log first batch outcome plots
                if (batch_idx == 0 and i % LOG_EPOCHS == 0):
                    print("Logging generate shape quality")
                    fig = comparePointClouds(inputs.cpu().detach().numpy()[0].T, outputs.cpu().detach().numpy()[0].T)
                    save3DPointsImage(fig, save_path = os.path.join(CHECKPOINT_PATH, "image"), title = f"epoch_{i:03d}")
                    
                    # img = imread(os.path.join("image", MODEL_NAME, "epoch_%d"%i + ".png"))
                    # writer.add_image("epoch_%d"%i + ".png", img, i)


        avg_val_loss = val_loss / len(val_loader)    
        valid_loss_lst.append(avg_val_loss.cpu().detach().numpy())
        writer.add_scalar('validation loss', avg_val_loss, i)
        
        print("Training loss: %.4f, Validation loss: %.4f" % (avg_train_loss, avg_val_loss))

        # save the model checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            print("Saving Model...")
            state = {'state_dict': model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': i,
                    'best_loss': best_loss,
                    'train_loss_lst': train_loss_lst,
                    'valid_loss_lst': valid_loss_lst}
            torch.save(state, os.path.join(CHECKPOINT_PATH, 'model.pth'))
            
        print('')

        scheduler.step()

    print("="*50)
    print(f"==> Optimization finished in {time.time() - start:.2f}s! Best validation loss: {best_loss:.4f}")

    plt.plot(train_loss_lst, label='Train Loss')
    plt.plot(valid_loss_lst, label='Test Loss')
    plt.title(MODEL_NAME + " Learning Curve")
    plt.legend()
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(CHECKPOINT_PATH, 'learning_curve.png'))

    del model, train_loader, val_loader
    torch.cuda.empty_cache()
