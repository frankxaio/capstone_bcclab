import torch
import numpy as np
from sklearn.metrics import accuracy_score
from utility_tempose import adjust_lr

#import wandb

def train(model,loader,run,device,cfg):
    training_config = cfg['training']
    if cfg == None:
        print('Invalid training config')

    warmup_e = training_config['warm_up']
    total_epochs = training_config['epochs']
    learning_rate = training_config['learning_rate_start']
    learning_rate_min = training_config['learning_rate_min']
    model.to(device)

    if cfg['hyperparameters']['criterion'] == 'CE':   
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    if cfg['hyperparameters']['optimizer'] == 'Adam':   
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif cfg['hyperparameters']['optimizer'] == 'SGD':   
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #exp = wandb.init(project='Tempose-badmin', config=cfg) 
    #wandb.watch(model)
    if run =='train-val':
        trainloader, valloader = loader
    elif run == 'train':
        trainloader = loader 
    
    for epoch in range(0,total_epochs): ### Run the model for a certain number of iterations/epochs
        
        # 
        epLoss = 0
        epAcc = 0
        # 
        adjust_lr(optimizer,learning_rate,learning_rate_min,epoch,total_epochs,warmup_e)
        for X, pad, y in trainloader: ### Fetch a batch of training inputs
            X = tuple(map(lambda x: x.to(device), X))
            x = X[0].reshape(*X[0].shape[:3], (X[0].shape[3] * X[0].shape[4]))
            pos = X[1].reshape(*X[1].shape[:2], -1)
            sp = torch.cat([pos, X[2]], dim=-1)
            
            # print(x.shape)
            # print(pos.shape)
            # print(sp.shape)

            y = y.to(device)
            sp = sp.to(device)

            # print(f'x.shape = {x.shape}')

            yPred = model(x,sp,pad)

            
            # print("yPred shape:", yPred.shape)
            # print("y shape:", y.shape)
            # print("y min/max:", y.min().item(), y.max().item())
            # print("y dtype:", y.dtype)


            # ============================== TODO [Ex 3.7] ==============================
            # Compute the loss between yPred and y with `criterion` and store it in
            # `loss` (it is used below), then do the backward pass: clear the old
            # gradients, back-propagate, update the weights.
            # ===========================================================================
            raise NotImplementedError("TODO [Ex 3.7]")
            # 
            epLoss += loss.item()
            acc = accuracy_score(model.predict(x,sp,pad).numpy(),y.cpu().numpy())
                
            epAcc += acc
            if epoch % 50 == 0:
                print('[%d] trloss: %.3f tracc: %.3f' % (epoch + 1, epLoss/len(trainloader), epAcc/len(trainloader)))
        #wandb.log({'train_loss': epLoss/len(loader)})
        #wandb.log({'train_acc': epAcc/len(loader)})
    
        # 
        if run =='train-val':
            epLoss = 0
            epAcc = 0
            model.eval()
            for x, y,sp,pad in valloader: #### Fetch validation samples
                x = x.type(torch.FloatTensor).to(device)
                y = y.to(device)
                sp = sp.to(device)
                yPred = model(x,sp,pad)
                loss = criterion(yPred,y)
                epLoss += loss.item()

                acc = accuracy_score(model.predict(x,sp,pad).numpy(),y.cpu().numpy())
                epAcc += acc
            if epoch % 50 == 0:
                print('[%d] trloss: %.3f tracc: %.3f' % (epoch + 1, epLoss/len(valloader), epAcc/len(valloader)))
            #wandb.log({'val_loss': epLoss/len(loader)})
            #wandb.log({'val_acc': epAcc/len(loader)})
   
    ## savemodel 
    EPOCH = epoch
    PATH =  'model.pt'
    LOSS = loss
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, PATH)
    #wandb.save('model.pt')

    print('Finished Training')
    #wandb.finish()
