import torch

# import torch.nn as nn
# from torch.nn import functional as F

import numpy as np
import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

'''
def model_train(model,data,optimizer,device,EPOCH,loss_func,alpha=None,beta=None,vbeta=1):
    """
    function for training,(alpha,beta) for L2L1v2 regularizer
    """
    print(datetime.datetime.now())
    model=model.to(device)
    model=model.train()
    loss=0.0
    train_loss=0.0
    for epoch in range(EPOCH):
        #LRscheduler.step()
        for step, (x,y) in enumerate(data):
            #some preprocessing
            #x=preprocess_np(x,model,pre_method,region)
            x=torch.from_numpy(x).float()
            y=torch.from_numpy(y).float()
            b_x = x.to(device)
            b_y = y.to(device)
            if 'dn' in model.__class__.__name__:
                b_x=b_x+0.05*torch.randn(b_x.size()).to(device)
                b_x=torch.clamp(b_x,min=-1.0,max=1.0)
            elif 'vae' in model.__class__.__name__:
                encoded, mu, logvar, decoded = model(b_x)
                loss = loss_func(decoded, b_y, mu, logvar)
            elif 'L2L1' in model.__class__.__name__:
                encoded = model(b_x)
                loss=loss_func(decoded, b_y,alpha,beta,[model.conv1],encoded)
            #elif 'L2L1v2' in model.__class__.__name__:
            #    encoded, decoded = model(b_x)
            #    loss=loss_func(decoded, b_y,alpha,beta,[model.conv1],encoded)
            elif 'VanillaLN' in model.__class__.__name__:
                encoded = model(b_x)
                loss=loss_func(encoded, b_y,alpha,beta,[model.fc1],[model.fc1])
            elif 'Variational' in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood=\
                model.sample_elbo(b_x, b_y,vbeta=vbeta)
                #ratio=np.power(2,train_loader_length-step-1)/(np.power(2,train_loader_length)-1)
                #ratio=1e1#/train_loader_length
                #loss = (log_variational_posterior - log_prior)*ratio + negative_log_likelihood
            else:
                encoded = model(b_x)
                loss=loss_func(encoded, b_y,alpha,beta,[model.conv1,model.conv2],[model.fc1])

            #last epoch to get the training loss, keep the same sample size as validation
            if epoch==EPOCH-1:
                #if step % np.int16(train_loader_length/(val_loader_length/BATCH_SIZE)) ==0:
                #    train_loss=train_loss+loss.cpu().data.numpy()
                train_loss=train_loss+loss.cpu().data.numpy()
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 30 == 0:
                print('Model: ',model.__class__.__name__,'|Epoch: ', epoch,\
                      '| train loss: %.4f' % loss.cpu().data.numpy())
    print ('finish training!')
    print(datetime.datetime.now())
    #train_loss=train_loss/(val_loader_length/BATCH_SIZE)
    train_loss=train_loss/len(data)
    print ('Model: ',model.__class__.__name__,'|train loss: %.4f' % train_loss)
    return train_loss
    #winsound.Beep(400, 3000)#sound alarm when code finishes, freq=400, duration=3000ms
#_=model_train(ae3D_4,'normalize',sky',train_loader_shuffle,optimizer,device,EPOCH,loss_mse)
'''


def model_train(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    alpha=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            b_x = x.to(device)
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(
                    b_x, b_y, vbeta=vbeta / len(traindata)
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.conv1], [model.fc1])
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


def model_train_map(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    alpha=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            b_x = x.to(device)
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, negative_log_likelihood = model.map(
                    b_x, b_y
                )  # important to be divided by len(traindata)
            elif "MAP" in model.__class__.__name__:
                encoded, log_prior = model(b_x)
                loss = loss_func( encoded, b_y, log_prior, vbeta/len(traindata) )
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


"""
#use the trained model to test the validation loss
#show one example results
#val_eg: the example used to show results
#val_num: the number of validation dataset, when using gpu, may have memory problem, then set it small
def model_val(model,data,val_eg,device,loss_func):
    model=model.to(device)
    model=model.eval()
    #mse as metric,compute validation loss
    #loss_mse = nn.MSELoss()
    (x,y)=data
    x=torch.from_numpy(x).float()
    #y=torch.from_numpy(y).float()
    b_x = x.to(device)
    b_y = y.to(device)
    if 'dn' in model.__class__.__name__:
        b_x=b_x+0.05*torch.randn(b_x.size()).to(device)
        b_x=torch.clamp(b_x,min=-1.0,max=1.0)
    elif 'vae' in model.__class__.__name__:
        encoded, mu, logvar, decoded = model(b_x)
        val_loss = loss_func(decoded, b_y, mu, logvar)
    else:
        encoded = model(b_x)
        #val_loss=loss_func(encoded, b_y)
    #print ('Model: ',model.__class__.__name__,'|validation loss: %.4f' % val_loss.cpu().data.numpy())
    encoded_np=encoded.cpu().data.numpy()
    valcc,_=pearsonr(encoded_np.flatten(), y.flatten())
    print ('Model: ',model.__class__.__name__,'|validation cc: %.4f' % valcc)
    #
    #show one example
    fig,ax=plt.subplots(nrows=1, ncols=1,figsize=(10,2))
    ax.plot(data[1][:,val_eg],color='r',label='Target')
    ax.plot(encoded.cpu().data.numpy()[:,val_eg],color='g',label='Predict')
    ax.legend(loc='best',fontsize=12)
    #return val_loss.cpu().data.numpy()
    return valcc
#_=model_val(ae3D_4,'normalize','sky',val_loader_shuffle_sky,200,1000,device_cpu,loss_mse)

"""


def model_val(model, data, device, sample_num=10):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
    """
    model = model.to(device)
    model = model.eval()
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encodeds[i] = model(b_x, sampleFlag=True)
            encodeds[sample_num] = model(b_x, sampleFlag=False)
            encoded = encodeds.mean(0)
        elif "MAP" in model.__class__.__name__:
            encoded, _ = model(b_x)
        else:
            encoded = model(b_x)
    encoded_np = encoded.cpu().data.numpy()
    testcc, testpvalue = pearsonr(encoded_np.T.flatten(), y.T.flatten())
    return testcc


def model_predict(model, data, device, sample_num=10):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
    """
    model = model.to(device)
    model = model.eval()
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encoded = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encoded[i] = model(b_x, sampleFlag=True)
            encoded[sample_num] = model(b_x, sampleFlag=False)  # do not sample
            # encoded = encodeds.mean(0)
        else:
            encoded = model(b_x)
    encoded_np = encoded.cpu().data.numpy()
    # testcc,testpvalue=pearsonr(encoded_np.T.flatten(), y.T.flatten())
    return encoded_np, y

def model_predict_map(model, data, device, sample_num=10):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
    """
    model = model.to(device)
    model = model.eval()
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encoded = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encoded[i] = model(b_x, sampleFlag=True)
            encoded[sample_num] = model(b_x, sampleFlag=False)  # do not sample
            # encoded = encodeds.mean(0)
        elif "MAP" in model.__class__.__name__:
            encoded, _ = model(b_x)
        else:
            encoded = model(b_x)
    encoded_np = encoded.cpu().data.numpy()
    # testcc,testpvalue=pearsonr(encoded_np.T.flatten(), y.T.flatten())
    return encoded_np, y


# def model_test(model,data,device, sample_num=10):
#     """
#     Using pearson correlation as metric
#     Parameters:
#         sample_num: number of sample times, for Variational model
#     """
#     model=model.to(device)
#     model=model.eval()
#     (x,y)=data
#     x=torch.from_numpy(x).float()
#     b_x = x.to(device)
#     with torch.no_grad():
#         if 'Variational' in model.__class__.__name__:
#             encodeds = torch.zeros(sample_num+1, *(y.shape)).to(device)
#             for i in range(sample_num):
#                 encodeds[i] = model(b_x, sampleFlag=True)
#             encodeds[sample_num] = model(b_x, sampleFlag=False)
#             encoded = encodeds.mean(0)
#         else:
#             encoded = model(b_x)
#     encoded_np=encoded.cpu().data.numpy()
#     testcc,testpvalue=pearsonr(encoded_np.T.flatten(), y.T.flatten())
#     #show the best example
#     testccs=np.zeros(y.shape[1])
#     encoded_np=encoded_np+1e-5 #in case all zeros
#     for ii in range(len(testccs)):
#         testccs[ii],_=pearsonr(encoded_np[:,ii], y[:,ii])
#     test_best=np.argmax(testccs)
#     fig,ax=plt.subplots(nrows=1, ncols=1,figsize=(10,2))
#     ax.plot(data[1][:,test_best],color='r',linestyle='-',alpha=0.5,label='Target')
#     ax.plot(encoded_np[:,test_best],color='g',linestyle='-',alpha=0.5,label='Predict')
#     ax.legend(loc='best',fontsize=12)
#     print ('Overall pearson correlation coefficient: ',testcc, ' and p-value: ',testpvalue)
#     return testcc,testpvalue


def model_test(model, data, device, sample_num=10, use_pad0_sti=True):
    """
    Using pearson correlation as metric
    Parameters:
        sample_num: number of sample times, for Variational model
        use_pad0_sti: use reponses of 0-padding stimulus, for spatial and temporal model
    """
    model = model.to(device)
    model = model.eval()
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    if "3d" in model.__class__.__name__:
        with torch.no_grad():
            if "Variational" in model.__class__.__name__:
                encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
                for i in range(sample_num):
                    encodeds[i] = model(b_x, sampleFlag=True)
                encodeds[sample_num] = model(b_x, sampleFlag=False)
                encoded = encodeds.mean(0)
            else:
                encoded = model(b_x)
        encoded_np = encoded.cpu().data.numpy()
        if use_pad0_sti == False:  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
            dims = x.shape
            pad_id = dims[2] - 1
            encoded_np = encoded_np[pad_id:, :]
            y = y[pad_id:, :]
        testcc, testpvalue = pearsonr(encoded_np.T.flatten(), y.T.flatten())
        # show the best example
        testccs = np.zeros(y.shape[1])
        encoded_np = encoded_np + 1e-5  # in case all zeros
        for ii in range(len(testccs)):
            testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
        test_best = np.argmax(testccs)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        ax.plot(y[:, test_best], "o", color="r", linestyle="-", alpha=0.5, label="Target")
        ax.plot(encoded_np[:, test_best], "o", color="g", linestyle="-", alpha=0.5, label="Predict")
        ax.legend(loc="best", fontsize=12)
        print("Overall pearson correlation coefficient: ", testcc, " and p-value: ", testpvalue)

    else:
        with torch.no_grad():
            if "Variational" in model.__class__.__name__:
                encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
                for i in range(sample_num):
                    encodeds[i] = model(b_x, sampleFlag=True)
                encodeds[sample_num] = model(b_x, sampleFlag=False)
                encoded = encodeds.mean(0)
            else:
                encoded = model(b_x)
        encoded_np = encoded.cpu().data.numpy()
        testcc, testpvalue = pearsonr(encoded_np.T.flatten(), y.T.flatten())
        # show the best example
        testccs = np.zeros(y.shape[1])
        encoded_np = encoded_np + 1e-5  # in case all zeros
        for ii in range(len(testccs)):
            testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
        test_best = np.argmax(testccs)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        ax.plot(data[1][:, test_best], color="r", linestyle="-", alpha=0.5, label="Target")
        ax.plot(encoded_np[:, test_best], color="g", linestyle="-", alpha=0.5, label="Predict")
        ax.legend(loc="best", fontsize=12)
        print("Overall pearson correlation coefficient: ", testcc, " and p-value: ", testpvalue)
    return testcc, testpvalue


def model_train_sample_num(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    alpha=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
    sampling_num=2,
):
    """
    function for training variational model using different sampling numbers
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            b_x = x.to(device)
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(
                    b_x, b_y, vbeta=vbeta / len(traindata), sample_num=sampling_num
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.conv1], [model.fc1])
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def model_train_drop_out(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    drop_out_num,
    alpha=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training MC-dropout models
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            b_x = x.to(device)
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(
                    b_x, b_y, vbeta=vbeta / len(traindata)
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.conv1], [model.fc1])
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val_drop_out(model, valdata, valdevice, drop_out_num)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val_drop_out(model, valdata, valdevice, drop_out_num)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


def model_val_drop_out(model, data, device, drop_out_num=100, sample_num=10, plot_flag=False):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
        drop_out_num: number of dropout times
    """
    model = model.to(device)
    model = model.eval()
    enable_dropout(model)  # set dropout layers to train mode
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encodeds[i] = model(b_x, sampleFlag=True)
            encodeds[sample_num] = model(b_x, sampleFlag=False)
            encoded = encodeds.mean(0)
        else:
            ##output_list=[]
            encodeds = torch.zeros(drop_out_num, *(y.shape)).to(device)
            for i in range(drop_out_num):  # getting outputs for drop_out_num forward passes
                encodeds[i] = model(b_x)
                ##output_list.append(torch.unsqueeze(model(b_x), 0))
            ## output_mean = torch.cat(output_list, 0).mean(0)
            ##  encoded = torch.squeeze(output_mean)
            encoded = encodeds.mean(0)
    encoded_np = encoded.cpu().data.numpy()
    encoded_np = np.nan_to_num(encoded_np)
    testcc, testpvalue = pearsonr(encoded_np.T.flatten(), y.T.flatten())
    if plot_flag == True:
        # show the best example
        testccs = np.zeros(y.shape[1])
        encoded_np = encoded_np + 1e-5  # in case all zeros
        for ii in range(len(testccs)):
            testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
        test_best = np.argmax(testccs)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        ax.plot(data[1][:, test_best], color="r", linestyle="-", alpha=0.5, label="Target")
        ax.plot(encoded_np[:, test_best], color="g", linestyle="-", alpha=0.5, label="Predict")
        ax.legend(loc="best", fontsize=12)
        print("Overall pearson correlation coefficient: ", testcc, " and p-value: ", testpvalue)
    return testcc


def model_predict_dropout(model, data, device, drop_out_num, sample_num=10):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
    """
    model = model.to(device)
    model = model.eval()
    enable_dropout(model)
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encoded = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encoded[i] = model(b_x, sampleFlag=True)
            encoded[sample_num] = model(b_x, sampleFlag=False)
            # encoded = encodeds.mean(0)
        else:
            encoded = torch.zeros(drop_out_num, *(y.shape)).to(device)
            for i in range(drop_out_num):
                encoded[i] = model(b_x)
    encoded_np = encoded.cpu().data.numpy()
    # testcc,testpvalue=pearsonr(encoded_np.T.flatten(), y.T.flatten())
    return encoded_np, y


def model_test_save(model, data, device, sample_mean_Flag, sample_num=10, use_pad0_sti=True):
    """
    save CCs and stds for each neuron
    """
    model = model.to(device)
    model = model.eval()
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    if "3d" in model.__class__.__name__:
        with torch.no_grad():
            if "Variational" in model.__class__.__name__:
                if sample_mean_Flag == True:  # sample  sample_num+1  times
                    encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
                    for i in range(sample_num):
                        encodeds[i] = model(b_x, sampleFlag=True)
                    encodeds[sample_num] = model(b_x, sampleFlag=False)
                if sample_mean_Flag == False:  # sample sample_num  times
                    encodeds = torch.zeros(sample_num, *(y.shape)).to(device)
                    for i in range(sample_num):
                        encodeds[i] = model(b_x, sampleFlag=True)

                encoded = encodeds.mean(0)
                encoded_np = encoded.cpu().data.numpy()
                encodeds_np = encodeds.cpu().data.numpy()
                if (
                    use_pad0_sti == False
                ):  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
                    dims = x.shape
                    pad_id = dims[2] - 1
                    encoded_np = encoded_np[pad_id:, :]
                    encodeds_np = encodeds_np[:, pad_id:, :]
                    y = y[pad_id:, :]
                testccs = np.zeros(y.shape[1])
                encoded_np = encoded_np + 1e-5  # in case all zeros
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])

                testvars2 = np.var(encodeds_np, axis=0, ddof=1)
                testvars2 = np.mean(testvars2, axis=0)
                testccs_each_neuron = np.zeros((len(encodeds_np), y.shape[1]))
                for ii in range(len(encodeds_np)):
                    for jj in range(y.shape[1]):
                        testccs_each_neuron[ii, jj], _ = pearsonr((encodeds_np[ii, :, jj].flatten()), y[:, jj])
                testvars = np.var(testccs_each_neuron, axis=0, ddof=1)
                return testccs, testvars, testvars2

            else:
                encoded = model(b_x)
                encoded_np = encoded.cpu().data.numpy()
                if (
                    use_pad0_sti == False
                ):  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
                    dims = x.shape
                    pad_id = dims[2] - 1
                    encoded_np = encoded_np[pad_id:, :]
                    y = y[pad_id:, :]
                encoded_np = encoded_np + 1e-5  # in case all zeros
                testccs = np.zeros(y.shape[1])
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
                return testccs

    else:
        with torch.no_grad():
            if "Variational" in model.__class__.__name__:
                if sample_mean_Flag == True:  # sample  sample_num+1  times
                    encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
                    for i in range(sample_num):
                        encodeds[i] = model(b_x, sampleFlag=True)
                    encodeds[sample_num] = model(b_x, sampleFlag=False)
                if sample_mean_Flag == False:  # sample sample_num  times
                    encodeds = torch.zeros(sample_num, *(y.shape)).to(device)
                    for i in range(sample_num):
                        encodeds[i] = model(b_x, sampleFlag=True)

                encoded = encodeds.mean(0)
                encoded_np = encoded.cpu().data.numpy()
                testccs = np.zeros(y.shape[1])
                encoded_np = encoded_np + 1e-5  # in case all zeros
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])

                encodeds_np = encodeds.cpu().data.numpy()
                # print(encodeds_np.shape)
                # print (y.shape)
                testvars2 = np.var(encodeds_np, axis=0, ddof=1)
                testvars2 = np.mean(testvars2, axis=0)
                testccs_each_neuron = np.zeros((len(encodeds_np), y.shape[1]))
                for ii in range(len(encodeds_np)):
                    for jj in range(y.shape[1]):
                        testccs_each_neuron[ii, jj], _ = pearsonr((encodeds_np[ii, :, jj].flatten()), y[:, jj])
                testvars = np.var(testccs_each_neuron, axis=0, ddof=1)
                return testccs, testvars, testvars2

            else:
                encoded = model(b_x)
                encoded_np = encoded.cpu().data.numpy()
                encoded_np = encoded_np + 1e-5  # in case all zeros
                testccs = np.zeros(y.shape[1])
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
                return testccs


def model_test_save_dropout(model, data, device, dropout_num=100, use_pad0_sti=True):
    """
    save CCs and stds for each neuron
    """
    model = model.to(device)
    model = model.eval()
    enable_dropout(model)
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    if "3d" in model.__class__.__name__:
        with torch.no_grad():
            if "Dropout" in model.__class__.__name__:
                encodeds = torch.zeros(dropout_num, *(y.shape)).to(device)
                for i in range(dropout_num):
                    encodeds[i] = model(b_x)
                encoded = encodeds.mean(0)
                encoded_np = encoded.cpu().data.numpy()
                encodeds_np = encodeds.cpu().data.numpy()
                if (
                    use_pad0_sti == False
                ):  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
                    dims = x.shape
                    pad_id = dims[2] - 1
                    encoded_np = encoded_np[pad_id:, :]
                    encodeds_np = encodeds_np[:, pad_id:, :]
                    y = y[pad_id:, :]
                testccs = np.zeros(y.shape[1])
                encoded_np = encoded_np + 1e-5  # in case all zeros
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])

                testvars2 = np.var(encodeds_np, axis=0, ddof=1)
                testvars2 = np.mean(testvars2, axis=0)
                testccs_each_neuron = np.zeros((len(encodeds_np), y.shape[1]))
                for ii in range(len(encodeds_np)):
                    for jj in range(y.shape[1]):
                        testccs_each_neuron[ii, jj], _ = pearsonr((encodeds_np[ii, :, jj].flatten()), y[:, jj])
                testvars = np.var(testccs_each_neuron, axis=0, ddof=1)
                return testccs, testvars, testvars2
            else:
                encoded = model(b_x)
                encoded_np = encoded.cpu().data.numpy()
                if (
                    use_pad0_sti == False
                ):  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
                    dims = x.shape
                    pad_id = dims[2] - 1
                    encoded_np = encoded_np[pad_id:, :]
                    y = y[pad_id:, :]
                encoded_np = encoded_np + 1e-5  # in case all zeros
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
                return testccs
    else:
        with torch.no_grad():
            if "Dropout" in model.__class__.__name__:
                encodeds = torch.zeros(dropout_num, *(y.shape)).to(device)
                for i in range(dropout_num):
                    encodeds[i] = model(b_x)
                encoded = encodeds.mean(0)
                encoded_np = encoded.cpu().data.numpy()
                testccs = np.zeros(y.shape[1])
                encoded_np = encoded_np + 1e-5  # in case all zeros
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])

                encodeds_np = encodeds.cpu().data.numpy()
                # print(encodeds_np.shape)
                testvars2 = np.var(encodeds_np, axis=0, ddof=1)
                testvars2 = np.mean(testvars2, axis=0)
                testccs_each_neuron = np.zeros((len(encodeds_np), y.shape[1]))
                for ii in range(len(encodeds_np)):
                    for jj in range(y.shape[1]):
                        testccs_each_neuron[ii, jj], _ = pearsonr((encodeds_np[ii, :, jj].flatten()), y[:, jj])
                testvars = np.var(testccs_each_neuron, axis=0, ddof=1)
                return testccs, testvars, testvars2
            else:
                encoded = model(b_x)
                encoded_np = encoded.cpu().data.numpy()
                encoded_np = encoded_np + 1e-5  # in case all zeros
                for ii in range(len(testccs)):
                    testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
                return testccs


def model_train_regularization_2conv(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    alpha1=None,
    alpha2=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            b_x = x.to(device)
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(
                    b_x, b_y, vbeta=vbeta / len(traindata)
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha1, alpha2, beta, [model.conv1], [model.conv2], [model.fc1])
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


def model_val_3d(model, data, device, sample_num=10, use_pad0_sti=True):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
    """
    model = model.to(device)
    model = model.eval()
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encodeds[i] = model(b_x, sampleFlag=True)
            encodeds[sample_num] = model(b_x, sampleFlag=False)
            encoded = encodeds.mean(0)
        else:
            encoded = model(b_x)
    encoded_np = encoded.cpu().data.numpy()
    if use_pad0_sti == False:  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
        dims = x.shape
        pad_id = dims[2] - 1
        encoded_np = encoded_np[pad_id:, :]
        y = y[pad_id:, :]
    testcc, testpvalue = pearsonr(encoded_np.T.flatten(), y.T.flatten())
    return testcc


def model_val_drop_out_3d(model, data, device, drop_out_num=100, sample_num=10, plot_flag=False, use_pad0_sti=True):
    """
    Parameters:
        sample_num: number of sample times, for Variational model
        drop_out_num: number of dropout times
    """
    model = model.to(device)
    model = model.eval()
    enable_dropout(model)  # set dropout layers to train mode
    (x, y) = data
    x = torch.from_numpy(x).float()
    b_x = x.to(device)
    with torch.no_grad():
        if "Variational" in model.__class__.__name__:
            encodeds = torch.zeros(sample_num + 1, *(y.shape)).to(device)
            for i in range(sample_num):
                encodeds[i] = model(b_x, sampleFlag=True)
            encodeds[sample_num] = model(b_x, sampleFlag=False)
            encoded = encodeds.mean(0)
        else:
            ##output_list=[]
            encodeds = torch.zeros(drop_out_num, *(y.shape)).to(device)
            for i in range(drop_out_num):  # getting outputs for drop_out_num forward passes
                encodeds[i] = model(b_x)
                ##output_list.append(torch.unsqueeze(model(b_x), 0))
            ## output_mean = torch.cat(output_list, 0).mean(0)
            ##  encoded = torch.squeeze(output_mean)
            encoded = encodeds.mean(0)
    encoded_np = encoded.cpu().data.numpy()
    encoded_np = np.nan_to_num(encoded_np)
    if use_pad0_sti == False:  # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
        dims = x.shape
        pad_id = dims[2] - 1
        encoded_np = encoded_np[pad_id:, :]
        y = y[pad_id:, :]
    testcc, testpvalue = pearsonr(encoded_np.T.flatten(), y.T.flatten())
    if plot_flag == True:
        # show the best example
        testccs = np.zeros(y.shape[1])
        encoded_np = encoded_np + 1e-5  # in case all zeros
        for ii in range(len(testccs)):
            testccs[ii], _ = pearsonr(encoded_np[:, ii], y[:, ii])
        test_best = np.argmax(testccs)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        ax.plot(data[1][:, test_best], color="r", linestyle="-", alpha=0.5, label="Target")
        ax.plot(encoded_np[:, test_best], color="g", linestyle="-", alpha=0.5, label="Predict")
        ax.legend(loc="best", fontsize=12)
        print("Overall pearson correlation coefficient: ", testcc, " and p-value: ", testpvalue)
    return testcc


def model_train_mse(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    alpha=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    Compared to model_train(), this function uses sample_elbo_mse() instead of sample_elbo() for
    variational methods, i.e., using MSE loss instead of Poisson loss.
    This function is used for V4 data which has many repeats for both training data and test data.
    To evaluate the influence of response variability on neural prediction, this function may use
    different response values for each epoch when we have repeats of responses for training. This
    can be seen from the shape of y in traindata.

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            b_x = x.to(device)
            if len(y.shape) == 2:  # mean response
                y = torch.from_numpy(y).float()
            elif len(y.shape) == 3:  # response with repeats
                y = torch.from_numpy(y[:, :, epoch % y.shape[2]]).float()
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo_mse(
                    b_x, b_y, vbeta=vbeta / len(traindata)
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.conv1], [model.fc1])
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


def model_train_mse_drop_out(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    drop_out_num,
    alpha=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    Compared to model_train(), this function uses sample_elbo_mse() instead of sample_elbo() for
    variational methods, i.e., using MSE loss instead of Poisson loss.
    This function is used for V4 data which has many repeats for both training data and test data.
    To evaluate the influence of response variability on neural prediction, this function may use
    different response values for each epoch when we have repeats of responses for training. This
    can be seen from the shape of y in traindata.

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            b_x = x.to(device)
            if len(y.shape) == 2:  # mean response
                y = torch.from_numpy(y).float()
            elif len(y.shape) == 3:  # response with repeats
                y = torch.from_numpy(y[:, :, epoch % y.shape[2]]).float()
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo_mse(
                    b_x, b_y, vbeta=vbeta / len(traindata)
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.conv1], [model.fc1])
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val_drop_out(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val_drop_out(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses


def model_train_mse_regularization_3conv(
    model,
    traindata,
    optimizer,
    device,
    EPOCH,
    loss_func,
    alpha1=None,
    alpha2=None,
    alpha3=None,
    beta=None,
    vbeta=1,
    earlystop=False,
    valdata=None,
    valdevice=None,
    verbose=True,
):
    """
    function for training
    Parameters:
        (alpha,beta) for L2L1v2 regularizer
        vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
        kl divergence and negative log likelihood

    Compared to model_train(), this function uses sample_elbo_mse() instead of sample_elbo() for
    variational methods, i.e., using MSE loss instead of Poisson loss.
    This function is used for V4 data which has many repeats for both training data and test data.
    To evaluate the influence of response variability on neural prediction, this function may use
    different response values for each epoch when we have repeats of responses for training. This
    can be seen from the shape of y in traindata.

    """
    print(datetime.datetime.now())
    loss = 0.0
    trainlosses = np.zeros((EPOCH))  # train losses
    vallosses = np.zeros((EPOCH))  # save validation losses of all epochs until early stopping
    for epoch in range(EPOCH):
        model = model.to(device)
        model = model.train()
        # LRscheduler.step()
        for step, (x, y) in enumerate(traindata):
            x = torch.from_numpy(x).float()
            b_x = x.to(device)
            if len(y.shape) == 2:  # mean response
                y = torch.from_numpy(y).float()
            elif len(y.shape) == 3:  # response with repeats
                y = torch.from_numpy(y[:, :, epoch % y.shape[2]]).float()
            b_y = y.to(device)
            if "VanillaLN" in model.__class__.__name__:
                encoded = model(b_x)
                loss = loss_func(encoded, b_y, alpha, beta, [model.fc1], [model.fc1])
            elif "Variational" in model.__class__.__name__:
                loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo_mse(
                    b_x, b_y, vbeta=vbeta / len(traindata)
                )  # important to be divided by len(traindata)
            else:
                encoded = model(b_x)
                loss = loss_func(
                    encoded, b_y, alpha1, alpha2, alpha3, beta, [model.conv1], [model.conv2], [model.conv3], [model.fc1]
                )
            #
            # last epoch to get the training loss, keep the same sample size as validation
            trainlosses[epoch] = trainlosses[epoch] + loss.detach().clone().cpu().data.numpy()
            #
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #
            if step % 100 == 0 and verbose == True:
                print(
                    "Model: ",
                    model.__class__.__name__,
                    "|Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.cpu().data.numpy(),
                )
        # one epoch done
        if epoch > 10 and earlystop == True:  # early stopping check after each epoch, use CC as a metric
            temploss = model_val(model, valdata, valdevice)
            vallosses[epoch] = temploss
            if epoch - np.argmax(vallosses) > 4:
                break
        # test
        trainlosses[epoch] = trainlosses[epoch] / len(traindata)
        if earlystop == False:
            vallosses[epoch] = model_val(model, valdata, valdevice)
    print("Epoch: {:} val loss: {:.4f}, finish training!".format(epoch, vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses, vallosses
