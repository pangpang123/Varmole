from torch._C import device
import warnings

#from Varmole import X_test
warnings.filterwarnings("ignore")

import torch
import math
import pandas as pd
import numpy as np
import scipy as sp
np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,recall_score,precision_score,log_loss,roc_auc_score,roc_curve

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('pdf')
plt.style.use('ggplot')
#%%matplotlib inline

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from sampler import ImbalancedDatasetSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess(x, y):
    return x.float().to(device), y.int().reshape(-1, 1).to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

class GRNeQTL(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GRNeQTL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        return input.matmul(self.weight.t() * adj) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Net(nn.Module):
    def __init__(self, adj, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        self.adj = adj
        self.GRNeQTL = GRNeQTL(D_in, H1)
        #self.dropout1 = torch.nn.Dropout(0)
        #self.dropout = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(H1, H2)
        #self.dropout2 = torch.nn.Dropout(0.25)
        self.linear3 = torch.nn.Linear(H2, H3)
        #self.dropout3 = torch.nn.Dropout(0.05)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        print("x shape", x.shape)
        cov_val = x[0:3,:]
        xx = x[3:,:]
        h1 = self.GRNeQTL(xx, self.adj).relu()
        print("h1.shape", h1.shape)
        h1 = np.concatenate((h1, cov_val), axis=1)
        print("h1.shape", h1.shape)
        #h1 = self.dropout1(h1)
		#h1a = self.dropout(h1)
        h2 = self.linear2(h1).relu()
        #h2 = self.dropout2(h2)
        h3 = self.linear3(h2).relu()
        #h3 = self.dropout3(h3)
        y_pred = self.linear4(h3).sigmoid()
        return y_pred

def loss_batch(model, loss_fn, xb, yb, opt=None):
    yhat = model(xb)
    loss = loss_fn(yhat, yb.float())
    for param in model.parameters():
            loss += L1REG * torch.sum(torch.abs(param))
            #loss += torch.sum(torch.abs(param))

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    yhat_class = np.where(yhat.detach().cpu().numpy()<0.5, 0, 1)
    #accuracy = balanced_accuracy_score(yb.detach().cpu().numpy(), yhat_class)
    accuracy = accuracy_score(yb.detach().cpu().numpy(), yhat_class)

    return loss.item(), accuracy

def fit(epochs, model, loss_fn, opt, train_dl, val_dl):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    for epoch in range(epochs):
        model.train()
        losses, accuracies = zip(
            *[loss_batch(model, loss_fn, xb, yb, opt) for xb, yb in train_dl]
        )
        train_loss.append(np.mean(losses))
        train_accuracy.append(np.mean(accuracies))

        model.eval()
        with torch.no_grad():
            losses, accuracies = zip(
                *[loss_batch(model, loss_fn, xb, yb) for xb, yb in val_dl]
            )
        val_loss.append(np.mean(losses))
        val_accuracy.append(np.mean(accuracies))
        
        if (epoch % 10 == 0):
            print("epoch %s" %epoch, np.mean(losses),np.mean(train_accuracy), np.mean(accuracies))
    
    return train_loss, train_accuracy, val_loss, val_accuracy

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

					#x = pd.read_csv(r'varmole_data.csv')
					#X_normalized = StandardScaler().fit_transform(x.values)
					#X = pd.DataFrame(X_normalized)
##-------------------------------------------------------------------#####
""" Use this if what to split data using Sklearn train_test_split function """
Tau=pd.read_csv(r'tau_01_15_2022.csv')
x=Tau.iloc[:,4:89]
#x = pd.read_csv(r'new_data.csv')
X_normalized = StandardScaler().fit_transform(x.values)
cov=np.array(Tau.iloc[:,2:4])
xx=np.concatenate((cov,X_normalized),axis=1)
#X = pd.DataFrame(X_normalized)
X = pd.DataFrame(xx)

x = np.array(x)
y=Tau.iloc[:,1]
#y = pd.read_csv(r'label_ad.csv')
X_train, X_val, y_train, y_val = train_test_split(X.values, np.reshape(y.values, (-1, 1)), test_size=0.20, random_state=73)

cov_train=X_train[:,0:2]
cov_val=X_val[:,0:2]
X_train=X_train[:,2:86]
X_val=X_val[:,2:86]

##-------------------------------------------------------------------####
#y = np.array(y)
##-------------------------------------------------------------------#####
""" Adj matrix to define relation between input and transparent layer """
adj_in = pd.read_csv(r'adj_0.5.csv')
adj_in = adj_in.set_index('probe')
adj = np.array(adj_in)
##------------------------------------------------------------------####

#X_train, X_test, y_train, y_test = train_test_split(X.values, np.reshape(y.values, (-1, 1)), test_size=0.10, random_state=73)
#X_train, X_val, y_train, y_val = train_test_split(X.values, np.reshape(y.values, (-1, 1)), test_size=0.20, random_state=73)
##-----------------------------------------------------------------#####
""" Use this If data is already split 
X_train = pd.read_csv('X_train.csv',header=None)
X_val = pd.read_csv('X_test.csv',header=None)
y_train = pd.read_csv('y_train.csv',header=None)
y_val = pd.read_csv('y_test.csv',header=None)
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
"""
##----------------------------------------------------------------#####
#trainx = pd.DataFrame(X_train)
#trainy = pd.DataFrame(y_train)
#testx = pd.DataFrame(X_val)
#testy = pd.DataFrame(y_val)
#trainx.to_csv('X_train.csv')
#trainy.to_csv('y_train.csv')
#testx.to_csv('X_test.csv')
#testy.to_csv('y_test.csv')

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=73)1
##------------------------------------------------------------------------------------------------------#####

X_train, y_train, X_val_t, y_val_t = map(torch.tensor, (X_train, y_train, X_val, y_val))

BS = 53

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val_t, y_val_t)


train_dl = DataLoader(dataset=train_ds, sampler = ImbalancedDatasetSampler(train_ds), batch_size=BS)
val_dl = DataLoader(dataset=val_ds,batch_size=27)

train_dl = WrappedDataLoader(train_dl, preprocess)
val_dl = WrappedDataLoader(val_dl, preprocess)

##--------------------------------------------------------------------------------#####
#train_dl = DataLoader(train_dl)
#val_dl = DataLoader(val_dl)
#device = 'cpu'
##--------------------------------------------------------------------------------######
""" Define the shape of the netwoek  """
D_in, H1, H2, H3, D_out = X_train.shape[1], adj.shape[1]+2, 60, 20, 1
a = torch.from_numpy(adj).float().to(device)
#a = torch.from_numpy(adj.todense()).float().to(device)
model = Net(a, D_in, H1, H2, H3, D_out).to(device)
##-------------------------------------------------------------------------------######

L1REG = 0.0001
#loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.BCELoss()
#loss_fn = nn.BCELoss()
L2REG = 0.0001

LR = 0.0001
weight_decay=L2REG
opt = torch.optim.Adam
opt = opt(model.parameters(), lr=LR, weight_decay=L2REG,betas=(0.9, 0.999),amsgrad=True)
#opt = opt(model.parameters(), lr=LR, weight_decay=L2REG)

epochs = 200
train_loss, train_accuracy, val_loss, val_accuracy = fit(epochs, model, loss_fn, opt, train_dl, val_dl)

fig, ax = plt.subplots(3, 1, figsize=(12,12))
#print(val_loss)
#print(train_loss)
ax[0].plot(train_loss)
ax[0].plot(val_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss,Params_400_100,wd:{0},LR:{1},BS:{2},epochs:{3}'.format(L2REG,LR,BS,epochs))

ax[1].plot(train_accuracy)
ax[1].plot(val_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')



#plt.tight_layout()
#plt.savefig('graph')
#plt.show(block=False)

#x_tensor_train = torch.from_numpy(np.array(X_train)).float().to(device)
#model.eval()
#yhat_train = model(x_tensor_train)
#y_hat_train = np.where(yhat_train.detach().cpu().numpy()<0.5, 0, 1)
    #print("Test Accuracy {:.2f}".format(test_accuracy))
    
#train_accuracy = accuracy_score(np.array(y_train).reshape(-1,1), y_hat_train)
#cm_train = confusion_matrix(np.array(y_train).reshape(-1,1), y_hat_train)
    
#print(cm_train)
#print(train_accuracy)

from sklearn.metrics import classification_report



with torch.no_grad():
    x_tensor_test = torch.from_numpy(X_val).float().to(device)
    model.eval()
    yhat = model(x_tensor_test)
    y_hat_class = np.where(yhat.cpu().numpy()<0.5, 0, 1)
    #test_accuracy = balanced_accuracy_score(y_val.reshape(-1,1), y_hat_class)
    #print("Test Accuracy {:.2f}".format(test_accuracy))
    
    test_accuracy = accuracy_score(y_val, y_hat_class)
    f1 = f1_score(y_val, y_hat_class)
    cm = confusion_matrix(y_val, y_hat_class)
    recall = recall_score(y_val, y_hat_class)
    classification = classification_report(y_val.reshape(-1,1), y_hat_class,digits=4)
    precision = precision_score(y_val.reshape(-1,1), y_hat_class)
    loss = log_loss(y_val.reshape(-1,1), y_hat_class)
    fpr, tpr, threshold = roc_curve(y_val.reshape(-1,1), y_hat_class)
    auc_score = roc_auc_score(y_val.reshape(-1,1), y_hat_class)
    tn, fp, fn, tp = confusion_matrix(y_val.reshape(-1,1), y_hat_class).ravel()
    specificity = tn / (tn+fp)
    pr = tp / (tp+fp)
    real = tp / (tp+fn)


print('pr',pr)
print('recall',real)
print("Learning Rate:",LR)
print("Weight decay:",weight_decay)
print("Precision:",precision)
print("Test Accuracy {:.2f}".format(test_accuracy))
print("F1 {:.2f}".format(f1))
print("recall:",recall)
print("confusion matrix:",cm)
print("auc_score:",auc_score)
print("specificity",specificity)
print(classification)


ax[2].set_title('Receiver Operating Characteristic')
ax[2].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
ax[2].legend(loc = 'lower right')
ax[2].plot([0, 1], [0, 1],'r--')
ax[2].set_xlim([0, 1])
ax[2].set_ylim([0, 1])
ax[2].set_ylabel('True Positive Rate')
ax[2].set_xlabel('False Positive Rate')
plt.tight_layout()
#plt.show()
plt.show()
#plt.savefig('New_network{0}-{1}-{2}.pdf'.format(epochs,LR,BS))

#### Lasso
from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.1)

reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

# Training data
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
#print('MSE training set', round(mse_train, 2))
print("acu_lasso_train",roc_auc_score(y_train, pred_train))
#from sklearn.metrics import r2_score
#print("r2 train",r2_score(y_train, pred_train))

# Test data
pred = reg.predict(X_val)
mse_test =mean_squared_error(y_val, pred)
print('MSE test set', round(mse_test, 2))
print("acu_lasso_test",roc_auc_score(y_val.reshape(-1,1), pred))
#print("r2 test",r2_score(y_val.reshape(-1,1), pred))
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.tight_layout()

################################# This Part is if you want to generate the weights of the connection #################################

""" from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import csv


ls = list(zip(x.index.tolist()[:133], y_hat_class.tolist()))
df = pd.DataFrame(ls, columns = ['individualID', 'diagnosis'])

outFile = 'Predictions_test2.csv'

print('Saving predictions to file {}'.format(outFile))
df.to_csv(outFile)


print('Interpreting SNP and TF importance...')
feature_names = list(x.columns)

model.adj = model.adj.cpu()
ig = IntegratedGradients(model.cpu())

test_input_tensor = torch.from_numpy(X_val).type(torch.FloatTensor)
attr, delta = ig.attribute(test_input_tensor, return_convergence_delta=True)
attr = attr.detach().numpy()

importances = dict(zip(feature_names, np.mean(abs(attr), axis=0)))

outFile = 'Final_FeatureImportance2_test1.csv'

print('Saving SNP and TF importance to file {}'.format(outFile))
with open(outFile, 'w') as f:
    for key in importances.keys():
        f.write("%s,%s\n"%(key,importances[key]))
        
print('Interpreting gene importance...')
cond = LayerConductance(model, model.GRNeQTL)

cond_vals = cond.attribute(test_input_tensor)
cond_vals = cond_vals.detach().numpy()

importances_layer1 = dict(zip(adj_in.columns.tolist(), np.mean(abs(cond_vals), axis=0)))

outFile = 'final_GeneImportance2_test1.csv'

print('Saving gene importance to file {}'.format(outFile))
with open(outFile, 'w') as f:
    for key in importances_layer1.keys():
        f.write("%s,%s\n"%(key,importances_layer1[key]))
        

neuron_cond = NeuronConductance(model, model.GRNeQTL)
print(neuron_cond)

outFile = 'final_connection_test1.csv'
with open(outFile, 'w') as f:
    print('Interpreting eQTL and GRN connections importance...')
    for gene in adj_in.columns.tolist():

        neuron_cond_vals = neuron_cond.attribute(test_input_tensor, neuron_selector=adj_in.columns.tolist().index(gene))
        #print(gene,neuron_cond_vals)
        importances_neuron = dict(zip(feature_names, abs(neuron_cond_vals.mean(dim=0).detach().numpy())))
        importances_neuron = {key:val for key, val in importances_neuron.items() if val != 0}
        
        for key in importances_neuron.keys():
            f.write("%s,%s,%s\n"%(gene,key,importances_neuron[key]))

print('Succesfully saved eQTL and GRN connections importance to file') """


#######################################################################################################################################