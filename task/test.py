import sklearn.metrics as skm
import numpy as np
import torch.nn as nn
import torch
import pickle

def cal_acc(label, pred):
    logs = nn.LogSoftmax()
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    return precision

# Load the tensor from the pickle file
with open('./CS598_PROJECT/output/label.pickle', 'rb') as f:
    numpy_array = pickle.load(f)

# Convert the NumPy array back to a tensor object
label = torch.from_numpy(numpy_array)

# Now you can use the tensor object as needed
print('label', label)
print(label.shape)


# Load the tensor from the pickle file
with open('./CS598_PROJECT/output/pred.pickle', 'rb') as f:
    numpy_array = pickle.load(f)

# Convert the NumPy array back to a tensor object
pred = torch.from_numpy(numpy_array)

# Now you can use the tensor object as needed
print('pred:', pred)
print(pred.shape)