import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math




class Calculate_accuracy(nn.Module):
    def __init__(self, error):
        super(Calculate_accuracy, self).__init__()
        self.error = error
        # units: meters

    def forward(self, preds, trues):
        s = preds[:,:,0] /180 * np.pi
        lon1, lat1, lon2, lat2 = map(lambda x: x / 180 * np.pi, [preds[:,:,0], preds[:,:,1], trues[:,:,0], trues[:,:,1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(a ** 0.5)
        r = 6371000  # meter
        dis = c * r

        right = np.sum(dis<self.error)
        all = np.size(dis)
        acc = right/all
        return acc


# args
map_root = '/home/user/Documents/Yangkaisen/OE-GMLTP/GCN_Informer_test/map/map_Aarea.png'
Preds = np.load('./output_f4_16/Trans/Preds.npy')
Reals = np.load('./output_f4_16/Trans/Reals.npy')
#Preds = np.load('/home/user/Documents/Yangkaisen/Shanghai/Shanghai/output_f2_16/LSTM/Preds.npy')
#Reals = np.load('/home/user/Documents/Yangkaisen/Shanghai/Shanghai/output_f2_16/LSTM/Reals.npy')


st = 0  # how many ship to visual
ed = 20
K = 20
observed = 16
"""
for i in range(50):
    plt.scatter(Reals[:, i, 0], Reals[:, i, 1], c='b', s=3)
    imp = plt.imread(map_root)
    plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
    plt.show()
    print(';')
"""

# show
mean_true = Reals.mean(axis=(0, 1), keepdims = True)
std_true = Reals.std(axis=(0, 1), keepdims = True)
mean_true = mean_true[:, :, :2]
std_true = std_true[:, :, :2]

Preds = Preds[:,:,:2] * std_true + mean_true
Reals = Reals[observed:,:, :2]

cal_acc = Calculate_accuracy(error=120)
acc = cal_acc(Preds, Reals)

# hongkong, 16-16pred &ade0.052: err&acc: [20m-0.097, 50m-0.40, 100m-0.72, 120m-0.80]
#           16-16pred &ade0.036: err&acc: [20m-0.24, 50m-0.64, 100m-0.86, 120m-0.90]
# shanghai  16-16pred &ade0.023: err&acc: [20m-0.40, 50m-0.78, 100m-0.93, 120m-0.95]
#           16-16pred &ade0.018: err&acc: [20m-0.65, 50m-0.91, 100m-0.98, 120m-0.99]
i = [n for n in range(20)]

#for i in range(16):
plt.scatter(Reals[:, i, 0], Reals[:, i, 1],c='b',s=3)
plt.scatter(Preds[:, i, 0], Preds[:, i, 1],c='r', s=3)

imp = plt.imread(map_root)
plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
plt.show()
print(';')



