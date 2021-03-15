import pickle as pkl
import numpy as np

num_areas = 356
num_time_steps = 84

a = []
for m in range(num_time_steps):
    l = [[0 for x in range(num_areas)] for x in range(num_areas)] 
    for i in range(0, num_areas):
        for j in range(0, num_areas):
            if i == j:
                l[i][j] = 0.8
            elif j == i+1:
                l[i][j] = 0.1
            elif j == i-1:
                l[i][j] = 0.1
    a.append(l)

arr = np.array(a)

filepath = 'data/data_municipalities/od_municipalities.pkl'
with open(filepath ,'wb') as f:
    pkl.dump(arr, f)
    
with open(filepath,'rb') as f:
     x = pkl.load(f)
     print(x[0])
     print(x.shape)