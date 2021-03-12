import pickle
import numpy as np

num_counties = 11
num_time_steps = 84

a = []
for m in range(num_time_steps):
    l = [[0 for x in range(num_counties)] for x in range(num_counties)] 
    for i in range(0, num_counties):
        for j in range(0, num_counties):
            if i == j:
                l[i][j] = 0.8
            elif j == i+1:
                l[i][j] = 0.1
            elif j == i-1:
                l[i][j] = 0.1
    a.append(l)

arr = np.array(a)

with open('peder\data_counties\od_counties.pkl','wb') as f:
    pickle.dump(arr, f)
    
with open('peder\data_counties\od_counties.pkl','rb') as f:
     x = pickle.load(f)
     print(x[0])
     print(x.shape)