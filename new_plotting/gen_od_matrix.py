import pickle
import numpy as np

num_counties = 12
num_time_steps = 84

a = []
for m in range(num_time_steps):
    l = [[0 for x in range(num_counties)] for x in range(num_counties)] 
    for i in range(0, num_counties):
        for j in range(0, num_counties):
            if j == i: 
                l[i][j] = 0.8
            if j == i+1 and j < num_counties -1:
                l[i][j+1] = 0.1
            if j == i-1 and j > 1:
                l[i][j-1] = 0.1
    a.append(l)

arr = np.array(a)

with open('od_counties.pkl','wb') as f:
    pickle.dump(arr, f)
    
with open('od_counties.pkl','rb') as f:
     x = pickle.load(f)
     print(x)
     print(x.shape)