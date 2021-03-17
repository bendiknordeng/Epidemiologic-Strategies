import pickle as pkl
import numpy as np

# In[ ]:

def generate_od_matrix(num_time_steps, num_regions):
    """ generate an OD-matrix used for illustrative purposes only
    Paramters:
        num_regions:
        num_time_steps:
    Returns:
        An OD-matrix with dimensions (num_time_steps, num_regions, num_regions) with 0.8 on its diagonal and 0.1 on cells next to diagonal 
    """
    a = []
    for m in range(num_time_steps):
        l = [[0 for x in range(num_regions)] for x in range(num_regions)] 
        for i in range(0, num_regions):
            for j in range(0, num_regions):
                if i == j:
                    l[i][j] = 0.8
                elif j == i+1:
                    l[i][j] = 0.1
                elif j == i-1:
                    l[i][j] = 0.1
        a.append(l)
    return np.array(a)

def generate_vaccine_matrix(num_time_steps, num_regions):
    """ generate an vaccine matrix used for illustrative purposes only
    Paramters:
        num_counties:
        num_time_steps:
    Returns:
        An vaccine matrix with dimensions (num_time_steps, num_counties) 
    """
    a = [[10 for x in range(num_regions)] for i in range (num_time_steps)]
    return np.array(a)
    
def write_pickle(filepath, arr):
    """ writes arr as a pickle to filepath
    Paramters:
        filepath: string filepath
        arr: array that is written to file 
    """
    with open(filepath,'wb') as f:
        pkl.dump(arr, f)

def read_pickle(filepath):
    """ read pickle from filepath
    Paramters:
        filepath: string filepath
    Returns:
        arr: array that is read from filepath 
    """
    with open(filepath,'rb') as f:
        return pkl.load(f)


def main():
    # f = 'covid/data/data_counties/od_counties.pkl'
    # od = generate_od_matrix(84, 11)
    # write_pickle(f, od)
    # x = read_pickle(f)
    # print(x[0])
    # print(x.shape)

    # f = 'covid/data/data_municipalities/od_municipalities.pkl'
    # od = generate_od_matrix(84, 356)
    # write_pickle(f, od)
    # x = read_pickle(f)
    # print(x[0])
    # print(x.shape)

    v = 'covid/data/data_municipalities/vaccines_municipalities.pkl'
    m = generate_vaccine_matrix(84, 356)
    write_pickle(v, m)
    y = read_pickle(v)
    print(y)
    print(y.shape)

if __name__ == '__main__':
    pass