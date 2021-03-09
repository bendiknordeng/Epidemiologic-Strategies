#import the basic libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
from seir import seir, seir_plot, seir_plot_one_cell
from collections import namedtuple


# load OD matrices
pkl_file = open('peder/Data/Yerevan_OD_matrices.pkl', 'rb') # change to your desired directory
OD_matrices = pickle.load(pkl_file)
pkl_file.close()

# load population densities
pkl_file = open('peder\Data\Yerevan_population.pkl', 'rb')
pop = pickle.load(pkl_file)
pkl_file.close()

# set up model
r = OD_matrices.shape[0]
n = pop.shape[1]
N = 1000000.0
initialInd = [334, 353, 196, 445, 162, 297] # Muncipalities that is initially infected
initial = np.zeros(n)
initial[initialInd] = 50  
Param = namedtuple('Param', 'R0 DE DI I0 HospitalisationRate HospitalIters')
model = Param(R0=2.4, DE= 5.6 * 12, DI= 5.2 * 12, I0=initial, HospitalisationRate=0.1, HospitalIters=15*12)
alpha = np.ones(OD_matrices.shape)
iterations = 3000 # 250 days
inf = 50

# Run model
res = {}  
res['case1'] = seir(model, pop, OD_matrices, alpha, iterations, inf)

# print for one cell
print(res['case1'][1][::12, :, :].shape)
print(res['case1'][1][::12, :, :])

# Visualize for single cell
print("Max number of hospitalised people in cell #: ", int(res["case1"][0][1,4].max()))
print("Day with max hospitalised people in cell #: ", int(res["case1"][0][1,4].argmax()/12))
seir_plot_one_cell(res["case1"][1], 191) 

# Visualize for country 
print("Max number of hospitalised people: ", int(res["case1"][0][:,4].max()))
print("Day with max hospitalised people: ", int(res["case1"][0][:,4].argmax()/12))
seir_plot(res["case1"][0])