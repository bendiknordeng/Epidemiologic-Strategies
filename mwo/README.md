# Optimizing Resource Allocation during Epidemic Outbreaks: An Approximate Dynamic Programming Approach for Cholera
This code is a part of the course TIØ4905 at NTNU, constituting the Master's Thesis in Managerial Economics and Operations Research for the Spring of 2020.

## Authors
Martin Willoch Olstad is the primary author of this project code. The class ```VarArraySolutionPrinter``` is written as an example to retrieve solutions at the OR-Tools website and is not written by the primary author. In addition, certain visualizations has 
minor code snippets from StackOverflow, such as the ```map_plot``` function in ```Case.py```. If any problems occur, do not hesitate to contact the primary author.

## Packages required
```
Geopy
Matplotlib
Numba
Numpy
OR-Tools
Pandas
Plotly
PyTorch
Scipy
Sklearn
Tqdm
```

## Usage
A simple interface switching is developed. The particular part of the project to run can be defined by changing the ```computational_study``` variable in ```main.py```. Parameters are changed either in part of ```main.py``` to run, in one of the parameter files in the ```data``` folder, for instance, ```haiti_resource_allocation_parameters.json```, ```haiti_parameters.csv``` and ```epidemic_parameters.json```.

Output is written to a file ```output.txt```, whose name can be changed in the interface. 

Loading and saving models is written in such a way that it should work regardless of operating system and file structure outside the project folder. If it does not, feel free to reach out. The most recently trained model is saved as ```model_v6.pth``` and ```scaler_v6.py```. To save them for future use, copy both and give them the preferred names. 

The project consists of three folders. ```data``` consists of epidemiological and operational data collected for the various cases and parameter files for different scenarios, as well as spreadsheets for estimation and computation. ```figures``` is the folder automatically saved figures are located. ```models``` consists another three folders: ```cholera_model```, which consists of the code defining and running the epidemic model, ```resource_allocation_model```, which makes up the approximate dynamic programming framework, and ```trained_vfa_models``` which is where trained models are saved.

The ```.py``` files consists of:\
```main.py```: Previously explained. Used to run the major components of the project.\
```Instance.py```: Defines functions to initialize the models, run them and compare results.\
```Case.py```: Main component of epidemic model. Contains several ```Region``` objects, and runs the numerical algorithm to simulate epidemics. Important functions include ```simulate_epidemic()``` and ```h()```, the latter of which defines the epidemic model.\
```Region.py```: Contains the simulated epidemiological data for each region, also contains some regional plot methods.\
```MarkovDecisionProcess.py```: Main component of resource allocation model. Contains functions to employ various policies during epidemics. Also contains functions to update the neural network used to represent the value function approximation. In addition, it contains the heuristic employed to solve the problem for each time period. Important functions include ```policy()```, ```update_value_function()``` and ```is_facility_and_personnel_feasible()```.\
```State.py```: Defines the current state of the system, in the perspective of the resource allocation model.\
```ValueFunctionApproximation.py```: Neural network used to represent the value function approximation, built upon the torch.nn.Module\
```parameter_estimation.py```: Support code to estimate data for considered case studies.\

### Example
To train a new model with the Haiti epidemic model, weekly decisions for 120 days, performing 500 epsilon-greedy iterations set the following parameters in ```main.py```:
```
computational_study = 'Model Training'
hypothetical = False
max_iters = 500
horizon = 120.0
decision_period = 7.0
```

To run the base case with the alternative epidemic model with a pre-trained value function approximation, set the following parameters in ```main.py```:
```
computational_study = 'Base Case'
hypothetical = True
load_path = 'vfa_trained_on_alternative_model'
iterations = 100
horizon = 120.0
decision_period = 7.0
```
The ```load_path``` must be a pretrained model with an available scaler. For the alternative model on the base case, this is 'hypothetical_adp_best'.