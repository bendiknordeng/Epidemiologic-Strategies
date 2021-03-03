import pyreadr
import pandas as pd

result = pyreadr.read_r("C:\\Users\\peder\\Github\\Epidemiologic-Vaccine-Strategies\\fhi\\data\\asymmetric_mobility_dummy_betas.rda")

result.to_csv(r'Hello.csv')


#print(result.keys())
#df1 = result['asymmetric_mobility_dummy_betas']

#print(df1)