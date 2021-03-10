import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


df_data = pd.read_excel('counties_info.xlsx')

df_merged = pd.read_csv('../CrimeStatebyState_1960-2014.csv')

year = 1960
	
df_sected_crime = df_merged[(df_merged['State']!= 'District of Columbia' ) & (df_merged['Year']== year )]

scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'], 
       [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds

for col in df_sected_crime.columns:
    df_sected_crime[col] = df_sected_crime[col].astype(str)
	
df_sected_crime['text'] = df_sected_crime['State']+'Pop: 'df_sected_crime['Population']'Murder rate: '+df_sected_crime['Murder_per100000']

data = [ dict(
            type='choropleth', # type of map-plot
            colorscale = scl,
            autocolorscale = False,
            locations = df_sected_crime['State_code'], # the column with the state
            z = df_sected_crime['Murder_per100000'].astype(float), # the variable I want to color-code
            locationmode = 'USA-states',
            text = df_sected_crime['text'], # hover text
            marker = dict(     # for the lines separating states
                        line = dict (
                                  color = 'rgb(255,255,255)', 
                                  width = 2) ),               
            colorbar = dict(
                        title = "Murder rate per 100,000 people")
            ) 
       ]

layout = dict(
        title = year,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
#              showlakes = True,  # if you want to give color to the lakes
#             lakecolor = 'rgb(73, 216, 230)'  
            ),
             )

fig = dict( data=data, layout=layout )
	
plotly.offline.iplot(fig)

offline.plot(fig, auto_open=True, image = 'png', image_filename="map_us_crime_"+str(year),
             image_width=2000, image_height=1000, 
             filename='/your_path/'/span>"map_us_crime_"str(year)+'.html', validate=True)

