from region import Region

regions = {}
region_id = 0
N = 681067
initial_infected = 0.05
longitude = 59.91273
latitude = 10.74609
regions[0] = Region(region_id, N, initial_infected, longitude, latitude)

region_id = 1
N = 41460
initial_infected = 0
longitude = 59.924507
latitude = 10.954048
regions[1] = Region(region_id, N, initial_infected, longitude, latitude)

region_id = 2
N = 127731
initial_infected = 0
longitude = 59.89455
latitude = 10.546343
regions[2] = Region(region_id, N, initial_infected, longitude, latitude)

region_id = 3
N = 59288
initial_infected = 0
longitude = 59.735206
latitude = 10.908211
regions[3] = Region(region_id, N, initial_infected, longitude, latitude)

"""
mobility_matrix = [ [0,0.1,0.1,0.1],
                    [0.5,0,0,0],
                    [0.5,0,0,0],
                    [0.5,0,0,0]]
"""

    
