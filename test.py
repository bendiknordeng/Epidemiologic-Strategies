from datetime import datetime, timedelta

start_date= '20200228'
time_delta = 10

dt = datetime.strptime(start_date, '%Y%m%d').date()
dt += timedelta(days=time_delta)

print(dt)
print(type(dt))