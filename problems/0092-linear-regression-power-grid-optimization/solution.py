import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.

	daily_fluc = [10*math.sin(2*PI*(i+1) / 10) for i in range(10)]
	data_without_fluc = [consumption_data[i]-daily_fluc[i] for i in range(10)]
	days = [(i+1) for i in range(10)]

	b = 0
	day_mean = 0 
	for i in range(10):
		b += data_without_fluc[i]
		day_mean += (i+1)
	b = b / 10
	day_mean = day_mean / 10

	# debias 
	for i in range(10):
		data_without_fluc[i] -= b
		days[i] -= day_mean

	# weights
	w_numerater = 0
	w_denorminator = 0
	for i in range(10):
		w_numerater += data_without_fluc[i] * days[i]
		w_denorminator += days[i] * days[i]
	w = w_numerater/w_denorminator

	predict_without_fluc = w * (15-day_mean) + b
	fluc = 10*math.sin(2*PI*15 / 10)
	pred = (predict_without_fluc + fluc) * 1.05
	pred = math.ceil(pred)

	return pred