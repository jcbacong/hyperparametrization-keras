from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd
import keras
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler

## For reproducibility
from numpy.random import seed
seed(10241996)

def data():

	"""
	Prepares data. Separated so that hyperopt wont reload data for each eval run.
	"""
	from sklearn.model_selection import train_test_split
	csvfile = "energydata.csv"
	split=0.30
	df = pd.read_csv(csvfile)
	labels = ["Appliances", "Lights"]
	features = [i for i in df.columns if i not in labels]

	y_data = df[labels]
	x_data = df[features]
	x_scaler = MinMaxScaler()
	y_scaler = MinMaxScaler()

	x_scldata = pd.DataFrame(x_scaler.fit_transform(x_data), columns=x_data.columns)
	y_scldata = pd.DataFrame(y_scaler.fit_transform(y_data), columns=y_data.columns)

	_x_train, _x_test, _y_train, _y_test = train_test_split(x_scldata, y_scldata, test_size=split)


	x_train = _x_train.values
	y_train = _y_train.values
	x_test = _x_test.values
	y_test = _y_test.values

	return x_train, y_train, x_test, y_test


def create_model(x_train,y_train,x_test,y_test):
	"""
	Keras model function.
	"""

	inshape = len(x_train[0])
	outshape = len(y_train[0])
	min_hlayers=3

	model = Sequential()
	for i in range(min_hlayers):
		if i==0:
			model.add(Dense({{ choice(range(50)) }},input_shape=(inshape,)))
			model.add(Activation({{ choice(['relu','sigmoid']) }})) ## Choose between relu or signmoid activation
			model.add(Dropout({{ uniform(0,1) }})) ## Choose dropout value using uniform distribution of values from 0 to 1
		else:
			model.add(Dense({{ choice(range(50)) }}))
			model.add(Activation({{ choice(['relu','sigmoid']) }}))
			model.add(Dropout({{ uniform(0,1) }}))

	model.add(Dense(outshape))
	model.add(Activation({{ choice(['relu','sigmoid']) }}))

	## Hyperparameterization of optimizers and learning rate
	_adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
	_rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
	_sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})

	opt_choiceval = {{ choice( ['_adam', '_rmsprop', '_sgd'] ) }}

	if opt_choiceval == '_adam':
		optim = _adam
	elif opt_choiceval == '_rmsprop':
		optim = _rmsprop
	else:
		optim = _sgd
	
	model.compile(loss='mean_absolute_error', metrics=['mse'],optimizer=optim)
	model.fit(x_train, y_train,
		batch_size=100,
		epochs=5,
		verbose=2,
		validation_data=(x_test, y_test))

	score, acc = model.evaluate(x_test, y_test)
	predicted = model.predict(x_test)

	## Print validation set
	# for i in range(5):
	# 	print("Pred: ",predicted[i], " Test: ",y_test[i])
	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def main():
		
	trials =Trials()
	best_run, best_model = optim.minimize(model=create_model,
	                                      data=data,
	                                      algo=tpe.suggest,
	                                      max_evals=3,
	                                      trials=trials)
	x_train, y_train, x_test, y_test = data()
	print("\n >> Hyperparameters  ")
	for t in best_run.items():
		print("[**] ",t[0],": ", t[1])

	print("\nSaving model...")
	model_json = best_model.to_json()
	with open("model_num.json","w") as json_file:
		json_file.write(model_json)
	best_model.save_weights("model_num.h5")

	



if __name__ == "__main__":
	main()
	print("done..")
