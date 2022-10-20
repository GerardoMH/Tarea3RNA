import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

class Funciones(Sequential): 
	def __init__(self, **kwargs):
		super().__init__(**kwargs) #para heredar desde ODEsolver
		self.loss_tracker = keras.metrics.Mean(name = "loss") #para imprimir valor de costo


	@property
	def metrics(self): #cuál variable tomará para la función de costo?
		return [self.loss_tracker] 
 
	def train_step(self, data): #esta función ya tiene dentro el modelo secuencial, modificaremos una clase que ya esta predefinida xd
		batch_size = tf.shape(data)[0] #tamaño del minibatch xddd, para hacer más fina la resolución 
		x = tf.random.uniform((batch_size, 1), minval=-2, maxval=2) #crea un tensor de numeros aleatorios de una columna
		

		with tf.GradientTape() as tape: #calcula derivadas, tiene la esencia de backpropagation, "graba" todas las operaciones realizadas
		#calcula gradientes de la red neuronal, pesos y biases
			
			y_pred = self(x, training=True)
			eq = y_pred - (3 * tf.math.sin(np.pi * x))

			loss = keras.losses.mean_squared_error(0., eq)
        	

		grads = tape.gradient(loss, self.trainable_variables) #Como cambia la func de costo respecto a las dif variables"
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

		self.loss_tracker.update_state(loss)
		return {"loss": self.loss_tracker.result()}

model = Funciones()
model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])
x=tf.linspace(-1,1,100)
history = model.fit(x, epochs=3000, verbose=1)

x_testv = tf.linspace(-1,1,100)
a=model.predict(x_testv)
plt.plot(x_testv,a)
plt.plot(x_testv,3 * np.sin(np.pi * x))
plt.suptitle('3sin(pi*x)')
leyendas = ['RNAy(x)','y(x)']
plt.legend(loc = "upper right", labels = leyendas)
plt.show()
exit()

#para salvar modelo en el disco
model.save("red.h5")

#para cargar la red
modelo_cargado = tf.keras.models.load_model('red.h5')