import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

class ODEsolver(Sequential):
	def __init__(self, **kwargs):
		super().__init__(**kwargs) #para heredar desde ODEsolver
		self.loss_tracker = keras.metrics.Mean(name = "loss") #para imprimir valor de costo


	@property
	def metrics(self): #cuál variable tomará para la función de costo?
		return [self.loss_tracker] 
 
	def train_step(self, data): #esta función ya tiene dentro el modelo secuencial, modificaremos una clase que ya esta predefinida xd
		batch_size = tf.shape(data)[0] #tamaño del minibatch xddd, para hacer más fina la resolución 
		x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5) #crea un tensor de numeros aleatorios de una columna
		

		with tf.GradientTape() as tape: #calcula derivadas, tiene la esencia de backpropagation, "graba" todas las operaciones realizadas
		#calcula gradientes de la red neuronal, pesos y biases
			with tf.GradientTape() as tape2:
				tape2.watch(x)
				y_pred = self(x, training=True)
				dy = tape2.gradient(y_pred, x)
				x_o = tf.zeros((batch_size, 1))
				y_o = self(x_o, training = True)
			eq = x*dy + y_pred - x**2*tf.math.cos(x)  #estamos resolviendo esta ec dy/dt=-2xy
			ic = y_o - 0
			loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0.,ic)


			grads = tape.gradient(loss, self.trainable_variables) #Como cambia la func de costo respecto a las dif variables"
			self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

			self.loss_tracker.update_state(loss)
			return {"loss": self.loss_tracker.result()}

model = ODEsolver()
model.add(Dense(100, activation='tanh', input_shape=(1,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=Adam(),metrics=['loss'])
x=tf.linspace(-5,5,1000)
history = model.fit(x, epochs=3000, verbose=1)

x_testv = tf.linspace(-5,5,1000)
a=model.predict(x_testv)
plt.plot(x_testv,a)
plt.plot(x_testv, x*np.sin(x)+2*np.cos(x)-2*np.sin(x)/x)
plt.suptitle('Solución primera Ec Diferencial')
leyendas = ['RNA[y(x)]','y(x)']
plt.legend(loc = "upper right", labels = leyendas)
plt.show()
exit()

#para salvar modelo en el disco
model.save("red.h5")

#para cargar la red
modelo_cargado = tf.keras.models.load_model('red.h5')