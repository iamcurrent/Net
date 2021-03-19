import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)) / 255.0
x_test = x_test.reshape((-1, 28*28)) / 255.0


code_dim = 32
inputs = layers.Input(shape=(x_train.shape[1],), name='inputs')
code = layers.Dense(code_dim, activation='relu', name='code')(inputs)
outputs = layers.Dense(x_train.shape[1], activation='softmax', name='outputs')(code)
auto_encoder = keras.Model(inputs, outputs) ##自编码器
encoder = keras.Model(inputs,code) ##解码器
decoder_input = keras.Input((code_dim,))##解码器
decoder_output = auto_encoder.layers[-1](decoder_input)
decoder = keras.Model(decoder_input, decoder_output)

auto_encoder.compile(optimizer='adam',
                    loss='binary_crossentropy')
history = auto_encoder.fit(x_train, x_train, batch_size=64, epochs=100, validation_split=0.1)
encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)