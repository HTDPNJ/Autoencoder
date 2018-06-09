import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

(x_train,_),(x_test,y_test)=mnist.load_data()

x_train=x_train.astype('float32')/255.-0.5
x_test=x_test.astype('float32')/255.-0.5
x_train=x_train.reshape((x_train.shape[0],-1))
x_test=x_test.reshape((x_test.shape[0],-1))
print(x_train.shape)
print(x_test.shape)

encoding_dim=2
input_img=Input((784,))
#encode_layers
encoded=Dense(128,activation='relu')(input_img)  #后面括号是输入,这里是压缩层
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoder_output=Dense(encoding_dim,)(encoded)
#decoder layers
decoded=Dense(10,activation='relu')(encoder_output)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded=Dense(784,activation='tanh')(decoded)
#组建autoencoder
autoencoder=Model(input=input_img,outputs=decoded)
#组建单独的encoder
encoder=Model(inputs=input_img,outputs=encoder_output)
autoencoder.compile(optimizer='adam',loss='mse')
#training
autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=256,shuffle=True) #压缩解压同一数据对比结果
#plotting
encoded_imgs=encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test)
plt.show()
