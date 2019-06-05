# coding: utf-8
from keras.layers import Conv2D, Input
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
import math
import numpy as np
import os
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image

# 設定---------------------------------
BATCH_SIZE = 32
NUM_EPOCH = 20
GENERATED_IMAGE_PATH = '/Users/aa366864/Desktop/gan_demo/result' # 生成画像の保存先
# ---------------------------------

## generaterモデルの定義
def generator_model():
    data_format = 'channels_last'
    model = Sequential() # 入力層~隠れ層~出力層と順番につながれたモデル（シーケンシャルモデル）
    model.add(Dense(input_shape=(100,),units=1024, kernel_initializer='he_normal')) # 入力層の定義。層内の各ニューロンを次の層の各ニューロンに全結合させる。unitはMNISTデータのピクセル数28*28を表す。重みの値の初期化も行う。(ReLU関数を用いるときはHeの初期値を用いると良いらしい)
    model.add(BatchNormalization()) # 入力値を平均0分散1に正規化。精度向上、および収束を早める効果あり。
    model.add(Activation('relu')) # 活性化関数をreluとする。
    model.add(Dense(128*7*7)) # 次層の定義
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,))) # データの形を(縦,横,チャンネル)のテンソルにreshape
    model.add(UpSampling2D((2, 2), data_format=data_format)) # アップサンプリングを行い、本物データ(mnist)と同じ28*28の画像にする
    model.add(Conv2D(64, (5, 5), padding='same', data_format=data_format, kernel_initializer='he_normal')) #畳み込み層の定義。Conv2Dによる畳み込み計算。padding='same'で入力画像と出力画像のサイズをあわせる。
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2), data_format=data_format))
    model.add(Conv2D(1, (5, 5), padding='same', data_format=data_format, kernel_initializer='he_normal'))
    model.add(Activation('tanh'))
    return model


# discriminatorモデルの定義
def discriminator_model():
    data_format = 'channels_last'
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', data_format=data_format,
                            input_shape=(28, 28, 1), kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5), strides=(2, 2), data_format=data_format, kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# 画像の統合
def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[ :, :, 0]
    return combined_image

# 実行部
def train():
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator （discriminator部分の重みは固定）
    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            # 生成画像を出力
            if index % 200 == 0:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8))\
                    .save(GENERATED_IMAGE_PATH+"%04d_%04d.png" % (epoch, index))

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))
        generator.save_weights('generator.h5')
        discriminator.save_weights('discriminator.h5')
        
# 実行
train()