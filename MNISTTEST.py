from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers
from keras import backend as K
import cv2
# MNISTデータを前処理する
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 使わないため_に
img_size = x_train.shape[1]
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.175)  # 訓練ようと試験用に分ける0.175は割合 過学習防止
x_train = x_train.astype('float32') / 255.  # int型を画素ちの最大255でわる．(正規化)
x_valid = x_valid.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), img_size, img_size, 1))  # 3次元データを2次元データにlenは要素数
x_valid = np.reshape(x_valid, (len(x_valid), img_size, img_size, 1))  # np.prodで1番目からの要素の積(784)
x_test = np.reshape(x_test, (len(x_test), img_size, img_size, 1))

# 入力用の変数
x = input_img = Input(shape=(img_size, img_size, 1))  # 28*28
batch_size = 64
latent_dim = 2
epoch = 50
x = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same')(x)

shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
# 潜在変数
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # default=random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


z = Lambda(sampling)([z_mean, z_log_var])

encoder = Model(input_img, [z_mean, z_log_var, z])
encoder.summary()
# decorder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding='same')(x)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(latent_inputs, decoded)
decoder.summary()
decoded = decoder(encoder(input_img)[2])
autoencoder = Model(input_img, decoded)  # 入力と出力

# loss関数
# Compute VAE loss
reconstruction_loss = binary_crossentropy(K.flatten(input_img),
                                          K.flatten(decoded))
reconstruction_loss *= img_size * img_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
# AdaDeltaで最適化, loss関数はbinary_crossentropy
autoencoder.add_loss(vae_loss)
autoencoder.compile(optimizer='adam', loss='')
autoencoder.summary()
x_train = x_train[y_train == 1]
x_valid = x_valid[y_valid == 1]
# autoencoderの実行
history = autoencoder.fit(x_train,  # kerasのfitメゾット
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(x_valid, None))
# 画像化して確認
decoded_img = autoencoder.predict(x_test)

n = 10  # (表示個数)
plt.figure(figsize=(20, 6))  # 横インチ縦インチの大きさ
for i in range(n):
    # オリジナル
    origin_img = x_test[i].reshape(28, 28)
    ax = plt.subplot(3, n, i + 1)  # 何行何列何番目
    plt.imshow(origin_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)  # 軸の値非表示
    ax.get_yaxis().set_visible(False)
    # 変換された画像
    change_img = decoded_img[i].reshape(28, 28)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(change_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 差分
    diff_img = origin_img - change_img
    diff_img_abs = np.abs(diff_img)
    diff_img = diff_img_abs * 255
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(diff_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# 損失関数のグラフの軸ラベルを設定
plt.xlabel('time step')
plt.ylabel('loss')
# グラフ縦軸の範囲を0以上と定める
plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))
# 損失関数の時間変化を描画
val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
loss, = plt.plot(history.history['loss'], c='#E69F00')
# グラフの凡例（はんれい）を追加
plt.legend([loss, val_loss], ['loss', 'val_loss'])
# 描画したグラフを表示
plt.show()


# https://qiita.com/fukuit/items/2f8bdbd36979fff96b07
