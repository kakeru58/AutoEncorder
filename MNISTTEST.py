# -*- coding: utf-8 -*-
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
def plotHistory(history):
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

# 入力用の変数
input_img = Input(shape=(28, 28, 1)) #28*28

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)#入力と出力
# AdaDeltaで最適化, loss関数はbinary_crossentropy
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# MNISTデータを前処理する
(x_train, _), (x_test, _) = mnist.load_data()#使わないため_にしてる
x_train, x_valid = train_test_split(x_train, test_size=0.175)#訓練ようと試験用に分ける0.175は割合 過学習防止
x_train = x_train.astype('float32')/255.#int型を画素ちの最大255でわる．(正規化)
x_valid = x_valid.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train,(len(x_train),28,28,1))#3次元データを2次元データにlenは要素数
x_valid = np.reshape(x_valid,(len(x_valid),28,28,1))#np.prodで1番目からの要素の積(784)
x_test = np.reshape(x_test,(len(x_test),28,28,1))
# autoencoderの実行
plotHistory(
    autoencoder.fit(x_train, x_train,#kerasのfitメゾット
                    epochs=20,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_valid, x_valid))
    )
# 画像化して確認
decoded_img = autoencoder.predict(x_test)

n = 10#(表示個数)
plt.figure(figsize=(20, 4))#横インチ縦インチの大きさ
for i in range(n):
    #オリジナルのテスト画像
    ax = plt.subplot(2, n, i+1)#何行何列何番目
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)#軸の値非表示
    ax.get_yaxis().set_visible(False)
    #変換された画像
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#https://qiita.com/fukuit/items/2f8bdbd36979fff96b07
