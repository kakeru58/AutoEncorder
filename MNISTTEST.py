# -*- coding: utf-8 -*-
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers
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
input_img = Input(shape=(784, )) #28*28
# 入力された画像がencodeされたものを格納する変数
encoded = Dense(128, activation='relu')(input_img)#32次元でreluという活性化関数にinput_imgを通す
encoded = Dense(64, activation='relu')(encoded)#
encoded = Dense(32, activation='relu')(encoded)#
# ecnodeされたデータを再構成した画像を格納する変数
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
# 入力画像を再構成するModelとして定義
autoencoder = Model(input_img, decoded)#入力と出力

# AdaDeltaで最適化, loss関数はbinary_crossentropy
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# MNISTデータを前処理する
(x_train, _), (x_test, _) = mnist.load_data()#使わないため_にしてる
x_train, x_valid = train_test_split(x_train, test_size=0.175)#訓練ようと試験用に分ける0.175は割合 過学習防止
x_train = x_train.astype('float32')/255.#int型を画素ちの最大255でわる．(正規化)
x_valid = x_valid.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))#3次元データを2次元データにlenは要素数
x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))#np.prodで1番目からの要素の積(784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# autoencoderの実行
plotHistory(
    autoencoder.fit(x_train, x_train,#kerasのfitメゾット
                    epochs=100,
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
