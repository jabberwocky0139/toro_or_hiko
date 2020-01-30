import os

# データセットディレクトリ作成
saved_model_dir = './data'
base_dir = '.'
os.mkdir(base_dir)

# 訓練・検証・テストディレクトリ作成
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 訓練ディレクトリ @ トロ用・ひこにゃん用
train_cats_dir = os.path.join(train_dir, 'toro')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'hiko')
os.mkdir(train_dogs_dir)

# 検証用ディレクトリ @ トロ用・ひこにゃん用
validation_cats_dir = os.path.join(validation_dir, 'toro')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'hiko')
os.mkdir(validation_dogs_dir)

# テスト用ディレクトリ @ トロ用・ひこにゃん用
test_cats_dir = os.path.join(test_dir, 'toro')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'hiko')
os.mkdir(test_dogs_dir)


# データ数が少ないので交差検証したほうがよいかもしれない

from sklearn.model_selection import train_test_split
import os, shutil

toro_path = os.listdir(saved_model_dir + '/井上トロ')
hiko_path = os.listdir(saved_model_dir + '/ひこにゃん')

train_toro, test_toro, _, _  = train_test_split(toro_path, [0]*len(toro_path), test_size=0.4)
test_toro, validation_toro, _, _ = train_test_split(test_toro, [0]*len(test_toro), test_size=0.5)
train_hiko, test_hiko, _, _  = train_test_split(hiko_path, [0]*len(hiko_path), test_size=0.4)
test_hiko, validation_hiko, _, _ = train_test_split(test_hiko, [0]*len(test_hiko), test_size=0.5)

base_dir = './toro_or_hiko'

# トロ画像をtrain/toroにコピー
for fname in train_toro:
    src = os.path.join(saved_model_dir + '/井上トロ', fname)
    dst = os.path.join(base_dir + '/train/toro', fname)
    shutil.copyfile(src, dst)

# トロ画像をtest/toroにコピー
for fname in test_toro:
    src = os.path.join(saved_model_dir + '/井上トロ', fname)
    dst = os.path.join(base_dir + '/test/toro', fname)
    shutil.copyfile(src, dst)

# トロ画像をvalidation/toroにコピー
for fname in test_toro:
    src = os.path.join(saved_model_dir + '/井上トロ', fname)
    dst = os.path.join(base_dir + '/validation/toro', fname)
    shutil.copyfile(src, dst)

# ひこにゃん画像をtrain/hikoにコピー
for fname in train_hiko:
    src = os.path.join(saved_model_dir + '/ひこにゃん', fname)
    dst = os.path.join(base_dir + '/train/hiko', fname)
    shutil.copyfile(src, dst)

# ひこにゃん画像をtest/hikoにコピー
for fname in test_hiko:
    src = os.path.join(saved_model_dir + '/ひこにゃん', fname)
    dst = os.path.join(base_dir + '/test/hiko', fname)
    shutil.copyfile(src, dst)

# ひこにゃん画像をvalidation/hikoにコピー
for fname in test_hiko:
    src = os.path.join(saved_model_dir + '/ひこにゃん', fname)
    dst = os.path.join(base_dir + '/validation/hiko', fname)
    shutil.copyfile(src, dst)



from keras import layers, models, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# モデルの構築
model = models.Sequential()

# Conv-Pool 4段 + Dropout + FC 2段
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])

# ディレクトリ内の画像をすべて150x150に変換
# binary_crossentropyを用いるために2値ラベルが必要なので、class_mode='binary'
# ディレクトリごとにラベルが決定される
# |- toro_or_hiko
#    |- train ← なうここ. catsとdogsでラベリングされる
#       |- cats
#       |- dogs
#    |- validation
#       |- cats
#       |- dogs
#    |- test
#       |- cats
#       |- dogs
# 20個ごとにバッチにまとめて出力
# RGB画像(20, 150, 150, 3)と2値ラベル(20,)を生成する
train_dir = './toro_or_hiko/train'
validation_dir = './toro_or_hiko/validation'


# -20〜20度の範囲でランダムに回転
# 水平・垂直に20%ランダムに平行移動
# 20%の割合でランダムに等積変形(平行四辺形っぽく)
# 20%の割合でランダムにズーム
# x軸反転を許可
train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')



# ジェネレータからモデルの学習に入る
history = model.fit_generator(train_generator,
                              steps_per_epoch=20,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=30)

model.save('toro_or_hiko_2.h5')
saved_model_dir = './gdrive/My Drive/Colab Notebooks/saved_model'


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
