#   导包
import os
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

BASE_DIR = "./data"
EPOCHS = 50

#   将VGG16卷积基实例化
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

#   建立模型
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

#   利用冻结的卷积基端到端地训练模型
train_dir = os.path.join(BASE_DIR, "train")
validation_dir = os.path.join(BASE_DIR, "validation")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#   配置模型并拟合
model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5), loss="binary_crossentropy", metrics=["acc"]
)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=50,
)
model.save("./model/cats_and_dogs.h5")

#   评估模型
acc = history.history["acc"]
loss = history.history["loss"]
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, "bo", label="accuracy")
plt.plot(epochs, loss, "b", label="loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()