from keras.models import load_model
import cv2
# import numpy as np

TEST_IMAGE = "./data/test/5.jpg"

# 引入模型，显示构成
model = load_model("./model/cats_and_dogs.h5")
model.summary()

#   利用模型进行测试
original_image = cv2.imread(TEST_IMAGE)
resize_image = cv2.resize(original_image, (150, 150))  # 将测试图片缩小
reshape_image = resize_image.reshape(1, 150, 150, 3)  # 把图片转换成模型输入的维度，150*150，RGB三通道
predict = (model.predict(reshape_image) > 0.5).astype("int32")
if predict[0] == 0:
    print("识别为：猫")
else:
    print("识别为：狗")

#   显示图片
cv2.imshow("Test_Image", resize_image)
cv2.waitKey(0)
