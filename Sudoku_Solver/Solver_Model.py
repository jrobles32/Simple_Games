import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2 as cv


def adding_new_images(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for pic in list_of_files:
        image_file_name = os.path.join(image_directory, pic)
        if '.png' in image_file_name:
            img = cv.imread(image_file_name, flags=cv.IMREAD_GRAYSCALE)
            img = np.array(img)

            img_f = cv.resize(img, (28, 28))

            img_arr = img_f.reshape(1, 28, 28, 1)
            features_data = np.append(features_data, img_arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data


path = r'C:\Users\Javier\AppData\Roaming\JetBrains\PyCharmCE2021.3\scratches\Digits'


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

for folders in os.listdir(path):
    directory = os.path.join(path, folders)
    x_train, y_train = adding_new_images(folders, directory, x_train, y_train)


x_train /= 255
x_test /= 255

print(x_train.shape)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

with tf.device('/gpu:0'):
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0], '\nTest accuracy: ', score[1])
model.save('new_number_reader.model')


# new_model = tf.keras.models.load_model('new_number_reader.model')
# test_img = cv.imread('C:/Users/Javier/AppData/Roaming/JetBrains/PyCharmCE2021.2/scratches/Digits/8/image38.png')
# test_img = np.asarray(test_img)
# test_img = cv.resize(test_img, (28, 28)).astype('float32')
#
# test_img /= 255
# test_img = test_img.reshape(1, 28, 28, 1)
# test_img /= 255
# predictions = new_model.predict([test_img])
# cv.imshow('test', test_img)
#
#
# print(np.argmax(predictions[0]))
# print(test_img.shape)
# cv.waitKey(0)