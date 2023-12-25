import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os
from matplotlib import pyplot as plt

######### 1.Loading Data #########

DATA_DIR = 'data'
BATCH_SIZE = 20
TIMES = 20
image_exts = ['jpg']
plants = os.listdir(DATA_DIR)
n_classes = len(plants)

data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    batch_size=BATCH_SIZE,
    # default image_size 256x256
    # image_size=(1600, 1200)
)

# data_iterator = data.as_numpy_iterator()

# batch = data_iterator.next()
# print(batch[0])  # images
# print(batch[1])  # labels


data = data.map(lambda images, labels: (images/255, labels))

######### 2.Preprocessing Data #########

# print(data.as_numpy_iterator().next()[0].max())
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# Visualize batch
""" 
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
"""
# plt.show()

######### 3.Deep Neural Network(Machine Learning) #########

train_size = int(len(data)*.7)
validation_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

train = data.take(train_size)
validation = data.skip(train_size).take(validation_size)
test = data.skip(train_size+validation_size).take(test_size)

# 3.1 Building a Neural Network

model = Sequential()

# adding layers

#                                                                         3 rgb channels
model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

# model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(n_classes, activation='sigmoid'))
model.add(Dense(n_classes, activation='softmax'))

# compile the model
model.compile('adam',
              #   loss=tf.losses.BinaryCrossentropy(),
              loss=tf.losses.SparseCategoricalCrossentropy(
                  from_logits=False),
              metrics=['accuracy'])

# to give the summary of the model layers
# model.summary()

# 3.2 Training

# logging
LOG_DIR = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
hist = model.fit(train,
                 epochs=TIMES,
                 validation_data=validation,
                 callbacks=[tensorboard_callback]
                 )

# 3.3 Plot performance

# loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='validation_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'],
         color='orange', label='validation_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

######### 4.Testing #########
""" 
img = cv2.imread('test.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
resize = tf.image.resize(256, 256)
plt.imshow(resize.numpy().astype(int))
plt.show()
"""

######### 5.Save the model #########

MODEL_FILE_PATH = os.path.join('models', 'plants.h5')
if os.path.exists(MODEL_FILE_PATH):
    os.remove(MODEL_FILE_PATH)
model.save(MODEL_FILE_PATH)


# tfjs.converters.save_keras_model(model, '/content/')
# run this in shell

"""
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
"""
