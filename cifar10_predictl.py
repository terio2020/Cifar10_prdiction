import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

# import scipy
import imageio
# import cv2
from PIL import Image
# from scipy import ndimage
# import scipy.misc

#%%
# Load the Cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

for i in [train_images, train_labels, test_images, test_labels]:
    print(i.shape)
# print(train_labels[0][0])   # train_labels[0] = [6]---->train_labels[0][0] = 6
#%%
# Show the samples
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

plt.figure(figsize=(10, 10))
j = 0
for i in range(25, 50):
    plt.subplot(5, 5, j + 1)
    j += 1
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


#%%
# Load the model
model = tf.keras.models.load_model('Ciafr10_model.h5')

#%%
# Show one image of test_images for predicting
index = 454
plt.imshow(test_images[index])
plt.show()
print("Sample:" + class_names[test_labels[index][0]])

# Predict one picture of test_images
picture = test_images[index]
picture = np.expand_dims(picture, axis=0)
predict = model.predict(picture)
print("Predict:" + class_names[np.argmax(predict)])
#%%
# Predict my image
my_image = "deer.jpg"
fname = './my_image/' + my_image
# image = np.array(imageio.imread(fname))   # It's need (32, 32, 3) image
image = Image.open(fname)                   # This and next code can resize any size
image = image.resize((32, 32),Image.ANTIALIAS)  # into (32, 32, 3)

plt.imshow(image)
plt.show()

image = np.expand_dims(image, axis=0)
image = tf.cast(image, tf.float32)      # Maybe the dtype of loaded image is not correct
predict = model.predict(image)
print("Predict my image:" + class_names[np.argmax(predict)])