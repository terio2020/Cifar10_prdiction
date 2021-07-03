import tensorflow as tf
import matplotlib.pylab as plt
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
for i in range(25, 50):
    j = 0
    plt.subplot(5, 5, j + 1)
    j += 1
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#%%
# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"))

model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"))

model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"))

model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10))

model.summary()

#%%
# Compile and train the model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

#%%
# Save the model
model.save('Ciafr10_model.h5')

#%%
# Evaluate the model
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.0, 1])
plt.legend()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
