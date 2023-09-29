import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

###########################MAGIC HAPPENS HERE##########################
# Change the hyper-parameters to get the model performs well
config = {
    'batch_size': 64,
    'image_size': (224,224),
    'epochs': 20,
    'optimizer': 'adam'
}
###########################MAGIC ENDS  HERE##########################

def read_data():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "./images/flower_photos",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        labels='inferred',
        label_mode = 'int'
    )
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    return train_ds, val_ds, test_ds

def data_processing(ds):
    ds = ds.shuffle(1000)
    data_augmentation = keras.Sequential(
        [
            ###########################MAGIC HAPPENS HERE##########################
            # Use dataset augmentation methods to prevent overfitting,
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.25),
            layers.RandomZoom(.2, .2)
            ###########################MAGIC ENDS HERE##########################
        ]
    )
    ds = ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    ###########################MAGIC HAPPENS HERE##########################
    # Build up a neural network to achieve better performance.
    # Use Keras API like `x = layers.XXX()(x)`
    # Hint: Use a Deeper network (i.e., more hidden layers, different type of layers)
    # and different combination of activation function to achieve better result.
    hidden_units = 224
    x = layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(hidden_units, activation='relu')(x)

    ###########################MAGIC ENDS HERE##########################
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer='he_normal')(x)
    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model


# Display the misclassified images
def plot_misclassified_images(images, labels, predictions, num_images=3):
    figsize = (20, 20)
    fig = plt.figure(figsize=figsize)
    
    for i in range(num_images):
        idx = misclassified_indices[i]
        image = images[idx]
        true_label = labels[idx]
        predicted_label = predictions[idx]
        
        # Display the misclassified image along with true and predicted labels
        plt.subplot(1, num_images, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.astype(np.uint8))
        plt.xlabel(f'True: {true_label}, Predicted: {predicted_label}')
    
    plt.show()

if __name__ == '__main__':
    # Load and Process the dataset
    train_ds, val_ds, test_ds = read_data()
    train_ds = data_processing(train_ds)
    # Build up the ANN model
    model = build_model(config['image_size']+(3,), 5)
    # Compile the model with optimizer and loss function
    model.compile(
        optimizer=config['optimizer'],
        loss='SparseCategoricalCrossentropy',
        metrics=["accuracy"],
    )
    # Fit the model with training dataset
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds
    )
    ###########################MAGIC HAPPENS HERE##########################
    print(history.history)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\nTest Accuracy: ", test_acc)
    test_images = np.concatenate([x for x, y in test_ds], axis=0)
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)
    test_prediction = np.argmax(model.predict(test_images),1)

    model.save('Model_12.h5')

    # 1. Visualize the confusion matrix by matplotlib and sklearn based on test_prediction and test_labels
    ConfusionMatrixDisplay.from_predictions(test_labels, test_prediction)
    plt.show()
    
    # 2. Report the precision and recall for 10 different classes
    # Hint: check the precision and recall functions from sklearn package or you can implement these function by yourselves.
    print(classification_report(test_labels, test_prediction))

    # 3. Visualize three misclassified images
    # Hint: Use the test_images array to generate the misclassified images using matplotlib
    misclassified_indices = np.where(test_prediction != test_labels)[0]

    # Shuffle the misclassified indices to get random misclassified images
    np.random.shuffle(misclassified_indices)

    # Display three random misclassified images
    plot_misclassified_images(test_images, test_labels, test_prediction, num_images=3)



    ###########################MAGIC HAPPENS HERE##########################
    