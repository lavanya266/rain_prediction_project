import os
import requests
from datetime import datetime
import numpy as np

 
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from datetime import datetime
import matplotlib.pyplot as plt
from bing_image_downloader import downloader
from pathlib import Path
import imghdr


image_path = "./data/cat_dogs/"
def download_images(query, limit, output_dir):
     downloader.download(query,
                           limit=limit,
                           output_dir=output_dir,
                           adult_filter_off=True,
                           force_replace=False,
                           timeout=60)
 download_images("cat", 20, image_path)
 download_images("dog", 20, image_path)

 

for category in ["cat","dog"]:
    data_dir = os.path.join(image_path, category)
    image_extensions = [".png", ".jpg"]  # add there all your images file extensions

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

 

# Define hyperparameters and input data

learning_rate = 0.01
num_epochs = 15
batch_size = 32
input_shape = (224, 224, 3)

 

# Define names for tensorboard logging and mlflow

experiment_name = "cat-dog-classifier-mobilenet"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

 

# Load the dataset

train_dataset = keras.preprocessing.image_dataset_from_directory(
    image_path,
    validation_split=0.1,
    subset="training",
    seed=1337,
    image_size=input_shape[:2],
    batch_size=batch_size,
)

 

val_dataset = keras.preprocessing.image_dataset_from_directory(
    image_path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=input_shape[:2],
    batch_size=batch_size,
)

 

# Visualize training images

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):  # Take a single batch
    for i in range(min(len(images), 9)):  # Use min() to avoid out-of-bounds
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

 

# Visualize validation images

plt.figure(figsize=(10, 10))
for images, labels in val_dataset.take(1):  # Take a single batch
    for i in range(min(len(images), 9)):  # Use min() to avoid t-of-bounds
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

 

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images, training=True)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off") 

augmented_train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y))


# Define the base model and add a classifier on top
base_model = MobileNet(input_shape=input_shape, include_top=False, weights="imagenet")
base_model.trainable = False
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

 

keras.utils.plot_model(model, show_shapes=True)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)

 

logdir = os.path.join("logs", experiment_name, run_name)
tb_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=1)

 

# Train the model and log metrics and the model itself to MLflow

history = model.fit(
    augmented_train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    verbose=2,
    callbacks=[tb_callback]
)

 
mlflow.set_tracking_uri("http://172.16.51.127:5002") 
# Set the experiment name and create an MLflow run

mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name = run_name) as mlflow_run:
    mlflow.set_experiment_tag("base_model", "MobileNet")
    mlflow.set_tag("optimizer", "keras.optimizers.Adam")
    mlflow.set_tag("loss", "sparse_categorical_crossentropy")
 
    mlflow.keras.log_model(model, "model")
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("input_shape", input_shape)

    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("train_acc", history.history["accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("val_acc", history.history["val_accuracy"][-1])

    mlflow.log_artifact("model.png", "model_plot")

    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)

 

#%load_ext tensorboard

#%tensorboard --logdir logs/cat-dog-classifier-mobilenet

 

img = keras.preprocessing.image.load_img(
    os.path.join(image_path, "cat/Image_17.jpg"), target_size=input_shape
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
 
predictions = model.predict(img_array)
print("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(predictions[0][0]),
                                                          100 * float(predictions[0][1])))

plt.imshow(img_array[0].numpy().astype("uint8"))


# Logged model in MLFlow
logged_model_path = f"runs:/{mlflow_run_id}/model"
 
# Load model as a Keras model
loaded_model = mlflow.keras.load_model(logged_model_path)
predictions = loaded_model.predict(img_array)
print("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(predictions[0][0]),
                                                          100 * float(predictions[0][1])))

 

plt.imshow(img_array[0].numpy().astype("uint8"))
model_name = "cat_dog_classifier"
model_version = 1
print("MLFlow Run ID: ", mlflow_run_id)


with mlflow.start_run(run_id=mlflow_run_id) as run:
    result = mlflow.register_model(
        logged_model_path,
        model_name
    )

# Load model as a Keras model
loaded_model = mlflow.keras.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)


predictions = loaded_model.predict(img_array)
print("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(predictions[0][0]), 100 * float(predictions[0][1])))

 
plt.imshow(img_array[0].numpy().astype("uint8"))
#client = mlflow.tracking.MlflowClient()
client = MlflowClient()
# client.transition_model_version_stage(
#     name=model_name,
#     version=model_version,
#     stage="Production"
# )


client.transition_model_version_stage(
    name="cat_dog_classifier",  # Model name
    version=1,                  # Model version
    stage="Staging"             # Desired stage: "Staging" or "Production"
)

 

# Transition model to Production

client.transition_model_version_stage(
    name="cat_dog_classifier",
    version=1,
    stage="Production"
)

 

# Load model as a Keras model
loaded_model = mlflow.keras.load_model(
    model_uri=f"models:/{model_name}/production"
)

predictions = loaded_model.predict(img_array)
print("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(predictions[0][0]),100 * float(predictions[0][1])))

plt.imshow(img_array[0].numpy().astype("uint8"))
