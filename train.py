import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ----------------------------------------------------
# 1. Paths
# ----------------------------------------------------
train_dir = "data2/train"
test_dir = "data2/test"

# ----------------------------------------------------
# 2. Image Data Generators
# ----------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

num_classes = len(train_data.class_indices)
print("Total classes =", num_classes)

# ----------------------------------------------------
# 3. CNN Model
# ----------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------------------------------
# 4. Callbacks
# ----------------------------------------------------
checkpoint = ModelCheckpoint(
    "asl_cnn_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ----------------------------------------------------
# 5. Train Model
# ----------------------------------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,
    callbacks=[checkpoint, early_stop]
)

# ----------------------------------------------------
# 6. Save Final Model
# ----------------------------------------------------
model.save("Model/asl_final_model.h5")
print("Model training completed.")
