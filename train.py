import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from model_builder import build_model


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "RESULTDATASET"

# ===================== DATA GENERATORS =====================

# Augmentation فقط للـ TRAIN
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)


train = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "Train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


val = val_gen.flow_from_directory(
    os.path.join(DATA_DIR, "Validation"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ===================== CLASS WEIGHTS =====================
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train.classes),
    y=train.classes
)
weights = dict(enumerate(weights))
print("\n Class Weights:", weights)

# ===================== BUILD MODEL =====================
print("\n Building VGG16 model...")
model = build_model(input_shape=(224, 224, 3), n_classes=3)
model.summary()

# ===================== CALLBACKS =====================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_accuracy')
]

print("\n Starting Training...\n")

history = model.fit(
    train,
    epochs=EPOCHS,
    validation_data=val,
    class_weight=weights,
    callbacks=callbacks
)
