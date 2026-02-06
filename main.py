import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "RESULTDATASET"
MODEL_PATH = "best_model.keras"


model = load_model(MODEL_PATH)
print(" Model loaded successfully!")

# ===================== TEST DATA GENERATOR =====================
test_gen = ImageDataGenerator(rescale=1./255)
test = test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "Test"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test)
print(f"\n📌 Test Accuracy: {test_acc:.4f}")
print(f"📌 Test Loss: {test_loss:.4f}")


preds = model.predict(test)
pred_classes = np.argmax(preds, axis=1)
true_classes = test.classes
class_labels = list(test.class_indices.keys())

# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ===================== CLASSIFICATION REPORT =====================
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print("\n Classification Report:\n")
print(report)
errors = np.where(pred_classes != true_classes)[0]

for i in errors[:5]:
    img, _ = test[i]
    plt.imshow(img[0])
    plt.title(f"True: {class_labels[true_classes[i]]} | Pred: {class_labels[pred_classes[i]]}")
    plt.axis('off')
    plt.show()
