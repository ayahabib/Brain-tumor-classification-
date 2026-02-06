import os
import shutil
from sklearn.model_selection import train_test_split

SEED = 111
CLASS_TYPES = ['BENIGN', 'MALIGNANT', 'NORMAL']

def split_dataset(original_dir, output_dir, test_size=0.15, val_size=0.15):
    print("Splitting dataset (NO augmentation)...")

    for folder in ["Train", "Validation", "Test"]:
        for cls in CLASS_TYPES:
            os.makedirs(os.path.join(output_dir, folder, cls), exist_ok=True)

    for cls in CLASS_TYPES:
        cls_path = os.path.join(original_dir, cls)
        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.png'))]

        images = [os.path.join(cls_path, img) for img in images]

        # Test split
        train_val, test = train_test_split(images, test_size=test_size, random_state=SEED)

        # Validation split
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=SEED)

        for file in train:
            shutil.copy(file, os.path.join(output_dir, "Train", cls))

        for file in val:
            shutil.copy(file, os.path.join(output_dir, "Validation", cls))

        for file in test:
            shutil.copy(file, os.path.join(output_dir, "Test", cls))

        print(f"✔ {cls}: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    print("Splitting done successfully!")

split_dataset(
    original_dir=r"Modified_Dataset",
    output_dir=r"RESULTDATASET",
    test_size=0.15,
    val_size=0.15
)
