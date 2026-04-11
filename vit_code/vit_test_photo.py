import os
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm

# ================================================================
# 1. FIXED SETTINGS
# ================================================================
DATA_DIR = r"C:\Users\5520\Documents\IITH BTECH ENGINEERING\SEM 6\CV\Project\monster_project\PACS_dataset"

TARGET_DOMAIN = "photo"
MODEL_PATH = "vit_tiny_pacs_ACS_source_best.pth"

TEST_DOMAIN_PATH = os.path.join(DATA_DIR, TARGET_DOMAIN)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

ID_TO_CLASS = {
    # Fill only if labels.csv contains numeric IDs instead of class names
}

# ================================================================
# 2. CUSTOM DATASET
# ================================================================
class FlattenedPACSDomainDataset(Dataset):
    def __init__(self, domain_dir, class_to_idx, transform=None, id_to_class=None):
        self.domain_dir = domain_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.id_to_class = id_to_class if id_to_class is not None else {}
        self.samples = []

        labels_csv = os.path.join(domain_dir, "labels.csv")

        if not os.path.exists(domain_dir):
            raise FileNotFoundError(f"Domain folder not found: {domain_dir}")

        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"labels.csv not found in: {domain_dir}")

        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            required_cols = {"label", "new_filename"}
            if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{labels_csv} must contain columns {required_cols}. Found: {reader.fieldnames}"
                )

            for row in reader:
                raw_label = row["label"].strip()
                new_filename = row["new_filename"].strip()

                label_name = self.id_to_class.get(raw_label, raw_label)

                if label_name not in class_to_idx:
                    raise ValueError(
                        f"Label '{label_name}' in {labels_csv} not found in checkpoint class mapping."
                    )

                img_path = os.path.join(domain_dir, new_filename)

                if not os.path.exists(img_path):
                    print(f"Warning: missing file skipped: {img_path}")
                    continue

                label_idx = class_to_idx[label_name]
                self.samples.append((img_path, label_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {domain_dir}")

        print(f"Loaded {os.path.basename(domain_dir)}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# ================================================================
# 3. TESTING
# ================================================================
def main():
    print("--- Starting ViT-Tiny Testing ---")
    print(f"Testing on domain: {TARGET_DOMAIN}")
    print(f"Using model: {MODEL_PATH}")
    print(f"Using device: {DEVICE}\n")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]
    saved_target_domain = checkpoint["target_domain"]
    saved_source_domains = checkpoint["source_domains"]
    model_name = checkpoint["model_name"]
    best_val_acc = checkpoint.get("best_val_acc", None)
    best_epoch = checkpoint.get("best_epoch", None)
    validation_type = checkpoint.get("validation_type", "unknown")

    print("Checkpoint metadata:")
    print(f"  Held-out target domain for final testing: {saved_target_domain}")
    print(f"  Source domains used for training: {saved_source_domains}")
    print(f"  Validation type during training: {validation_type}")
    print(f"  Saved model name: {model_name}")
    if best_val_acc is not None:
        print(f"  Best ACP validation accuracy: {best_val_acc:.2f}%")
    if best_epoch is not None:
        print(f"  Best epoch: {best_epoch}")
    print(f"  Class names: {class_names}\n")

    if saved_target_domain != TARGET_DOMAIN:
        raise ValueError(
            f"Mismatch: checkpoint target domain is '{saved_target_domain}' "
            f"but current TARGET_DOMAIN is '{TARGET_DOMAIN}'"
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = FlattenedPACSDomainDataset(
        domain_dir=TEST_DOMAIN_PATH,
        class_to_idx=class_to_idx,
        transform=transform,
        id_to_class=ID_TO_CLASS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Total test images in {TARGET_DOMAIN}: {len(test_dataset)}")
    print(f"Number of test batches: {len(test_loader)}\n")

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)

        for images, labels in progress_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            running_acc = 100.0 * correct / total
            progress_bar.set_postfix({"acc": f"{running_acc:.2f}%"})

    final_acc = 100.0 * correct / total

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Target Domain:   {TARGET_DOMAIN}")
    print(f"Model Used:      {MODEL_PATH}")
    print(f"Test Images:     {total}")
    print(f"Test Accuracy:   {final_acc:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()