import os
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
from tqdm import tqdm
import timm

# ================================================================
# 1. FIXED SETTINGS
# ================================================================
DATA_DIR = r"C:\Users\5520\Documents\IITH BTECH ENGINEERING\SEM 6\CV\Project\monster_project\PACS_dataset"

TARGET_DOMAIN = "sketch"

ALL_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
SOURCE_DOMAINS = [d for d in ALL_DOMAINS if d != TARGET_DOMAIN]

domain_code = "".join([d[0].upper() for d in sorted(SOURCE_DOMAINS)])
SAVE_PATH = f"vit_tiny_pacs_{domain_code}_source_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 0.05
NUM_CLASSES = 7
TRAIN_SPLIT = 0.8
SEED = 42

# ================================================================
# 2. GLOBAL CLASS MAPPING
# ================================================================
CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

ID_TO_CLASS = {
    # Fill only if labels.csv contains numeric IDs instead of class names
}

# ================================================================
# 3. BASE DATASET WITHOUT TRANSFORM
# ================================================================
class FlattenedPACSDomainDataset(Dataset):
    def __init__(self, domain_dir, class_to_idx, id_to_class=None):
        self.domain_dir = domain_dir
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

                if label_name not in self.class_to_idx:
                    raise ValueError(
                        f"Label '{label_name}' in {labels_csv} not found in CLASS_NAMES.\n"
                        f"CLASS_NAMES = {list(self.class_to_idx.keys())}"
                    )

                img_path = os.path.join(domain_dir, new_filename)

                if not os.path.exists(img_path):
                    print(f"Warning: missing file skipped: {img_path}")
                    continue

                label_idx = self.class_to_idx[label_name]
                self.samples.append((img_path, label_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {domain_dir}")

        print(f"Loaded {os.path.basename(domain_dir)}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label

# ================================================================
# 4. WRAPPER TO APPLY TRANSFORM AFTER SPLIT
# ================================================================
class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# ================================================================
# 5. EVALUATION FUNCTION
# ================================================================
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

            running_acc = 100.0 * correct / total
            progress_bar.set_postfix({"acc": f"{running_acc:.2f}%"})

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc

# ================================================================
# 6. MAIN TRAINING
# ================================================================
def main():
    print("--- Starting ViT-Tiny Training ---")
    print(f"Target Domain Held Out For Final Testing: {TARGET_DOMAIN}")
    print(f"Using ACP only for training/validation: {SOURCE_DOMAINS}")
    print(f"Saving best model to: {SAVE_PATH}")
    print(f"Using Device: {DEVICE}\n")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    source_list = []
    for domain in SOURCE_DOMAINS:
        domain_path = os.path.join(DATA_DIR, domain)

        if not os.path.exists(domain_path):
            print(f"CRITICAL ERROR: Folder not found at {domain_path}")
            return

        ds = FlattenedPACSDomainDataset(
            domain_dir=domain_path,
            class_to_idx=CLASS_TO_IDX,
            id_to_class=ID_TO_CLASS
        )
        source_list.append(ds)

    full_acp_dataset = ConcatDataset(source_list)

    total_size = len(full_acp_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(
        full_acp_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Total ACP images: {total_size}")
    print(f"Train split (80%): {len(train_dataset)}")
    print(f"Val split (20%): {len(val_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}\n")

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)

        for images, labels in train_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

            running_loss = total_loss / (train_bar.n + 1)
            running_acc = 100.0 * correct / total

            train_bar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.2f}%"
            })

        scheduler.step()

        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = 100.0 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, DEVICE, criterion)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": IDX_TO_CLASS,
                "target_domain": TARGET_DOMAIN,
                "source_domains": SOURCE_DOMAINS,
                "model_name": "vit_tiny_patch16_224",
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "validation_type": "ACP_80_20_split"
            }

            torch.save(checkpoint, SAVE_PATH)
            print(f"Best model updated and saved at epoch {best_epoch} with ACP val acc {best_val_acc:.2f}%")

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best ACP validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved as: {SAVE_PATH}")

if __name__ == "__main__":
    main()