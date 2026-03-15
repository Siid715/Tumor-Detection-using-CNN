import os, json, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# ===================== الإعدادات =====================
@dataclass
class Config:
    data_dir: str = r"BR35Hdataset"   # <-- عدّل هذا للمجلد الذي يحوي yes/ و no/
    out_dir: str  = "./artifacts_cnn_baseline_low"

    img_size: int = 128               # تصغير الإدخال يقلّل قدرة النموذج
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42

    # تقسيم
    test_size: float = 0.20
    val_size: float  = 0.10

    # تدريب (قصير ومتحفّظ لعدم الوصول لأداء عالٍ جداً)
    epochs: int = 20                   # عدد عصور قليل
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7                 # إيقاف مبكر
    dropout: float = 0.3              # تزيد من صرامة التعلّم

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

# ===================== الداتا =====================
def list_images_by_class(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    classes = ["no", "yes"]
    paths, labels = [], []
    for idx, cname in enumerate(classes):
        cdir = os.path.join(root_dir, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        for fname in os.listdir(cdir):
            fp = os.path.join(cdir, fname)
            if os.path.isfile(fp) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(fp); labels.append(idx)
    return paths, labels, classes

class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]; y = self.labels[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.transform(im)    # فقط Resize + ToTensor (بدون Normalize/Aug)
        return x, y

def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),       # بدون Normalize
    ])

# ===================== نموذج CNN صغير =====================
class SmallCNN(nn.Module):
    """
    شبكة صغيرة عمدًا (3 بلوكات فقط) لتكون baseline أقل من الهجين.
    المدخل: 3x128x128 → مخرجات ثنائية.
    """
    def __init__(self, num_classes=2, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                nn.Dropout(p_drop),

            # block 2
            nn.Conv2d(32,64,3,padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),   nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                nn.Dropout(p_drop),

            # block 3
            nn.Conv2d(64,128,3,padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                nn.Dropout(p_drop),
        )
        # 128x128 → بعد 3 MaxPool(2): 16x16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===================== تدريب/تقييم =====================
def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss, y_true, y_pred = 0.0, [], []
    for xb, yb in tqdm(loader, desc="Train", ncols=100):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        running_loss += loss.item()*xb.size(0)
        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits,1).detach().cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred)
    return running_loss/len(loader.dataset), acc, f1

@torch.inference_mode()
def eval_model(model, loader, device, criterion):
    model.eval()
    loss_sum, y_true, y_pred, y_prob = 0.0, [], [], []
    for xb, yb in tqdm(loader, desc="Eval", ncols=100):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb); loss = criterion(logits, yb)
        loss_sum += loss.item()*xb.size(0)
        probs = torch.softmax(logits, dim=1)[:,1]
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits,1).cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_prob)
    except Exception: auc = float("nan")
    return loss_sum/len(loader.dataset), acc, f1, auc, np.array(y_true), np.array(y_pred)

def main():
    print("Config:", cfg)
    device = get_device(); print("Device:", device)

    all_paths, all_labels, classes = list_images_by_class(cfg.data_dir)
    print(f"Found {len(all_paths)} images. Classes={classes}")

    # تقسيم
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_paths, all_labels, test_size=cfg.test_size, random_state=cfg.seed, stratify=all_labels
    )
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, random_state=cfg.seed, stratify=y_temp
    )
    print(f"Split -> train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

    tfm = build_transforms(cfg.img_size)
    pin_mem = (device.type=="cuda")
    dl_train = DataLoader(ImageListDataset(X_train,y_train,tfm), batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=pin_mem)
    dl_val   = DataLoader(ImageListDataset(X_val,  y_val,  tfm), batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=pin_mem)
    dl_test  = DataLoader(ImageListDataset(X_test, y_test, tfm), batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=pin_mem)

    # نموذج صغير من الصفر
    model = SmallCNN(num_classes=2, p_drop=cfg.dropout).to(device)
    # ملاحظة: لا class weights ولا Normalize — baseline بسيط ومنخفض
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val, patience_ctr = -1.0, 0
    best_path = os.path.join(cfg.out_dir, "smallcnn_best.pt")

    for epoch in range(1, cfg.epochs+1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, dl_train, optimizer, device, criterion)
        va_loss, va_acc, va_f1, va_auc, _, _ = eval_model(model, dl_val, device, criterion)
        scheduler.step()
        print(f"Train | loss {tr_loss:.4f}  acc {tr_acc:.4f}  f1 {tr_f1:.4f}")
        print(f"Val   | loss {va_loss:.4f}  acc {va_acc:.4f}  f1 {va_f1:.4f}  auc {va_auc:.4f}")

        score = (va_acc + va_f1)/2.0
        if score > best_val:
            best_val = score; patience_ctr = 0
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(">> Saved best model.")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(">> Early stopping.")
                break

    # اختبار بالأفضل
    print("\n=== TEST ===")
    ckpt = torch.load(best_path, map_location=device); model.load_state_dict(ckpt["model"])
    _, te_acc, te_f1, te_auc, y_true, y_pred = eval_model(model, dl_test, device, criterion)
    print(f"Test | acc {te_acc:.4f}  f1 {te_f1:.4f}  auc {te_auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes))

    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
