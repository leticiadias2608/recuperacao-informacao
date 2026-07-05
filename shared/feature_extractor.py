import pathlib
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# --------
# Configuração
# --------

FEATURES_DIR = pathlib.Path("features")
FEATURES_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -----
# Carregamento do modelo
# -----
def load_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    return feature_extractor

# -----
# Leitura dataset
# -----
def scan_dataset(root_dir):

    root = pathlib.Path(root_dir)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    entries = []
    class_names = sorted([
        d.name for d in root.iterdir() if d.is_dir()
    ])
    
    """ if not class_names:
        # Tenta nível a mais (ex: corel_raw/Corel-1K/<classes>)
        for sub in root.iterdir():
            if sub.is_dir():
                inner = sorted([d.name for d in sub.iterdir() if d.is_dir()])
                if inner:
                    print(f"[INFO] Usando subpasta '{sub.name}' como raiz das classes")
                    root = sub
                    class_names = inner
                    break
    """
    if not class_names:
        raise FileNotFoundError(
            f"Nenhuma subpasta de classes encontrada em '{root_dir}'.\n"
        )
    
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_dir = root / cls_name
        imgs = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in supported
        ]
        imgs.sort()
        for img_path in imgs:
            entries.append((str(img_path), label_map[cls_name], cls_name))
    
    return entries, class_names


# ----
# Extração de features
# -----

def extract_features(entries, model, device, batch_size=32):
    """
    Extrai vetores de características para todas as imagens.
    Retorna: features (N,2048), labels (N,), paths (N,)
    """
    all_features = []
    all_labels   = []
    all_paths    = []
    
    n = len(entries)
    print(f"\n[INFO] Extraindo características de {n} imagens (batch={batch_size})...")
    
    skipped = 0
    batch_imgs   = []
    batch_labels = []
    batch_paths  = []
    
    def flush_batch():
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model(tensor)           # (B, 2048, 1, 1)
            feats = feats.squeeze(-1).squeeze(-1)  # (B, 2048)
        all_features.append(feats.cpu().numpy())
        all_labels.extend(batch_labels)
        all_paths.extend(batch_paths)
        batch_imgs.clear()
        batch_labels.clear()
        batch_paths.clear()
    
    for i, (img_path, label, cls_name) in enumerate(entries):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img)
            batch_imgs.append(tensor)
            batch_labels.append(label)
            batch_paths.append(img_path)
        except Exception as e:
            skipped += 1
            print(f"[WARN] Imagem ignorada: {img_path} — {e}")
            continue
        
        if len(batch_imgs) == batch_size:
            flush_batch()
    
    flush_batch()  # processa sobras
    
    features = np.vstack(all_features)   # (N, 2048)
    labels   = np.array(all_labels)
    paths    = np.array(all_paths)
    
    print(f"     Imagens processadas : {len(paths)}")
    print(f"     Imagens ignoradas   : {skipped}")
    print(f"     Shape das features  : {features.shape}  (N × 2048)")
    print(f"     Dtype               : {features.dtype}")
    
    return features, labels, paths

""" def l2_normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return features / norms """
 
# ----
# Salvamento das features
# ----
def save_features(features, labels, paths, class_names, normalize=True):
    raw_path  = FEATURES_DIR / "features_raw.npy"
    lab_path  = FEATURES_DIR / "labels.npy"
    pth_path  = FEATURES_DIR / "image_paths.npy"
    cls_path  = FEATURES_DIR / "class_names.npy"
    
    np.save(raw_path,  features)
    np.save(lab_path,  labels)
    np.save(pth_path,  paths)
    np.save(cls_path,  np.array(class_names))

    # Arquivo 1: index + vetor de features
    with open(FEATURES_DIR / "features.txt", "w") as f:
        for feat in features:
            valores = " ".join(map(str, feat.tolist()))
            f.write(f"{valores}\n")

    # Arquivo 2: index + label + class_name
    with open(FEATURES_DIR / "metadata.txt", "w") as f:
        for label in labels:
            class_name = class_names[label]
            f.write(f"{label} {class_name}\n")

