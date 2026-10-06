"""
Generate OfficeHome train/test split txt files.
Structure: OfficeHome/{domain}/{class}/{image}.jpg
Same format as PACS txt files: {domain}/{class}/{image} {label_id}
"""
import os
import random

DATASET_ROOT = "./datasets/OfficeHome"
OUTPUT_TXT_DIR = "./datasets/officehome"
DOMAINS = ["Art", "Clipart", "Product", "Real World"]
FIRST_SPLIT_TEST_RATIO = 0.1
SEED = 42
SUPPORT_IMG_EXT = (".jpg", ".png", ".jpeg")

random.seed(SEED)
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

# Auto-detect class names from the first available domain
all_classes = set()
for domain in DOMAINS:
    domain_path = os.path.join(DATASET_ROOT, domain)
    if os.path.exists(domain_path):
        for cls_name in os.listdir(domain_path):
            cls_path = os.path.join(domain_path, cls_name)
            if os.path.isdir(cls_path):
                all_classes.add(cls_name)

CLASS_NAMES = sorted(all_classes)
CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

print(f"Found {len(CLASS_NAMES)} classes: {CLASS_NAMES[:5]}...")
print(f"Domains: {DOMAINS}")

for domain in DOMAINS:
    domain_path = os.path.join(DATASET_ROOT, domain)
    if not os.path.exists(domain_path):
        print(f"  [SKIP] {domain} not found")
        continue

    domain_all_samples = {cls: [] for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(domain_path, cls)
        if not os.path.exists(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(SUPPORT_IMG_EXT):
                rel_path = os.path.join(domain, cls, img_name)
                domain_all_samples[cls].append(rel_path)

    trainval_pool = []
    test_set = []
    for cls, img_list in domain_all_samples.items():
        random.shuffle(img_list)
        n_test = int(len(img_list) * FIRST_SPLIT_TEST_RATIO)
        label = CLASS_TO_LABEL[cls]
        test_set.extend([(p, label) for p in img_list[:n_test]])
        trainval_pool.extend([(p, label) for p in img_list[n_test:]])

    with open(os.path.join(OUTPUT_TXT_DIR, f"{domain}_train.txt"), "w", encoding="utf-8") as f:
        for rel_p, lbl in trainval_pool:
            f.write(f"{rel_p} {lbl}\n")
    with open(os.path.join(OUTPUT_TXT_DIR, f"{domain}_test.txt"), "w", encoding="utf-8") as f:
        for rel_p, lbl in test_set:
            f.write(f"{rel_p} {lbl}\n")

    print(f"  {domain}: {len(trainval_pool)} trainval, {len(test_set)} test")
print("Done!")
