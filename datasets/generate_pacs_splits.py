import os
import random

# Configuration parameters, strictly matching your PACS directory structure
DATASET_ROOT = "./datasets/PACS"
OUTPUT_TXT_DIR = "./datasets/pacs"  # Corresponds to DomainNet's domainnet folder
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
FIRST_SPLIT_TEST_RATIO = 0.1  # Overall 90/10 split into trainval pool / test set
SEED = 42
SUPPORT_IMG_EXT = (".jpg", ".png", ".jpeg")

random.seed(SEED)
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

for domain in DOMAINS:
    domain_all_samples = {cls: [] for cls in CLASS_NAMES}
    # Iterate all images under each category in the current domain
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_ROOT, domain, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(SUPPORT_IMG_EXT):
                # Generate relative path consistent with DomainNet format: domain/class/image_name
                rel_path = os.path.join(domain, cls, img_name)
                domain_all_samples[cls].append(rel_path)
    
    # First-level stratified split: 90% per class into trainval pool, 10% into fixed test set
    trainval_pool = []
    test_set = []
    for cls, img_list in domain_all_samples.items():
        random.shuffle(img_list)
        total_num = len(img_list)
        test_num = int(total_num * FIRST_SPLIT_TEST_RATIO)
        test_pool_cls = img_list[:test_num]
        trainval_pool_cls = img_list[test_num:]
        # Add to corresponding set with label
        label = CLASS_TO_LABEL[cls]
        test_set.extend([(p, label) for p in test_pool_cls])
        trainval_pool.extend([(p, label) for p in trainval_pool_cls])
    
    # Save txt files: {domain}_train.txt stores trainval pool (for later internal train/val split), {domain}_test.txt stores fixed test set
    with open(os.path.join(OUTPUT_TXT_DIR, f"{domain}_train.txt"), "w", encoding="utf-8") as f:
        for rel_p, lbl in trainval_pool:
            f.write(f"{rel_p} {lbl}\n")
    with open(os.path.join(OUTPUT_TXT_DIR, f"{domain}_test.txt"), "w", encoding="utf-8") as f:
        for rel_p, lbl in test_set:
            f.write(f"{rel_p} {lbl}\n")
    print(f"Domain {domain} split complete; train/test txt files generated.")