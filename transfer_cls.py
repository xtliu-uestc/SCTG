import os
import time
import clip
import torch
import torchvision
import wandb
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import Subset, random_split
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import torch.nn.functional as F
from scipy import stats as st
import argparse
import torchvision.models as models
from util.tool import refine_classname, topk, _convert_image_to_rgb, add_weight_decay
from util.get_index import get_index
from torchvision.transforms import (
    Compose,
    ToTensor,
    InterpolationMode,
)
from sklearn.metrics import roc_auc_score
import cv2
import json
from dataset import get_domainnet_merged_dataset, DomainNetDataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg


class Pertubation_LLM(torch.nn.Module):
    def __init__(self, args, pad_h, pad_w, model_name, model_llm, clip_model, device):
        super().__init__()
        # mask=1: border padding region (learnable visual prompt region, updated during training)
        # mask=0: central original image region (never updated, preserves original pixels)
        mask = torch.ones((3, 224, 224))
        mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0
        self.register_buffer('mask', mask)

        delta = torch.zeros((3, 224, 224))
        delta.requires_grad = True
        self.perturbation = torch.nn.Parameter(delta.float(), requires_grad=True)

        self.clip_model = clip_model
        self.model_name = model_name
        self.device = device

        if model_name == 'minigpt-4':
            from util.minigpt4_utils import load_model, model_loss, CLS_PROMPT, model_loglikelihood_for_postfixes
            self.model = load_model(model_llm, device)
            self.loss = model_loss
            self.prompt = CLS_PROMPT
            self.probability = model_loglikelihood_for_postfixes

        elif model_name == 'instructblip':
            from util.instructblip_utils import load_model, model_loss, model_loglikelihood_for_postfixes, CLS_PROMPT
            self.model = load_model(model_llm, device)
            self.probability = model_loglikelihood_for_postfixes
            self.loss = model_loss
            self.prompt = CLS_PROMPT
        elif model_name == 'bliva':
            from util.bliva_utils import load_model, model_loss, model_loglikelihood_for_postfixes, CLS_PROMPT
            self.model = load_model(model_llm, device)
            self.probability = model_loglikelihood_for_postfixes
            self.loss = model_loss
            self.prompt = CLS_PROMPT

    def forward(self, images, class_names, prompt=None):
        if prompt is None:
            prompt = "This is a photo of a"
        if type(prompt) is str:
            prompt = [prompt] * images.shape[0]
        assert len(prompt) == images.shape[0]
        samples = {"image": images, "text_input": prompt, "prompt": self.prompt, "text_output": class_names}
        loss = self.loss(self.model, samples)['loss']
        return loss

    def infer(self, images, class_names, prompt=None):
        if prompt is None:
            prompt = "This is a photo of a"
        if type(prompt) is str:
            prompt = [prompt] * images.shape[0]
        assert len(prompt) == images.shape[0]
        samples = {"image": images, "text_input": prompt, "prompt": self.prompt}
        probs = self.probability(self.model, samples, class_names)
        return probs


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")

    # training
    parser.add_argument("--batch_size_train", type=int, default=16, help="batch_size")
    parser.add_argument("--batch_size_eval", type=int, default=32, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--Nc", type=int, default=300,
                        help="Number of batches per chunk for CMPL (DomainNet: 300, PACS: 100)")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=1, help="learning rate")

    # model
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    parser.add_argument("--model_name", type=str, default="minigpt-4")
    parser.add_argument("--model_llm", type=str, default="vicuna7b")
    parser.add_argument("--llm_loss", action='store_true', default=True)
    parser.add_argument("--post_load_black", action='store_true', default=False)
    parser.add_argument("--target_models", nargs='+', default=[])
    parser.add_argument("--fca", type=float, default=0.003)
    parser.add_argument("--da", type=float, default=1.0, help="weight for domain adaptation entropy loss (SHOT)")
    parser.add_argument("--lambda_clip", type=float, default=0.5, help="weight for CLIP pseudo-label loss (λ_CLIP)")
    parser.add_argument("--alpha", type=float, default=0.01, help="CMGA gradient shift coefficient α (paper eq15)")
    # Gradient Noise Regularization
    parser.add_argument("--use_grad_noise", action='store_true',
                        help="Inject small noise into perturbation gradient (regularization)")
    parser.add_argument("--grad_noise_scale", type=float, default=1e-7,
                        help="Scale of gradient noise (default 1e-7, matching FFT round-trip error)")
    # CLIP Cosine Alignment Loss + Gradient Balance
    parser.add_argument("--lambda_align", type=float, default=0.0,
                        help="Weight for CLIP cosine alignment loss (0 to disable, suggested 0.5)")
    parser.add_argument("--lambda_balance", type=float, default=0.0,
                        help="CLIP gradient boost factor (0 to disable, suggested 0.5 for cartoon/sketch)")
    parser.add_argument("--prompt_size", type=int, default=30, help="size for visual prompts")
    parser.add_argument("--clip_weight_path", type=str,
                        default="./model_weights/ViT-B-32.pt",
                        help="Local path of CLIP .pt weight")

    # dataset
    parser.add_argument("--root", type=str, default=os.path.expanduser("../data"), help="dataset")
    parser.add_argument("--dataset", type=str, default="DomainNet", help="dataset")
    parser.add_argument("--image_size", type=int, default=164, help="image size")
    parser.add_argument("--target_domain", type=str, default="clipart",
                        help="target domain for single-domain adaptation (clipart/infograph/painting/quickdraw/real/sketch)")

    # save
    parser.add_argument("--save_path", type=str, default="./save/", help="path to save models")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="seed for initializing training")

    # eval
    parser.add_argument('--evaluate', action='store_true', help='Perform evaluation only')
    parser.add_argument("--checkpoint", type=str, help="The checkpoint of trained model")

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--project", type=str, default="TVP", help="The name of wandb project name")
    parser.add_argument("--job_name", type=str, default="DomainNet", help="The name of wandb job name")
    parser.add_argument("--entity", type=str, default="", help="Your user name of wandb")
    parser.add_argument("--roc_auc", action="store_true", help="whether to calculate roc auc")

    args = parser.parse_args()
    return args


def sample_indices(dataset, classes, few_shot, name):
    label_index = {}
    for k in classes:
        label_index[k] = []
    for i in range(len(dataset)):
        target = dataset[i][1]
        label_index[classes[target]].append(i)
    chosen_index = []
    print([len(label_index[k]) for k in label_index.keys()])
    for key in classes:
        chosen_index.extend(random.choices(label_index[key], k=few_shot))
    random.shuffle(chosen_index)
    return chosen_index


def sample_percent_indices(dataset, classes, percent, name):
    label_index = {}
    for k in classes:
        label_index[k] = []
    for i in range(len(dataset)):
        target = dataset[i][1]
        label_index[classes[target]].append(i)
    chosen_index = []
    for key in classes:
        chosen_index.extend(random.choices(label_index[key], k=int(percent * len(label_index[key]))))
    random.shuffle(chosen_index)
    return chosen_index


def main():
    args = parse_option()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    save_name = f'{args.model_name}_domain_{args.target_domain}_lr_{args.learning_rate}_FCA_{args.fca}_DA_{args.da}_CLIP_{args.lambda_clip}'
    if args.use_grad_noise:
        save_name += f'_gn{args.grad_noise_scale}'
    save_name += f'_align{args.lambda_align}_bal{args.lambda_balance}'
    args.save_path = os.path.join(args.save_path, save_name)
    
    print("=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    print(f"Dataset:          {args.dataset}")
    print(f"Target Domain:    {args.target_domain}")
    print(f"Model Name:       {args.model_name}")
    print("-" * 70)
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size (Train): {args.batch_size_train}")
    print(f"Batch Size (Eval):  {args.batch_size_eval}")
    print(f"Prompt Size:      {args.prompt_size}")
    print(f"Nc (CMPL Chunk):  {args.Nc}")
    print("-" * 70)
    print(f"λ_FCA (FCA weight):  {args.fca}")
    print(f"λ_DA (DA weight):    {args.da}")
    print(f"λ_CLIP (CLIP weight):{args.lambda_clip}")
    print(f"α (CMGA alpha):      {args.alpha}")
    print("-" * 70)
    print(f"use_grad_noise:    {args.use_grad_noise}")
    print(f"grad_noise_scale:  {args.grad_noise_scale}")
    print(f"λ_align (align loss): {args.lambda_align}")
    print(f"λ_balance (grad balance): {args.lambda_balance}")
    print("-" * 70)
    print(f"Random Seed:      {args.seed}")
    print(f"Save Path:        {args.save_path}")
    print(f"Run Mode:         {'Evaluation' if args.evaluate else 'Training'}")
    if args.evaluate:
        print(f"Checkpoint:       {args.checkpoint}")
    print("=" * 70)
    print()

    log_wandb = args.use_wandb
    project = args.project
    job_name = args.job_name
    save_path = args.save_path

    if not os.path.exists(save_path) and not args.evaluate:
        os.makedirs(save_path)
    if log_wandb:
        wandb.init(project=str(project), name=str(args.dataset))

    clip_model, preprocess = clip.load(args.clip_weight_path, device=device)
    clip_model.requires_grad_(False)
    clip_model.eval()
    _, preprocess_test = clip.load(args.clip_weight_path, device=device)
    del _

    normalization = preprocess.transforms[-1]
    preprocess_test.transforms.pop(-1)

    preprocess = Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.RandomCrop(args.image_size),
        _convert_image_to_rgb,
        ToTensor(),
    ])
    preprocess_test = Compose([
        torchvision.transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(size=(args.image_size, args.image_size)),
        _convert_image_to_rgb,
        ToTensor(),
    ])

    if args.dataset == "DomainNet":
        from dataset import DomainNetDataset
        train_set = DomainNetDataset(root=args.root, domain_name=args.target_domain, split='train',
                                     transform=preprocess)
        classes_names = refine_classname(train_set.classes)

        text_tokens = clip.tokenize([f"This is a photo of a {c}" for c in classes_names]).to(device)
        clip_text_feats = clip_model.encode_text(text_tokens)
        clip_text_feats = clip_text_feats / clip_text_feats.norm(dim=-1, keepdim=True)

        sampled_valset = DomainNetDataset(root=args.root, domain_name=args.target_domain, split='val',
                                          transform=preprocess_test)
        test_set = DomainNetDataset(root=args.root, domain_name=args.target_domain, split='test',
                                    transform=preprocess_test)

    if args.dataset == "PACS":
        from dataset import PACSDataset
        args.Nc = 100
        train_set = PACSDataset(root=args.root, domain_name=args.target_domain, split='train', transform=preprocess)
        classes_names = refine_classname(train_set.classes)

        text_tokens = clip.tokenize([f"This is a photo of a {c}" for c in classes_names]).to(device)
        clip_text_feats = clip_model.encode_text(text_tokens)
        clip_text_feats = clip_text_feats / clip_text_feats.norm(dim=-1, keepdim=True)

        sampled_valset = PACSDataset(root=args.root, domain_name=args.target_domain, split='val',
                                     transform=preprocess_test)
        test_set = PACSDataset(root=args.root, domain_name=args.target_domain, split='test', transform=preprocess_test)
    if args.dataset == "OfficeHome":
        from dataset import OfficeHomeDataset
        args.Nc = 150
        train_set = OfficeHomeDataset(root=args.root, domain_name=args.target_domain, split='train',
                                      transform=preprocess)
        classes_names = refine_classname(train_set.classes)

        text_tokens = clip.tokenize([f"This is a photo of a {c}" for c in classes_names]).to(device)
        clip_text_feats = clip_model.encode_text(text_tokens)
        clip_text_feats = clip_text_feats / clip_text_feats.norm(dim=-1, keepdim=True)

        sampled_valset = OfficeHomeDataset(root=args.root, domain_name=args.target_domain, split='val',
                                           transform=preprocess_test)
        test_set = OfficeHomeDataset(root=args.root, domain_name=args.target_domain, split='test',
                                     transform=preprocess_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True,
                              pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_eval, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(sampled_valset, batch_size=args.batch_size_eval, shuffle=False,
                            pin_memory=True, num_workers=args.num_workers)

    prompt = Pertubation_LLM(args, args.prompt_size, args.prompt_size, args.model_name, args.model_llm, clip_model,
                             device)
    prompt = prompt.to(device)

    target_prompts = []
    if not args.post_load_black:
        for target in args.target_models:
            target_prompts.append(
                Pertubation_LLM(args, args.prompt_size, args.prompt_size, target, args.model_llm, None, device))
    else:
        target_prompts = args.target_models

    pad_length = int((224 - args.image_size) / 2)
    pad_dim = (pad_length, pad_length, pad_length, pad_length)

    prompt.model.requires_grad_(False)
    param_groups = add_weight_decay(prompt, 0.0, skip_list=("perturbation"))
    optimizer = torch.optim.SGD(param_groups, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    max_acc = 0
    max_auc = 0
    results = []

    if not args.evaluate:
        if log_wandb:
            wandb.watch(prompt)

        print('Start Training')
        for e in range(args.epochs):
            train_top1, args = train_with_prompt(
                args=args, epoch=e, train_loader=train_loader, prompt=prompt,
                text_inputs=classes_names, pad_dim=pad_dim, criterion=criterion,
                optim=optimizer, normalization=normalization, device=device,
                val_loader=val_loader, target_prompts=target_prompts,
                preprocess_test=preprocess_test, clip_model=clip_model,
                clip_text_feats=clip_text_feats
            )

            schedule.step()
            curr_val_acc = eval(args, val_loader, prompt, pad_dim, classes_names, normalization, device, clip_model,
                                clip_text_feats)

            print(f"\n[Epoch {e + 1}/{args.epochs}] Train top1: {train_top1:.4f}, Val top1: {curr_val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if curr_val_acc >= max_acc:
                max_acc = curr_val_acc
                model_state = prompt.state_dict()
                save_dict = {"perturbation": model_state["perturbation"]}
                os.makedirs(save_path, exist_ok=True)
                torch.save(save_dict, os.path.join(save_path, "checkpoint_best.pth"))
                print(f"New best val accuracy: {max_acc:.4f}, saved checkpoint_best.pth")

            model_state = prompt.state_dict()
            save_dict = {"perturbation": model_state["perturbation"]}
            os.makedirs(save_path, exist_ok=True)
            torch.save(save_dict, os.path.join(save_path, "checkpoint_last.pth"))

            results.append({
                "epoch": e + 1, "train_top1": train_top1, "val_top1": curr_val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            print("=" * 60)

            if (e + 1) == args.epochs:
                test_item = {}
                src_test_acc = eval(args, test_loader, prompt, pad_dim, classes_names, normalization, device,
                                    clip_model, clip_text_feats)
                test_item[prompt.model_name] = src_test_acc
                test_acc1 = src_test_acc

                for target_prompt in target_prompts:
                    if args.post_load_black:
                        target_prompt = Pertubation_LLM(
                            args, args.prompt_size, args.prompt_size,
                            target_prompt, args.model_llm, clip_model, device
                        )
                    target_prompt.perturbation.data = prompt.perturbation.data.clone()
                    tgt_test_acc = eval(args, test_loader, target_prompt, pad_dim, classes_names, normalization, device,
                                        clip_model, clip_text_feats)
                    test_item[target_prompt.model_name] = tgt_test_acc
                    if args.post_load_black:
                        del target_prompt

                results.append(test_item)
                print("\n" + "=" * 60)
                print(f"Target Domain: {args.target_domain} testing completed")
                print(f"Best val accuracy: {max_acc:.4f}")
                print(f"Source model ({prompt.model_name}) test accuracy: {src_test_acc:.4f}")
                print(f"Prompt max abs: {prompt.perturbation.abs().max():.4f}")

                if log_wandb:
                    log_status = {
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_top1": train_top1,
                        "val_acc": curr_val_acc,
                        "test_acc": test_acc1,
                    }
                    wandb.log(log_status, step=e)
    else:
        print('Start Evaluating')
        if args.checkpoint is not None:
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            perturbation_state = prompt.state_dict()
            perturbation_state["perturbation"] = state_dict["perturbation"]
            prompt.load_state_dict(perturbation_state)

        trained_perturbation = prompt.perturbation.data.clone()
        src_test_acc = eval(args, test_loader, prompt, pad_dim, classes_names, normalization, device, clip_model,
                            clip_text_feats)
        print(f"\nTarget Domain: {args.target_domain}")
        print(f"Source model ({prompt.model_name}) test acc: {src_test_acc:.4f}")

        del prompt
        torch.cuda.empty_cache()

        if args.target_models:
            print("\n" + "=" * 60)
            print("Evaluating transfer performance on target models")
            print("=" * 60)
            for target_name in args.target_models:
                target_prompt = Pertubation_LLM(
                    args, args.prompt_size, args.prompt_size,
                    target_name, args.model_llm, None, device
                )
                target_prompt.perturbation.data = trained_perturbation.clone().to(device)
                tgt_test_acc = eval(args, test_loader, target_prompt, pad_dim, classes_names, normalization, device,
                                    clip_model, clip_text_feats)
                print(f"Target model ({target_name}) test acc: {tgt_test_acc:.4f}")
                del target_prompt
                torch.cuda.empty_cache()


@torch.no_grad()
def generate_cmpl_pseudo_labels_chunk(args, prompt, chunk_batches, text_inputs, pad_dim, normalization, device,
                                      clip_text_feats, img_encoder):
    all_mllm_preds = []
    all_clip_preds = []
    all_mllm_features = []

    for images, _, _ in chunk_batches:
        images = F.pad(images, pad_dim, "constant", value=0)
        images = images.to(device)
        noise = prompt.perturbation.to(device)
        prompted_images = normalization(images + noise)

        clip_img_feats = prompt.clip_model.encode_image(prompted_images)
        clip_img_feats = clip_img_feats / clip_img_feats.norm(dim=-1, keepdim=True)
        clip_logits = prompt.clip_model.logit_scale.exp() * clip_img_feats @ clip_text_feats.T
        clip_probs = clip_logits.softmax(dim=-1)
        clip_pred = torch.argmax(clip_probs, dim=1)
        all_clip_preds.append(clip_pred.cpu().numpy())

        k_train = 5
        _, topk_indices = torch.topk(clip_probs, k=k_train, dim=-1)
        candidate_idx = torch.unique(topk_indices.flatten()).sort().values
        candidate_texts = [text_inputs[i] for i in candidate_idx.tolist()]

        mllm_candidate_probs = prompt.infer(prompted_images, candidate_texts)
        idx_map = {cand_idx: i for i, cand_idx in enumerate(candidate_idx.tolist())}
        batch_size = prompted_images.shape[0]
        mllm_pred = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            sample_topk = topk_indices[i].tolist()
            sample_cand_pos = torch.tensor([idx_map[y] for y in sample_topk], device=device)
            sample_probs = mllm_candidate_probs[i, sample_cand_pos]
            mllm_pred[i] = sample_topk[torch.argmax(sample_probs).item()]
        all_mllm_preds.append(mllm_pred.cpu().numpy())

        mllm_feats = img_encoder(prompted_images)
        mllm_feats = torch.mean(mllm_feats, dim=1)
        mllm_feats = mllm_feats / mllm_feats.norm(dim=-1, keepdim=True)
        all_mllm_features.append(mllm_feats.cpu().numpy())

    all_mllm_preds = np.concatenate(all_mllm_preds)
    all_clip_preds = np.concatenate(all_clip_preds)
    all_mllm_features = np.concatenate(all_mllm_features)

    num_samples = len(all_mllm_preds)
    num_classes = len(text_inputs)

    reliable_mask = (all_mllm_preds == all_clip_preds)
    reliable_indices = np.where(reliable_mask)[0]

    if args.dataset == "DomainNet":
        k_neighbors = 30
    elif args.dataset == "PACS":
        k_neighbors = 10
    elif args.dataset == "OfficeHome":
        k_neighbors = 20
    else:
        k_neighbors = 30

    k_neighbors = min(k_neighbors, num_samples - 1)
    k_neighbors = max(k_neighbors, 1)

    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', n_jobs=-1)
    knn.fit(all_mllm_features)
    distances, indices = knn.kneighbors(all_mllm_features)

    similarities = 1.0 - distances
    similarities = np.clip(similarities, 0.0, None)
    similarities = similarities ** 3

    row_idx = np.repeat(np.arange(num_samples), k_neighbors)
    col_idx = indices.flatten()
    sim_values = similarities.flatten()
    A_sparse = csr_matrix((sim_values, (row_idx, col_idx)), shape=(num_samples, num_samples))

    W_sparse = A_sparse + A_sparse.T
    W_sparse.setdiag(0)
    W_sparse.eliminate_zeros()

    degree = np.array(W_sparse.sum(axis=1)).flatten()
    degree = np.clip(degree, 1e-8, None)
    degree_inv_sqrt = 1.0 / np.sqrt(degree)
    D_inv_sqrt = diags(degree_inv_sqrt)
    W_norm = D_inv_sqrt @ W_sparse @ D_inv_sqrt

    Y_init = np.zeros((num_samples, num_classes), dtype=np.float32)
    Y_init[reliable_indices, all_mllm_preds[reliable_indices]] = 1.0

    beta_lp = 0.8
    I_sparse = diags(np.ones(num_samples, dtype=np.float32))
    A_lp = I_sparse - beta_lp * W_norm.astype(np.float32)

    Z = np.zeros_like(Y_init)
    cg_tol = 1e-6
    for c in range(num_classes):
        y_c = Y_init[:, c]
        z_c, _ = cg(A_lp, y_c, tol=cg_tol, maxiter=1000)
        Z[:, c] = z_c

    label_dist = Z
    tau_lp = 0.01
    label_dist = label_dist / tau_lp
    label_dist = np.exp(label_dist - np.max(label_dist, axis=1, keepdims=True))
    label_dist = label_dist / np.sum(label_dist, axis=1, keepdims=True)

    pseudo_labels = np.argmax(label_dist, axis=1)
    entropy = -np.sum(label_dist * np.log(label_dist + 1e-8), axis=1)
    sample_weights = 1.0 - entropy / np.log(num_classes)
    sample_weights = np.clip(sample_weights, 0.0, 1.0)

    return pseudo_labels, sample_weights


def train_with_prompt(
        args, epoch, train_loader, prompt, text_inputs, pad_dim,
        criterion, optim, normalization, device, val_loader,
        target_prompts, preprocess_test, clip_model, clip_text_feats=None
):
    device = prompt.device
    start_time = time.time()
    lr = optim.param_groups[0]["lr"]
    all_loss = []
    all_top1 = []
    idx = 0

    if prompt.model_name == 'minigpt-4':
        img_encoder = lambda x: prompt.model.encode_img(x)[0]
    elif prompt.model_name == 'instructblip':
        def img_encoder(x):
            with prompt.model.maybe_autocast():
                image_embeds = prompt.model.ln_vision(prompt.model.visual_encoder(x))
            return image_embeds.float()

    train_iter = iter(train_loader)
    chunk_size = args.Nc
    total_batches = len(train_loader)
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}")

    while True:
        chunk_batches = []
        for _ in range(chunk_size):
            try:
                batch = next(train_iter)
                chunk_batches.append(batch)
            except StopIteration:
                break
        if not chunk_batches:
            break

        chunk_pseudo_labels, chunk_sample_weights = generate_cmpl_pseudo_labels_chunk(
            args=args, prompt=prompt, chunk_batches=chunk_batches, text_inputs=text_inputs,
            pad_dim=pad_dim, normalization=normalization, device=device,
            clip_text_feats=clip_text_feats, img_encoder=img_encoder
        )

        batch_offset = 0
        for samples in chunk_batches:
            images, true_labels, indices = samples
            batch_len = images.shape[0]

            labels = torch.tensor(
                chunk_pseudo_labels[batch_offset:batch_offset + batch_len],
                dtype=torch.long, device=device
            )
            batch_weights = torch.tensor(
                chunk_sample_weights[batch_offset:batch_offset + batch_len],
                dtype=torch.float32, device=device
            )
            batch_offset += batch_len
            prompts = None

            clean_images = F.pad(images, pad_dim, "constant", value=0).to(device)
            targets = [text_inputs[labels[j].item()] for j in range(len(labels))]

            noise = prompt.perturbation
            noise = prompt.perturbation.to(device)
            noise = noise.repeat(images.size(0), 1, 1, 1)
            noise.retain_grad()

            img_grad = normalization(clean_images + noise)

            raw_mllm_vec = prompt(img_grad, targets, prompts)
            raw_mllm_loss = (raw_mllm_vec * batch_weights).mean()
            grad_mllm_delta = torch.autograd.grad(
                outputs=raw_mllm_loss, inputs=prompt.perturbation,
                retain_graph=True, create_graph=False
            )[0].detach().clone()

            clip_feat_grad = prompt.clip_model.encode_image(img_grad)
            clip_feat_grad = clip_feat_grad / clip_feat_grad.norm(dim=-1, keepdim=True)
            raw_logits_grad = prompt.clip_model.logit_scale.exp() * clip_feat_grad @ clip_text_feats.T
            raw_clip_vec = F.cross_entropy(raw_logits_grad, labels, reduction='none')
            raw_clip_loss = (raw_clip_vec * batch_weights).mean()
            grad_clip_delta = torch.autograd.grad(
                outputs=raw_clip_loss, inputs=prompt.perturbation,
                retain_graph=False, create_graph=False
            )[0].detach().clone()

            # Mask gradients (keep only gradients in border learnable region)
            grad_mllm_delta = grad_mllm_delta * prompt.mask
            grad_clip_delta = grad_clip_delta * prompt.mask

            grad_mllm_flat = grad_mllm_delta.view(-1)
            grad_clip_flat = grad_clip_delta.view(-1)
            norm_mllm = torch.norm(grad_mllm_flat)
            norm_clip = torch.norm(grad_clip_flat)
            eps = torch.tensor(1e-8, device=device)

            G_norm = norm_mllm * norm_clip + eps
            alpha = torch.tensor(args.alpha, device=device)

            delta_mllm_ga = prompt.perturbation - alpha * grad_clip_delta / G_norm
            delta_clip_ga = prompt.perturbation - alpha * grad_mllm_delta / G_norm

            img_total = normalization(clean_images + noise)

            # Pre-compute CLIP features (fixes undefined variable bug when da=0)
            clip_feat_total = prompt.clip_model.encode_image(img_total)
            clip_feat_total = clip_feat_total / clip_feat_total.norm(dim=-1, keepdim=True)
            raw_logits_total = prompt.clip_model.logit_scale.exp() * clip_feat_total @ clip_text_feats.T

            images_mllm_ga = normalization(clean_images + delta_mllm_ga)
            ga_mllm_vec = prompt(images_mllm_ga, targets, prompts)
            ga_mllm_loss = (ga_mllm_vec * batch_weights).mean()

            images_clip_ga = normalization(clean_images + delta_clip_ga)
            clip_ga_feat = prompt.clip_model.encode_image(images_clip_ga)
            clip_ga_feat = clip_ga_feat / clip_ga_feat.norm(dim=-1, keepdim=True)
            ga_logits = prompt.clip_model.logit_scale.exp() * clip_ga_feat @ clip_text_feats.T
            ga_clip_vec = F.cross_entropy(ga_logits, labels, reduction='none')
            ga_clip_loss = (ga_clip_vec * batch_weights).mean()

            da_loss = 0.0
            if args.da > 0 and clip_text_feats is not None:
                # Numerically stable: cast logits to fp32 + log_softmax to avoid exp overflow
                logits32 = raw_logits_total.float()
                log_p = F.log_softmax(logits32, dim=-1)
                p_clip = log_p.exp()
                # Conditional entropy H(p) = -sum p*log p; zero-prob entries produce no 0*log(0) -inf
                cond_entropy = -torch.sum(p_clip * log_p, dim=1).mean()
                mean_probs = p_clip.mean(dim=0)
                marg_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                da_loss = cond_entropy - marg_entropy

            emb_loss = 0.0
            if args.fca > 0:
                clean_feats_raw = img_encoder(normalization(clean_images))
                clean_batch_feats = clean_feats_raw.mean(dim=1)

                prompted_feats_raw = img_encoder(img_total)
                prompted_batch_feats = prompted_feats_raw.mean(dim=1)

                # Compute FCA loss (exactly matches paper formula)
                diff = prompted_batch_feats - clean_batch_feats
                emb_loss = torch.norm(diff, dim=-1).mean()

            # CLIP Cosine Alignment Loss: pulls prompted CLIP features toward their
            # class text prototypes. Gradient ∂(1-cos_sim)/∂feat ≠ 0 always—unlike
            # CE which vanishes for confident samples.
            align_loss = 0.0
            if args.lambda_align > 0:
                cos_to_correct = (clip_feat_total * clip_text_feats[labels]).sum(dim=-1)
                align_loss = ((1.0 - cos_to_correct) * batch_weights).mean()

            prompt.clip_model.logit_scale.data.clamp_(0, 4.6052)
            loss = ga_mllm_loss + args.lambda_clip * ga_clip_loss + args.da * da_loss + args.fca * emb_loss + args.lambda_align * align_loss

            with torch.no_grad():
                k_train = 5
                clip_probs = raw_logits_total.softmax(dim=-1)
                _, topk_indices = torch.topk(clip_probs, k=k_train, dim=-1)
                candidate_idx = torch.unique(topk_indices.flatten()).sort().values
                candidate_texts = [text_inputs[i] for i in candidate_idx.tolist()]
                mllm_candidate_probs = prompt.infer(img_total, candidate_texts)
                idx_map = {cand_idx: i for i, cand_idx in enumerate(candidate_idx.tolist())}
                batch_size = img_total.shape[0]
                batch_pred = torch.zeros(batch_size, dtype=torch.long, device=device)
                for i in range(batch_size):
                    sample_topk = topk_indices[i].tolist()
                    sample_cand_pos = torch.tensor([idx_map[y] for y in sample_topk], device=device)
                    sample_probs = mllm_candidate_probs[i, sample_cand_pos]
                    batch_pred[i] = sample_topk[torch.argmax(sample_probs).item()]
                batch_acc = (batch_pred == labels).float().mean().item()
                all_top1.append(batch_acc)

            optim.zero_grad()
            loss.backward()

            grad_p_t = prompt.perturbation.grad.clone()
            grad_p_t = grad_p_t * prompt.mask

            # Gradient noise injection (equivalent to FFT/IFFT floating-point noise)
            if args.use_grad_noise:
                noise = torch.randn_like(grad_p_t) * args.grad_noise_scale
                grad_p_t = grad_p_t + noise

            if torch.isnan(grad_p_t).any() or torch.isinf(grad_p_t).any():
                prompt.zero_grad()
                idx += 1
                pbar.update(1)

            # Gradient Balance: MLLM gradient norm is 10~2000x CLIP's, so the
            # update direction is dominated by MiniGPT-4. Re-use the per-model
            # gradients already computed for CMGA (Branch A) to boost the CLIP
            # signal before the final normalization.
            if args.lambda_balance > 0:
                g_mllm_norm = torch.norm(grad_mllm_delta)
                g_clip_norm = torch.norm(grad_clip_delta)
                if g_mllm_norm > 1e-10 and g_clip_norm > 1e-10:
                    g_clip_dir = grad_clip_delta / g_clip_norm
                    grad_p_t = grad_p_t + args.lambda_balance * g_clip_dir * g_mllm_norm

            g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
            scaled_g = grad_p_t / (g_norm + 1e-10)

            curr_lr = optim.param_groups[0]["lr"]
            update_step = scaled_g * curr_lr

            prompt.perturbation.data = prompt.perturbation.data - update_step.detach()
            prompt.zero_grad()

            all_loss.append(loss.float().detach().cpu().numpy())
            idx += 1
            pbar.update(1)

    pbar.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "At the {} epoch, the Lr is {}, the top1 is {} and training time  is {}".format(
            str(epoch), str(lr), str(np.mean(all_top1)), total_time_str))
    return np.mean(all_top1), args


@torch.no_grad()
def eval(args, test_loader, prompt, pad_dim, text_inputs, normalization, device, clip_model, clip_text_feats):
    device = prompt.device
    start_time = time.time()
    all_top1, all_top5 = [], []
    all_mean_logits = []
    all_gt_logits = []
    print("starting evaluation")
    for samples in tqdm(test_loader):
        images, labels, _ = samples
        prompts = None

        images = F.pad(images, pad_dim, "constant", value=0).to(device)
        noise = prompt.perturbation.to(device)
        images = normalization(images + noise)

        num_classes = len(text_inputs)
        k_test = min(10, num_classes)

        clip_img_feats = clip_model.encode_image(images)
        clip_img_feats = clip_img_feats / clip_img_feats.norm(dim=-1, keepdim=True)
        clip_probs = (clip_model.logit_scale.exp() * clip_img_feats @ clip_text_feats.T).softmax(dim=-1)
        _, topk_indices = torch.topk(clip_probs, k=k_test, dim=-1)

        candidate_idx = torch.unique(topk_indices.flatten()).sort().values
        candidate_texts = [text_inputs[i] for i in candidate_idx.tolist()]

        mllm_candidate_probs = prompt.infer(images, candidate_texts, prompts)

        batch_size = images.shape[0]
        probs = torch.full((batch_size, num_classes), float('-inf'), device=device)
        idx_map = {cand_idx: i for i, cand_idx in enumerate(candidate_idx.tolist())}

        for i in range(batch_size):
            for cand_idx in candidate_idx.tolist():
                probs[i, cand_idx] = mllm_candidate_probs[i, idx_map[cand_idx]]

        for k in range(images.shape[0]):
            all_mean_logits.append(probs[k].mean().item())
            all_gt_logits.append(probs[k][int(labels[k])].item())

        if len(text_inputs) > 5:
            top1, top5 = topk(probs, labels.to(device), ks=(1, 5))
            all_top1.extend(top1.cpu())
            all_top5.extend(top5.cpu())
        else:
            top1 = topk(probs, labels.to(device))
            all_top1.extend(top1[0].cpu())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    print(f"Average Mean Logits: {np.mean(all_mean_logits)}, Average GT Logits: {np.mean(all_gt_logits)}")
    return np.mean(all_top1)


if __name__ == "__main__":
    main()