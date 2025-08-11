import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model


def meta_optimizeation(
    target_meta_loader,
    replace_optimizer,
    model,
    args,
    criterion,
    replace_scheduler,
    source_edge_index,
    target_edge_index,
):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)

            replace_optimizer.zero_grad()
            out = model.meta_prediction(
                source_edge_index, target_edge_index, target_link
            ).squeeze()
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()


@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label):
    model.eval()

    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except:
        auc = 1.0
    logging.info(f"{name} AUC: {auc:4f}")

    model.train()
    return auc
def get_test_positive_dict(data):
    """
    根據 test link（data.target_test_link）建立 test set user 的正樣本字典。
    回傳: {user_id: [item1, item2, ...]}
    """
    test_user_item_dict = {}
    test_link = data.target_test_link.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        u, i = u.item(), i.item()
        if u not in test_user_item_dict:
            test_user_item_dict[u] = []
        test_user_item_dict[u].append(i)
    return test_user_item_dict

def evaluate_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()
    hit_count = 0
    all_target_items = set(range(data.num_target_items))

    # ✅ 取得 test set 的 user -> positive items 對應關係
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())  # 直接使用 test set 的 user
    print(f"✅ Test set user count: {len(sim_users)}")

    total_users = 0
    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, set())
            if len(pos_items) > 1:
                print(f"⚠️ Warning: User {user_id} has {len(pos_items)} positives in test set.")

            if len(pos_items) == 0:
                continue

            # ✅ 第一步：選擇一個正樣本
            pos_item = list(pos_items)[0]
            # print(f"\n=== [User {user_id}] ===")
            # print(f"👉 Positive item: {pos_item}")

            # ✅ 第二步：挑選負樣本（從非正樣本中隨機抽 num_candidates 個）
            negative_pool = list(all_target_items - set(pos_items))
            if len(negative_pool) < num_candidates:
                # print(f"❌ Negative pool too small for user {user_id}, skipping.")
                continue

            sampled_negatives = random.sample(negative_pool, num_candidates)
            # print(f"🎯 Sampled {num_candidates} negatives: {sampled_negatives[:10]}...")

            # ✅ 第三步：組成候選清單（正例 + 負例），並打亂
            candidate_items = sampled_negatives + [pos_item]
            random.shuffle(candidate_items)
            # print(f"🧮 Candidate items (shuffled): {candidate_items[:10]}...")

            # ✅ 第四步：轉成 tensor 並送入模型計算分數
            user_tensor = torch.tensor([user_id] * len(candidate_items), device=device)
            item_tensor = torch.tensor(candidate_items, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
            top_k_items = [candidate_items[i] for i in top_k_indices]

            # print(f"📈 Top-{top_k} prediction: {top_k_items}")
            # print(f"✔️ Hit? {'Yes ✅' if pos_item in top_k_items else 'No ❌'}")

            if pos_item in top_k_items:
                hit_count += 1
            total_users += 1

    hit_ratio = hit_count / total_users if total_users > 0 else 0.0
    logging.info(f"[HIT_RATIO@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
    return hit_ratio

# 🔍 統計每個 cold item 在 test set 中出現的次數（有幾個 user 買過）
def count_cold_item_occurrences(data, cold_item_set):
    item_count = {item: 0 for item in cold_item_set}
    test_link = data.target_test_link.cpu().numpy()
    for u, i in zip(*test_link):
        if i in cold_item_set:
            item_count[i] += 1
    return item_count

def find_cold_item_strict(data, target_train_edge_index, target_test_edge_index):
    import numpy as np
    from collections import Counter

    train_edges = target_train_edge_index.cpu().numpy()
    test_edges = target_test_edge_index.cpu().numpy()
    overlap_users = set(data.raw_overlap_users.cpu().numpy())  # ⬅️ overlap user list
    overlap_users = set(data.raw_overlap_users.cpu().numpy())  # ⬅️ overlap user list
    print(f"Overlap users count: {len(overlap_users)}")
    print(f"Overlap users sample: {list(overlap_users)[:10]}")  # 印前10個用戶ID


    # Step 1: 統計 overlap user 在 test set 中點擊的 item 次數
    test_user, test_item = test_edges
    item_counter = Counter()

    for u, i in zip(test_user, test_item):
        if u in overlap_users:
            item_counter[i] += 1

    candidate_items = {i for i, cnt in item_counter.items() if cnt == 1}

    train_items = set(train_edges[1])
    test_items = set(test_item)

    cold_items = [i for i in candidate_items if i not in train_items and i in test_items]

    if not cold_items:
        print("❌ 找不到符合條件的 cold item")
        return None

    selected = cold_items[0]
    print(f"🧊 Found cold item: {selected}")
    return selected

def evaluate_er_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    cold_item_set,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()

    all_target_items = set(range(data.num_target_items))
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    cold_item_hit_count = 0
    cold_item_ranks = []  # ⬅️ 儲存 cold item 被排進去時的排名

    with torch.no_grad():
        for user_id in sim_users:
            # 建立候選池
            negative_pool = list(all_target_items - cold_item_set)
            if len(negative_pool) < num_candidates:
                continue

            sampled_items = random.sample(negative_pool, num_candidates)
            sampled_items += list(cold_item_set)
            sampled_items = list(set(sampled_items))
            random.shuffle(sampled_items)

            user_tensor = torch.tensor([user_id] * len(sampled_items), device=device)
            item_tensor = torch.tensor(sampled_items, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            scores_list = scores.tolist()

            # 印出每個 item 的分數
            # print(f"\n=== [User {user_id}] ===")
            # for item, score in zip(sampled_items, scores_list):
            #     tag = "🧊 COLD" if item in cold_item_set else ""
            #     print(f"Item {item:4d} | Score: {score:.4f} {tag}")

            # 計算排序
            item_score_pairs = list(zip(sampled_items, scores_list))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_items = [item for item, _ in item_score_pairs]

            # 印出 cold item 的排名
            for cold_item in cold_item_set:
                if cold_item in sorted_items:
                    rank = sorted_items.index(cold_item) + 1
                    # print(f"🔍 Cold item {cold_item} ranked #{rank} / {len(sorted_items)}")

            top_k_items = sorted_items[:top_k]


            # ⬇️ 統計命中與排名
            cold_hits = [item for item in top_k_items if item in cold_item_set]
            if cold_hits:
                cold_item_hit_count += 1
                for cold_item in cold_hits:
                    rank = top_k_items.index(cold_item) + 1  # 1-based rank
                    cold_item_ranks.append(rank)

            total_users += 1

    er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
    avg_rank = sum(cold_item_ranks) / len(cold_item_ranks) if cold_item_ranks else -1
    median_rank = (
        sorted(cold_item_ranks)[len(cold_item_ranks) // 2] if cold_item_ranks else -1
    )

    logging.info(f"[ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")
    # logging.info(f"[ER@{top_k}] Cold item avg rank: {avg_rank:.2f}, median rank: {median_rank}")

    return er_ratio


def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, device):
    topk_list = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    print("\n📊 Evaluation for multiple top-K values:")
    for k in topk_list:
        hr = evaluate_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            top_k=k,
            num_candidates=99,
            device=device
        )

        er = evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            cold_item_set=cold_item_set,
            top_k=k,
            num_candidates=99,
            device=device
        )

def get_group_A_user_ids(edge_index, target_item_id):
    """
    從 target domain 的 edge_index ([2, num_edges]) 中找出買過 target_item_id 的 user ids
    """
    users = edge_index[0]
    items = edge_index[1]
    mask = items == target_item_id
    return users[mask].unique()


def find_hard_users(source_user_embs, group_A_emb_idx, top_ratio=0.1):
    """
    根據 source domain 用戶 embedding，找出 group B 中與 group A 距離最大 top_ratio 的 hard users。
    Args:
        source_user_embs: Tensor, shape [num_users, emb_dim]
        group_A_emb_idx: LongTensor，group A 用戶 embedding index
        top_ratio: float，取最大距離 top 多少比例
    Returns:
        hard_user_emb_idx: LongTensor，group B 中被挑選為 hard user 的 embedding index
    """
    num_users = source_user_embs.size(0)
    device = source_user_embs.device
    all_user_ids = torch.arange(num_users, device=device)

    # Group B 是所有用戶扣掉 Group A
    group_B_mask = ~torch.isin(all_user_ids, group_A_emb_idx.to(device))
    group_B_user_ids = all_user_ids[group_B_mask]  # embedding idx

    # L2 正規化 embedding
    group_A_embs = F.normalize(source_user_embs[group_A_emb_idx], p=2, dim=1)
    group_B_embs = F.normalize(source_user_embs[group_B_user_ids], p=2, dim=1)

    # 計算 cosine 相似度矩陣
    cosine_sim = torch.matmul(group_B_embs, group_A_embs.T)
    cosine_dist = 1 - cosine_sim  # 距離 = 1 - 相似度

    # 印出部分 cosine similarity 細節供檢視
    logging.info(f"Cosine similarity matrix shape: {cosine_sim.shape}")
    logging.info(f"Cosine similarity sample (first 5 Group B users vs first 5 Group A users):\n{cosine_sim[:5, :5]}")

    # 每個 Group B 用戶取距離最小值 (與最近的 Group A 用戶距離)
    min_dist_per_B_user, min_idx = torch.min(cosine_dist, dim=1)
    max_sim_per_B_user = cosine_sim[torch.arange(len(min_idx), device=device), min_idx] #這沒用
    logging.info(f"Sample minimal distances: {min_dist_per_B_user.tolist()}")
    logging.info(f"Sample maximal cosine similarities (closest user): {max_sim_per_B_user[:10].tolist()}") #這沒用

    # 閾值取最大 top_ratio 的距離
    threshold = torch.quantile(min_dist_per_B_user, 1 - top_ratio)
    hard_mask = min_dist_per_B_user >= threshold
    hard_user_emb_idx = group_B_user_ids[hard_mask]

    logging.info(f"Distance threshold for top {top_ratio*100}% hard users: {threshold:.4f}")
    logging.info(f"Total group B users: {len(group_B_user_ids)}, hard users count: {len(hard_user_emb_idx)}")
    logging.info(f"Sample hard user embedding idx: {hard_user_emb_idx[:10].tolist()}")

    return hard_user_emb_idx



def train(model, perceptor, data, args):
    device = args.device
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)

    (
        source_edge_index,
        source_label,
        source_link,
        target_train_edge_index,
        target_train_label,
        target_train_link,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
        target_test_edge_index,  # ✅ 新增這一項
    ) = link_split(data)
    data.target_test_link = target_test_link
    source_set_size = source_link.shape[1]
    train_set_size = target_train_link.shape[1]
    val_set_size = target_valid_link.shape[1]
    test_set_size = target_test_link.shape[1]
    logging.info(f"Train set size: {train_set_size}")
    logging.info(f"Valid set size: {val_set_size}")
    logging.info(f"Test set size: {test_set_size}")
    
    # ######################################
    

    # 1. 先從 target domain 邊取得所有原始用戶 ID（不重複）
    all_raw_user_ids = target_train_edge_index[0].unique().cpu()

    # 2. 取得 raw_overlap_users：模型embedding對應的用戶原始ID集合（可用於embedding索引）
    overlap_users = data.raw_overlap_users.cpu()  # Tensor，長度2809，例如
    overlap_users_set = set(overlap_users.numpy())

    # 3. 建立映射表：raw user id -> embedding 索引（0~2808）
    user_id_to_emb_idx = {uid.item(): idx for idx, uid in enumerate(overlap_users)}

    # 4. 找 Group A 原始用戶：買過目標商品的用戶 raw ID
    target_item_id = 3080
    mask_A = target_train_edge_index[1] == target_item_id
    group_A_raw_user_ids = target_train_edge_index[0][mask_A].cpu()
    print(f"Group A raw user IDs count: {len(group_A_raw_user_ids)}")
    print(f"Group A raw user IDs sample: {group_A_raw_user_ids[:20].tolist()}")  # 印前20個，可依需求調整


    # 5. 篩選 Group A，只保留在 overlap_users 內的有效 raw user id
    group_A_valid_mask = torch.tensor([uid.item() in overlap_users_set for uid in group_A_raw_user_ids])
    group_A_raw_user_ids_valid = group_A_raw_user_ids[group_A_valid_mask]

    # 6. 映射 Group A 原始用戶 ID 到 embedding 索引
    group_A_user_emb_idx = torch.tensor([user_id_to_emb_idx[uid.item()] for uid in group_A_raw_user_ids_valid])

    # 7. 對所有 raw 用戶，篩選只包含 overlap_users 的有效用戶，再映射成 embedding 索引
    valid_all_mask = torch.tensor([uid.item() in overlap_users_set for uid in all_raw_user_ids])
    valid_all_raw_user_ids = all_raw_user_ids[valid_all_mask]
    all_user_emb_idx = torch.tensor([user_id_to_emb_idx[uid.item()] for uid in valid_all_raw_user_ids])

    # 8. Group B 即所有有效用戶剔除 Group A，用 embedding 索引形式表示
    group_B_mask = ~torch.isin(all_user_emb_idx, group_A_user_emb_idx)
    group_B_user_emb_idx = all_user_emb_idx[group_B_mask]

    # 列印資訊確認
    logging.info(f"Group A user count: {len(group_A_user_emb_idx)}")
    logging.info(f"Group B user count: {len(group_B_user_emb_idx)}")
    # logging.info(f"Group A sample embedding idx: {group_A_user_emb_idx[:10].tolist()}")
    # logging.info(f"Group B sample embedding idx: {group_B_user_emb_idx[:10].tolist()}")

    # 9. 你可以用 group_A_user_emb_idx 和 group_B_user_emb_idx 這兩組embedding索引去索引 model.user_embedding.weight 做後續計算
    source_user_embs = model.user_embedding.weight  # shape [num_users, emb_dim]

    # 計算距離與找 hard user 等後續步驟...
    # 找出hard user
    hard_user_ids = find_hard_users(source_user_embs, group_A_user_emb_idx.to(source_user_embs.device), top_ratio=0.1)
    
    #emb_userID to org_userID
    emb_idx_to_user_id = {v: k for k, v in user_id_to_emb_idx.items()}
    hard_user_raw_ids = [emb_idx_to_user_id[idx.item()] for idx in hard_user_ids.cpu()]
    hard_user_raw_ids.sort()
    print(f"用戶原始ID（排序後）：{hard_user_raw_ids}")


    print(f"找到 hard user 數量: {len(hard_user_ids)}")
    #print(f"示例 hard user id: {hard_user_ids.tolist()}")#這是emb_userID
    
    ###################################
    
    # 確保 device 一致
    device = target_train_edge_index.device

    # 1. 將 hard user 原始ID轉 tensor
    hard_user_raw_ids_tensor = torch.tensor(hard_user_raw_ids, dtype=torch.long, device=device)

    # 2. 建立目標item id tensor (全填 target_item_id)
    target_item_ids_tensor = torch.full_like(hard_user_raw_ids_tensor, fill_value=target_item_id)

    # 3. 將用戶和商品合併成邊索引矩陣
    new_edges = torch.stack([hard_user_raw_ids_tensor, target_item_ids_tensor], dim=0)

    # 4. 合併新邊到 target train edge index
    target_train_edge_index = torch.cat([target_train_edge_index, new_edges], dim=1)

    # 5. 同步合併標籤 (全部1 = 正例)
    new_labels = torch.ones(hard_user_raw_ids_tensor.size(0), dtype=target_train_label.dtype, device=device)
    target_train_label = torch.cat([target_train_label, new_labels], dim=0)

    # 在這裡加上你要印的確認資訊
    print(f"原始train edge count: {target_train_edge_index.shape[1] - new_edges.shape[1]}")  # 新增前邊數63882
    print(f"新增後train edge count: {target_train_edge_index.shape[1]}")  # 新增後邊數64162
    
    ###################################

    # target_train_set = Dataset(
    #     target_train_link.to("cpu"),
    #     target_train_label.to("cpu"),
    # )
    # target_train_loader = DataLoader(
    #     target_train_set,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     collate_fn=target_train_set.collate_fn,
    # )
    target_train_set = Dataset(
        target_train_edge_index.cpu(),
        target_train_label.cpu(),
    )
    print(f"Dataset長度: {len(target_train_set)}")#32221

    target_train_loader = DataLoader(
        target_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )

    source_batch_size = int(args.batch_size * train_set_size / source_set_size)
    source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
    source_train_loader = DataLoader(
        source_train_set,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    target_meta_loader = DataLoader(
        target_train_set,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )
    target_meta_iter = iter(target_meta_loader)
    source_meta_batch_size = int(
        args.meta_batch_size * train_set_size / source_set_size
    )
    source_meta_loader = DataLoader(
        source_train_set,
        batch_size=source_meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    perceptor_optimizer = torch.optim.Adam(
        perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    meta_optimizer = MetaOptimizer(
        meta_optimizer=perceptor_optimizer,
        hpo_lr=args.hpo_lr,
        truncate_iter=3,
        max_grad_norm=10,
    )

    model_param = [
        param for name, param in model.named_parameters() if "preds" not in name
    ]
    replace_param = [
        param for name, param in model.named_parameters() if name.startswith("replace")
    ]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        replace_optimizer, T_max=args.T_max
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.model_path,
        trace_func=logging.info,
    )

    criterion = nn.BCELoss(reduction="none")
    iteration = 0
    for epoch in range(args.epochs):
        for (source_link, source_label), (target_link, target_label) in zip(
            source_train_loader, target_train_loader
        ):
            torch.cuda.empty_cache()
            source_link = source_link.to(device)
            source_label = source_label.to(device)
            target_link = target_link.to(device)
            target_label = target_label.to(device)
            weight_source = perceptor(source_link[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(
                source_edge_index, target_train_edge_index, source_link, is_source=True
            ).squeeze()
            target_out = model(
                source_edge_index, target_train_edge_index, target_link, is_source=False
            ).squeeze()
            source_loss = (
                criterion(source_out, source_label).reshape(-1, 1) * weight_source
            ).sum()
            target_loss = criterion(target_out, target_label).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1
            if (
                args.use_source
                and args.use_meta
                and iteration % args.meta_interval == 0
            ):
                logging.info(f"Entering meta optimization, iteration: {iteration}")
                meta_optimizeation(
                    target_meta_loader,
                    replace_optimizer,
                    model,
                    args,
                    criterion,
                    replace_scheduler,
                    source_edge_index,
                    target_train_edge_index,
                )

                try:
                    target_meta_link, target_meta_label = next(target_meta_iter)
                except StopIteration:
                    target_meta_iter = iter(target_meta_loader)
                    target_meta_link, target_meta_label = next(target_meta_iter)

                target_meta_link, target_meta_label = (
                    target_meta_link.to(device),
                    target_meta_label.to(device),
                )
                optimizer.zero_grad()
                target_out = model(
                    source_edge_index,
                    target_train_edge_index,
                    target_meta_link,
                    is_source=False,
                ).squeeze()
                meta_loss = criterion(target_out, target_meta_label).mean()

                for (source_link, source_label), (target_link, target_label) in zip(
                    source_meta_loader, target_meta_loader
                ):
                    source_link, source_label = source_link.to(device), source_label.to(
                        device
                    )
                    target_link, target_label = target_link.to(device), target_label.to(
                        device
                    )
                    weight_source = perceptor(source_link[1], source_edge_index, model)

                    optimizer.zero_grad()
                    source_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        source_link,
                        is_source=True,
                    ).squeeze()
                    target_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        target_link,
                        is_source=False,
                    ).squeeze()
                    source_loss = (
                        criterion(source_out, source_label).reshape(-1, 1)
                        * weight_source
                    ).sum()
                    target_loss = criterion(target_out, target_label).mean()
                    meta_train_loss = (
                        source_loss + target_loss if args.use_meta else target_loss
                    )
                    break

                torch.cuda.empty_cache()
                meta_optimizer.step(
                    train_loss=meta_train_loss,
                    val_loss=meta_loss,
                    aux_params=list(perceptor.parameters()),
                    parameters=model_param,
                    return_grads=True,
                    entropy=None,
                )
        train_auc = evaluate(
            "Train",
            model,
            source_edge_index,
            target_train_edge_index,
            target_train_link,
            target_train_label,
        )
        val_auc = evaluate(
            "Valid",
            model,
            source_edge_index,
            target_train_edge_index,
            target_valid_link,
            target_valid_label,
        )

        logging.info(
            f"[Epoch: {epoch}]Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
        )
        wandb.log(
            {
                "loss": loss,
                "train_auc": train_auc,
                "val_auc": val_auc
            },
            step=epoch,
        )

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        lr_scheduler.step()

    model = load_model(args).to(device)
    evaluate_hit_ratio(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,  # ✅ 正確傳入測試集 edge_index
        top_k=args.top_k,
        num_candidates=99,
        device=device,
    )
    #cold_item_id = find_cold_item_strict(data, target_train_edge_index, target_test_edge_index)
    cold_item_id = 3080
    if cold_item_id is not None:
        evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_train_edge_index,
            cold_item_set={cold_item_id},
            top_k=args.top_k,
            num_candidates=99,
            device=device,
        )

    # logging.info(f"Hit Ratio (no injection): {pre_hit_ratio:.4f}")
    test_auc = evaluate(
        "Test",
        model,
        source_edge_index,
        target_train_edge_index,
        target_test_link,
        target_test_label,
    )
    logging.info(f"Test AUC: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})
    # evaluate_multiple_topk(
    #     model=model,
    #     data=data,
    #     source_edge_index=source_edge_index,
    #     target_edge_index=target_train_edge_index,
    #     cold_item_set={cold_item_id},   # 注意這邊是 set，不是 cold_item_id=
    #     device=device
    # )

