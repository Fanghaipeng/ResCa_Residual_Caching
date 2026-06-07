import torch
import torch.nn.functional as F
from .Kmeans import Kmeans

def cluster_scheduler(cache_dic, current):
    return cache_dic['cluster_num'], cache_dic['k']

def _pairwise_cosine_similarity(X, eps=1e-6):
    X = F.normalize(X.float(), p=2, dim=-1, eps=eps)
    return torch.matmul(X, X.transpose(-1, -2))

def _update_temporal_similarity(X, cluster_info, alpha):
    similarity = _pairwise_cosine_similarity(X)
    temporal_similarity = cluster_info.get('temporal_similarity', None)
    if temporal_similarity is None or temporal_similarity.shape != similarity.shape:
        temporal_similarity = similarity
    else:
        temporal_similarity = alpha * similarity + (1.0 - alpha) * temporal_similarity.to(similarity.device)
    cluster_info['temporal_similarity'] = temporal_similarity.detach()
    return temporal_similarity

def _similarity_kmedoids(similarity, cluster_num, max_iters=8):
    B, N, _ = similarity.shape
    device = similarity.device
    cluster_num = min(cluster_num, N)
    initial = torch.linspace(0, N - 1, steps=cluster_num, device=device).round().long()
    medoids = initial.unsqueeze(0).expand(B, -1).clone()
    labels = torch.zeros((B, N), dtype=torch.long, device=device)

    for _ in range(max_iters):
        changed = False
        for batch in range(B):
            sim_b = similarity[batch]
            sim_to_medoids = sim_b[:, medoids[batch]]
            labels_b = sim_to_medoids.argmax(dim=-1)
            labels[batch] = labels_b

            next_medoids = medoids[batch].clone()
            for cluster_id in range(cluster_num):
                members = torch.where(labels_b == cluster_id)[0]
                if members.numel() == 0:
                    continue
                intra_sim = sim_b[members][:, members].sum(dim=-1)
                next_medoids[cluster_id] = members[intra_sim.argmax()]

            if not torch.equal(next_medoids, medoids[batch]):
                changed = True
                medoids[batch] = next_medoids
        if not changed:
            break

    for batch in range(B):
        labels[batch] = similarity[batch][:, medoids[batch]].argmax(dim=-1)
    return labels, medoids

def _sample_proxy_indices(similarity, cluster_indices, medoids, method, eps=1e-6):
    B, cluster_num = medoids.shape
    device = similarity.device
    proxy_indices = medoids.clone()

    if method == 'center':
        return proxy_indices
    if method not in ('random', 'center-random'):
        raise ValueError(f"Unknown ResCa proxy method: {method}")

    for batch in range(B):
        for cluster_id in range(cluster_num):
            members = torch.where(cluster_indices[batch] == cluster_id)[0]
            if members.numel() == 0:
                continue
            if method == 'random':
                choice = torch.randint(members.numel(), (1,), device=device)
                proxy_indices[batch, cluster_id] = members[choice].squeeze(0)
            else:
                center = medoids[batch, cluster_id]
                weights = similarity[batch, center, members].clamp(min=0.0) + eps
                proxy_indices[batch, cluster_id] = members[torch.multinomial(weights, 1)].squeeze(0)
    return proxy_indices

def get_cluster_info(X, cache_dic, current):
    cluster_num, k = cluster_scheduler(cache_dic, current)
    cluster_info = cache_dic['cluster_info']
    cluster_num = min(cluster_num, X.shape[1])

    if cache_dic.get('mode') == 'ResCa':
        temporal_similarity = _update_temporal_similarity(
            X,
            cluster_info,
            alpha=cache_dic.get('tet_alpha', 0.6),
        )
        cluster_indices, proxy_indices = _similarity_kmedoids(
            temporal_similarity,
            cluster_num=cluster_num,
            max_iters=cache_dic.get('tet_max_iters', 8),
        )
        proxy_indices = _sample_proxy_indices(
            temporal_similarity,
            cluster_indices,
            proxy_indices,
            method=cache_dic.get('resca_proxy_method', 'center'),
        )
        cluster_info['proxy_indices'] = proxy_indices
        cluster_info['centroids'] = None
    else:
        cache_centroids = cluster_info.get('centroids', None)
        init = 'kmeans++' if cache_dic.get('cluster_method') == 'kmeans++' else 'random'
        cluster_indices, cache_centroids = Kmeans(n_clusters=cluster_num, init=init).fit(X, cache_centroids)
        cluster_info['centroids'] = cache_centroids
        cluster_info['proxy_indices'] = None

    cache_dic['cluster_info']['cluster_num'] = cluster_num
    cache_dic['cluster_info']['k'] = k
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices

def construct_consecutive_cluster_info(X, cache_dic, current):
    '''
    construct consecutive cluster indices, every N//cluster_num tokens are grouped into one cluster
    '''
    cluster_num, k = cluster_scheduler(cache_dic, current)
    B, N, D = X.shape
    device = X.device
    segment_length = N // cluster_num
    cluster_indices = torch.arange(cluster_num, dtype=torch.long, device=device).repeat_interleave(segment_length)
    cluster_indices = cluster_indices.unsqueeze(0).expand(B, -1)
    cache_dic['cluster_info']['cluster_num'] = cluster_num
    cache_dic['cluster_info']['k'] = k
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    
def random_cluster_indices(X, cache_dic, current):
    '''
    randomly group the tokens into cluster_num groups(for ablation study)
    '''
    cluster_num, k = cluster_scheduler(cache_dic, current)
    B, N, D = X.shape
    device = X.device
    cluster_indices = torch.randint(0, cluster_num, (B, N), device=device)
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    cache_dic['cluster_info']['cluster_num'] = cluster_num
    cache_dic['cluster_info']['k'] = k
    
