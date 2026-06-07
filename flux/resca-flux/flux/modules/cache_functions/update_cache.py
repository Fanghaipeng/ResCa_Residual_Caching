import torch
import torch.nn.functional as F
import math
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    # Update the cached tokens at the positions


    indices = fresh_indices

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
    
def propagation_update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    step = current['step']
    layer = current['layer']
    module = current['module']

    fresh_tokens = fresh_tokens.to(torch.bfloat16)
    # cluster_info = cache_dic['cluster_info']
    cluster_info = cache_dic['cluster_info'][current['stream']][current['module']]
    cluster_indices, cluster_num, k = \
        cluster_info['cluster_indices'], cluster_info['cluster_num'], cluster_info['k']
    propagation_ratio = cache_dic['propagation_ratio']
    dim = fresh_tokens.shape[-1]
    # cache_dic['cache'][-1][current['stream']][layer][module][0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    # old_cache = cache_dic['cache'][-1][current['stream']][layer][module][0]
    # B, N, dim = old_cache.shape
    # device = old_cache.device
    # fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)
    # sum_per_cluster = torch.zeros((B, cluster_num, dim), device=device, dtype=torch.bfloat16)
    # sum_per_cluster.scatter_add_(
    #     dim=1,
    #     index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
    #     src=fresh_tokens
    # )
    # mean_per_cluster = sum_per_cluster                                                       # only when topk == 1
    # # mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)   # when topk > 1

    # new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))
    # cache_dic['cache'][-1][current['stream']][layer][module][0] = new_cache * propagation_ratio + old_cache * (1 - propagation_ratio)
    propagation_update_cache_compile(old_cache_dict=cache_dic['cache'][-1][current['stream']][layer][module],
                                fresh_indices=fresh_indices,
                                fresh_tokens=fresh_tokens,
                                cluster_indices=cluster_indices,
                                cluster_num=cluster_num,
                                k=k,
                                propagation_ratio=propagation_ratio)

@torch.compile
def propagation_update_cache_compile(old_cache_dict, fresh_indices, fresh_tokens, cluster_indices, cluster_num, k, propagation_ratio):
    B, N, dim = old_cache_dict[0].shape
    device = old_cache_dict[0].device
    old_cache_dict[0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)
    sum_per_cluster = torch.zeros((B, cluster_num, dim), device=device, dtype=torch.bfloat16)
    sum_per_cluster.scatter_add_(
        dim=1,
        index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
        src=fresh_tokens
        )
    if k == 1:
        mean_per_cluster = sum_per_cluster                                                       # only when topk == 1
    elif k > 1: # we found that when k == 1, we can already obtain effective results. So this branch is not used in final version for efficiency.
        count_per_cluster = torch.zeros((B, cluster_num), device=device)
        count_per_cluster.scatter_add_(
            dim=1,
            index=fresh_cluster_indices,
            src=torch.ones_like(fresh_indices, dtype=torch.float, device=device)
        )
        mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)     # for k > 1

    new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))
    old_cache_dict[0] = new_cache * propagation_ratio + old_cache_dict[0] * (1 - propagation_ratio)

def _resca_residual_orders(fresh_indices, fresh_tokens, cache_dic, current, max_order=1, eps=1e-6):
    """
    Estimate proxy-driven residual orders for all tokens.
    """
    layer = current['layer']
    module = current['module']
    stream = current['stream']

    cluster_info = cache_dic['cluster_info'][stream][module]
    cluster_indices = cluster_info['cluster_indices']
    k = cluster_info['k']
    if k != 1:
        raise ValueError("Minimal ResCa v0 expects exactly one proxy token per cluster (k=1).")

    old_cache_dict = cache_dic['cache'][-1][stream][layer][module]
    x_t = old_cache_dict[0]

    fresh_tokens = fresh_tokens.to(dtype=x_t.dtype)
    B, N, dim = x_t.shape
    proxy_index = fresh_indices.unsqueeze(-1).expand(-1, -1, dim)

    cached_orders = {}
    for order in range(max_order + 1):
        cached_orders[order] = old_cache_dict.get(order)
        if cached_orders[order] is None:
            cached_orders[order] = torch.zeros_like(x_t)

    proxy_orders_t = {
        order: cached_orders[order].gather(dim=1, index=proxy_index)
        for order in range(max_order + 1)
    }
    proxy_orders_next = {0: fresh_tokens}
    for order in range(1, max_order + 1):
        proxy_orders_next[order] = proxy_orders_next[order - 1] - proxy_orders_t[order - 1]

    cluster_index = cluster_indices.unsqueeze(-1).expand(-1, -1, dim)
    estimated_orders = {}
    for order in range(1, max_order + 1):
        residual_t = cached_orders[order]
        proxy_residual_t_for_token = proxy_orders_t[order].gather(dim=1, index=cluster_index)
        proxy_residual_next_for_token = proxy_orders_next[order].gather(dim=1, index=cluster_index)

        theta = F.cosine_similarity(residual_t.float(), proxy_residual_t_for_token.float(), dim=-1, eps=eps)
        theta = torch.clamp(theta, min=0.0).to(dtype=x_t.dtype).unsqueeze(-1)
        estimated_orders[order] = (1.0 - theta) * residual_t + theta * proxy_residual_next_for_token

    return x_t, cached_orders, estimated_orders, proxy_index, fresh_tokens

def _resca_context(fresh_indices, fresh_tokens, cache_dic, current, max_order=1):
    layer = current['layer']
    module = current['module']
    stream = current['stream']

    cluster_info = cache_dic['cluster_info'][stream][module]
    cluster_indices = cluster_info['cluster_indices']
    k = cluster_info['k']
    if k != 1:
        raise ValueError("Minimal ResCa v0 expects exactly one proxy token per cluster (k=1).")

    old_cache_dict = cache_dic['cache'][-1][stream][layer][module]
    x_t = old_cache_dict[0]
    fresh_tokens = fresh_tokens.to(dtype=x_t.dtype)
    B, N, dim = x_t.shape
    proxy_index = fresh_indices.unsqueeze(-1).expand(-1, -1, dim)
    cluster_index = cluster_indices.unsqueeze(-1).expand(-1, -1, dim)

    cached_orders = {}
    proxy_orders_t = {}
    for order in range(max_order + 1):
        cached_orders[order] = old_cache_dict.get(order)
        if cached_orders[order] is None:
            cached_orders[order] = torch.zeros_like(x_t)
        proxy_orders_t[order] = cached_orders[order].gather(dim=1, index=proxy_index)

    return x_t, cached_orders, proxy_orders_t, proxy_index, cluster_index, fresh_tokens

def _resca_estimate_order(residual_t, proxy_residual_t, proxy_residual_next, cluster_index, x_dtype, eps=1e-6):
    proxy_residual_t_for_token = proxy_residual_t.gather(dim=1, index=cluster_index)
    proxy_residual_next_for_token = proxy_residual_next.gather(dim=1, index=cluster_index)
    theta = F.cosine_similarity(residual_t.float(), proxy_residual_t_for_token.float(), dim=-1, eps=eps)
    theta = torch.clamp(theta, min=0.0).to(dtype=x_dtype).unsqueeze(-1)
    return (1.0 - theta) * residual_t + theta * proxy_residual_next_for_token

def _resca_implicit_taylor_tokens(fresh_indices, fresh_tokens, cache_dic, current, max_order=1, eps=1e-6):
    x_t, cached_orders, proxy_orders_t, proxy_index, cluster_index, fresh_tokens = _resca_context(
        fresh_indices=fresh_indices,
        fresh_tokens=fresh_tokens,
        cache_dic=cache_dic,
        current=current,
        max_order=max_order,
    )

    x_next = x_t.clone()
    proxy_order_next = fresh_tokens
    for order in range(1, max_order + 1):
        proxy_order_next = proxy_order_next - proxy_orders_t[order - 1]
        estimated_order = _resca_estimate_order(
            residual_t=cached_orders[order],
            proxy_residual_t=proxy_orders_t[order],
            proxy_residual_next=proxy_order_next,
            cluster_index=cluster_index,
            x_dtype=x_t.dtype,
            eps=eps,
        )
        x_next.add_(estimated_order, alpha=1.0 / math.factorial(order))

    return x_next, proxy_index, fresh_tokens

def _resca_first_order_residual(fresh_indices, fresh_tokens, cache_dic, current, eps=1e-6):
    """
    Estimate F^(1)(d_{t-1}) with proxy-driven denoising simulation.
    """
    x_t, cached_orders, estimated_orders, proxy_index, fresh_tokens = _resca_residual_orders(
        fresh_indices=fresh_indices,
        fresh_tokens=fresh_tokens,
        cache_dic=cache_dic,
        current=current,
        max_order=1,
        eps=eps,
    )
    return x_t, cached_orders[1], estimated_orders[1], proxy_index, fresh_tokens

def resca_simulate_tokens(fresh_indices, fresh_tokens, cache_dic, current, eps=1e-6):
    """
    Simulate all tokens in a ResCa sparse step.

    `ie` uses the implicit-Euler first-order reduction from the paper. `it`
    uses the unit-step implicit Taylor form up to `max_order`. `bdf2` uses the
    paper's first-order BDF2 form with an approximate d_{t+1}=d_t-F^(1)(d_t)
    recovered from the cached first residual.
    """
    solver = cache_dic.get('resca_solver', 'ie')
    max_order = cache_dic.get('max_order', 1)
    if solver not in ('ie', 'it', 'bdf2'):
        raise ValueError(f"Unknown ResCa solver: {solver}")

    if solver == 'it':
        order = max(1, max_order)
        x_next, proxy_index, fresh_tokens = _resca_implicit_taylor_tokens(
            fresh_indices=fresh_indices,
            fresh_tokens=fresh_tokens,
            cache_dic=cache_dic,
            current=current,
            max_order=order,
            eps=eps,
        )
        output_dtype = x_next.dtype
    elif solver == 'bdf2':
        x_t, residual_t, estimated_residual, proxy_index, fresh_tokens = _resca_first_order_residual(
            fresh_indices=fresh_indices,
            fresh_tokens=fresh_tokens,
            cache_dic=cache_dic,
            current=current,
            eps=eps,
        )
        x_prev = x_t - residual_t
        x_next = (4.0 / 3.0) * x_t - (1.0 / 3.0) * x_prev + (2.0 / 3.0) * estimated_residual
        output_dtype = x_t.dtype
    else:
        x_t, residual_t, estimated_residual, proxy_index, fresh_tokens = _resca_first_order_residual(
            fresh_indices=fresh_indices,
            fresh_tokens=fresh_tokens,
            cache_dic=cache_dic,
            current=current,
            eps=eps,
        )
        x_next = x_t + estimated_residual
        output_dtype = x_t.dtype

    x_next = x_next.to(dtype=output_dtype)
    x_next.scatter_(dim=1, index=proxy_index, src=fresh_tokens)
    return x_next

def resca_update_cache(fresh_indices, fresh_tokens, cache_dic, current, eps=1e-6):
    """
    Backward-compatible alias for the default ResCa sparse-step simulator.
    """
    return resca_simulate_tokens(
        fresh_indices=fresh_indices,
        fresh_tokens=fresh_tokens,
        cache_dic=cache_dic,
        current=current,
        eps=eps,
    )
