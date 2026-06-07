def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}

    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0
    cache_dic['mode']                 = model_kwargs.get('mode', 'ResCa')

    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [num_steps - 1]

    cache_dic['enable_resca']         = cache_dic['mode'] == 'ResCa' and model_kwargs.get('cluster_num', 0) > 0
    cache_dic['enable_clusca']        = cache_dic['mode'] == 'ClusCa' and model_kwargs.get('cluster_num', 0) > 0

    # ClusCa/ResCa parameters
    if cache_dic['enable_clusca'] or cache_dic['enable_resca']:
        cache_dic['cluster_num']              = model_kwargs['cluster_num']
        cache_dic['k']                        = model_kwargs['k']
        cache_dic['propagation_ratio']        = model_kwargs['propagation_ratio']
        cache_dic['cluster_method']           = model_kwargs.get('cluster_method', 'kmeans')
        cache_dic['tet_alpha']                = model_kwargs.get('tet_alpha', 0.6)
        cache_dic['tet_max_iters']            = model_kwargs.get('tet_max_iters', 8)
        cache_dic['resca_solver']             = model_kwargs.get('resca_solver', 'ie')
        cache_dic['resca_proxy_method']       = model_kwargs.get('resca_proxy_method', 'center')

        cache_dic['cluster_info'] = {
            'cluster_indices': None,
            'centroids': None,
            'proxy_indices': None,
            'temporal_similarity': None,
            'cluster_num': cache_dic['cluster_num'],
            'k': cache_dic['k'],
        }
    else:
        cache_dic['cluster_num'] = 0

    return cache_dic, current
    
