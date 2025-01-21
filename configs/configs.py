import ml_collections


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 64
    config.epochs = 300
    config.eval_epoch = 150
    config.mask_ratio = 0.75
    # config.start_TS_epoch = 100
    # config.masking_method = "random_masking"
    config.output_dir = "experiments/avenue"  # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L2'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)  # (360,640) --> (320, 640) --> (256, 448)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'  # train & inference
    config.resume = False
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-3
    # batch_size越大，momentum应该适当调小，以避免过度平滑更新。
    # batch_size较小，momentum可以相对较大，以帮助抵抗噪声并加速收敛。
    # for example use 0.9995 with batch size of 256.
    config.momentum_target = 0.99
    config.gamma = 1.0  # 两帧间重建损失权重

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "H:/AI/dataset/VAD/Featurize/Avenue"
    config.avenue_gt_path = "H:/AI/dataset/VAD/Featurize/Avenue/Avenue_gt"
    # config.percent_abnormal = 0.25
    # config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 0
    config.pin_mem = True

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 64
    config.epochs = 200
    config.eval_epoch = 175
    config.mask_ratio = 0.75
    # config.start_TS_epoch = 100
    # config.masking_method = "random_masking"
    config.output_dir = "experiments/shanghai"  # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L2'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "train"
    config.resume = True

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4
    config.momentum_target = 0.996

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "H:/AI/dataset/VAD/Featurize/ShanghaiTech"
    config.shanghai_gt_path = "H:/AI/dataset/VAD/Featurize/ShanghaiTech/Shanghai_gt"
    # config.percent_abnormal = 0.25
    # config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 8
    config.pin_mem = False

    return config
