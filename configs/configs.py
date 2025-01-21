import ml_collections


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 32
    config.epochs = 200
    config.mask_ratio = 0.95
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "experiments/avenue"  # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L2'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'  # train & inference
    config.resume = False
    # Optimizer parameters
    config.weight_decay = 0.05  # 0.05
    config.lr = 1e-4  # 1e-4

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "H:/AI/dataset/VAD/Featurize/Avenue"
    config.avenue_gt_path = "H:/AI/dataset/VAD/Featurize/Avenue/Avenue_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"
    config.previous_nums = 10

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 0
    config.pin_mem = False

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 64
    config.epochs = 140
    config.mask_ratio = 0.95
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "experiments/shanghai"  # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L1'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "inference"
    config.resume=False

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "H:/AI/dataset/VAD/ShanghaiTech"
    config.shanghai_gt_path = "H:/AI/dataset/VAD/ShanghaiTech/Shanghai_gt"
    config.percent_abnormal = 0
    config.input_3d = True
    config.device = "cuda"
    config.previous_nums = 12

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 4
    config.pin_mem = False

    return config
