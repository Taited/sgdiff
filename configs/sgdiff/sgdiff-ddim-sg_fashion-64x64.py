_base_ = [
    '../_base_/gen_default_runtime.py',
]

val_step = 5000

work_dir = './work_dirs/sgdiff'

# original glide model ckpt
# unet_ckpt_path = 'https://download.openmmlab.com/mmediting/glide/glide_laion-64x64-02afff47.pth'  # noqa
# fine-tuned glide model ckpt
unet_ckpt_path = 'glide-fine_tuned.pth'
# original glide upsampling ckpt
unet_up_ckpt_path = 'https://download.openxlab.org.cn/models/mmediting/GLIDE/weight/glide_laion-64-256'  # noqa

style_encoder_cfg = dict(
    type='ClipAttnEmbedding',
    name='ViT-B/32',
    cross_attn_cfg=dict(
        type='MultiHeadAttentionBlock',
        in_channels=512,
        num_heads=4,
        encoder_channels=512),
    residual_cfg=dict(learned_length=128))
model = dict(
    type='SGDiff',
    data_preprocessor=dict(type='DataPreprocessor', mean=[127.5], std=[127.5]),
    cond_prob={
        'txt': 0.2,
        'style': 0.2
    },
    unet=dict(
        type='MM2ImUNet',
        style_encoder_cfg=style_encoder_cfg,
        fix_glide=True,
        image_size=64,
        base_channels=192,
        in_channels=3,
        resblocks_per_downsample=3,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.1,
        num_classes=0,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=1,
            num_head_channels=64,
            use_new_attention_order=False,
            encoder_channels=512),
        use_scale_shift_norm=True,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
    ),
    diffusion_scheduler=dict(
        type='EditDDIMScheduler',
        variance_type='learned_range',
        beta_schedule='squaredcos_cap_v2'),
    unet_up=dict(
        type='SuperResText2ImUNet',
        image_size=256,
        base_channels=192,
        in_channels=3,
        output_cfg=dict(var='FIXED'),
        resblocks_per_downsample=2,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.1,
        num_classes=0,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=1,
            num_head_channels=64,
            use_new_attention_order=False,
            encoder_channels=512),
        use_scale_shift_norm=True,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
    ),
    diffusion_scheduler_up=dict(
        type='EditDDIMScheduler',
        variance_type='learned_range',
        beta_schedule='linear'),
    use_fp16=False)

optim_wrapper = dict(unet=dict(optimizer=dict(type='AdamW', lr=1e-5)))

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

train_dataloader = dict(batch_size=16, num_workers=4, pin_memory=True)
val_dataloader = dict(batch_size=8, num_workers=4, pin_memory=True)
test_dataloader = dict(batch_size=16, num_workers=4, pin_memory=True)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=200003,
    val_interval=val_step,
)

default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_step,
        by_epoch=False,
        out_dir=work_dir,
        max_keep_ckpts=3,
        save_optimizer=True))
