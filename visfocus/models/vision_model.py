import torch

from . import SwinTransformerV2


def get_vision_model(config):
    if config.model.type == "swinv2":
        vision_model = SwinTransformerV2(
            img_size=config.model.image_size,
            patch_size=config.model.swinv2.patch_size,
            in_chans=config.model.swinv2.in_chans,
            embed_dim=config.model.swinv2.embed_dim,
            depths=config.model.swinv2.depths,
            num_heads=config.model.swinv2.num_heads,
            window_size=config.model.swinv2.window_size,
            mlp_ratio=config.model.swinv2.mlp_ratio,
            qkv_bias=config.model.swinv2.qkv_bias,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            ape=config.model.swinv2.ape,
            patch_norm=config.model.swinv2.patch_norm,
            use_checkpoint=config.train.use_checkpoint,
            pretrained_window_sizes=config.model.swinv2.pretrained_window_sizes,
            do_shift=getattr(config.model.swinv2, 'do_shift', True),
            downsampling_method=getattr(config.model.swinv2, 'downsampling_method', 'patch_merging'),
            vl_cross_attn_layers=getattr(config.model.swinv2, 'vl_cross_attn_layers', []),
            vl_alpha=getattr(config.model.swinv2, 'vl_alpha', 0.5),
            lm_d_model=getattr(config.model.swinv2, 'lm_d_model', 512),
            cross_attention_cls_key=getattr(config.model.swinv2, 'cross_attention_cls_key', 'cross_attention'),
            input_type=getattr(config.model.swinv2, 'input_type', 'rgb'),
            vl_learned_ape=getattr(config.model.swinv2, 'vl_learned_ape', True),
            vl_self_attn_layers=getattr(config.model.swinv2, 'vl_self_attn_layers', []),
        )
        return vision_model
    else:
        raise NotImplementedError("Unsupport model type %s" % config.model.type)


def load_vision_pretrained(configs, model, logger):
    logger.info("Loading vision model from %s", configs.model.vision_resume_from)
    if configs.model.vision_resume_from.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            configs.model.vision_resume_from, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(configs.model.vision_resume_from, map_location="cpu")
    
    state_dict = checkpoint["model"]

    if "swin" in configs.model.type:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = model.vision_model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = state_dict[k]
            absolute_pos_embed_current = model.vision_model.state_dict()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                    state_dict[k] = absolute_pos_embed_pretrained_resized
        
        if model.vision_model.patch_embed.proj.weight.shape != state_dict['patch_embed.proj.weight'].shape:
            model.vision_model.input_type == 'flattened_patches'
            logger.warning(f"PatchEmbed (patch_embed) was not loaded, because input_type is falttened_patches.")
            del state_dict['patch_embed.proj.weight']


    # import pdb;pdb.set_trace()
    msg = model.vision_model.load_state_dict(state_dict, strict=False)

    # do not print unnecessary (vl attn is not loaded now)
    filtered_missing_keys = {k for k in msg.missing_keys
                             if 'vl_cross_attn_layers' not in k
                             or 'relative_position' not in k}
    filtered_missing_keys.union({'relative_position' for k in msg.missing_keys
                             if 'relative_position' not in k})
    # if len({k for k in msg.missing_keys if 'relative_' in k}) > 0:
    #     logger.warning(f'Relative position were not loaded')
    # filtered_missing_keys.union()
    logger.warning(f'Missing keys: {set(msg.missing_keys) - filtered_missing_keys}')
    logger.warning(f'Unexpected keys: {msg.unexpected_keys}')
    
    # logger.warning(msg)

    logger.info("Loaded model successfully from %s", configs.model.vision_resume_from)

    del checkpoint
    torch.cuda.empty_cache()
