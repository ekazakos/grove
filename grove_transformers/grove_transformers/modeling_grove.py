import torch
import torch.nn as nn
from typing import List
from .configuration_grove import GroveConfig
from .model.SAM import build_sam_vit_h
from .model.llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from .model.llava.model.multimodal_encoder.builder import build_vision_tower
from types import SimpleNamespace
from transformers import CLIPVisionConfig
from .tokenization_grove import GroveTokenizer


class GroveTextProjection(nn.Sequential):
    def __init__(self, config: GroveConfig):
        in_dim, out_dim = config.hidden_size, config.out_dim
        super().__init__(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        )


class GroveBaseModel(LlavaLlamaModel):
    def __init__(self, config: GroveConfig, vision_pretrained: str | None = None):
        if not hasattr(config, 'mm_hidden_size'):
            try:
                clip_cfg = CLIPVisionConfig.from_pretrained(config.mm_vision_tower)
                config.mm_hidden_size = clip_cfg.hidden_size
            except Exception:
                config.mm_hidden_size = 1024
        super().__init__(config)
        model_args = SimpleNamespace(
            vision_tower=config.mm_vision_tower,
            mm_vision_select_layer=config.mm_vision_select_layer,
            mm_vision_select_feature=config.mm_vision_select_feature,
            with_region=config.with_region,
            pretrain_mm_mlp_adapter=None,
            tune_mm_mlp_adapter=False,
            freeze_mm_mlp_adapter=True,
        )
        self.initialize_vision_modules(model_args)
        self.grounding_encoder = build_sam_vit_h(vision_pretrained, use_temp_objectness=config.use_temp_objectness)
        self.text_hidden_fcs = nn.ModuleList([GroveTextProjection(config)])
        self._configure_settings()

    def _configure_settings(self):
        cfg = self.config
        cfg.use_cache = False
        cfg.vision_module = getattr(cfg, 'mm_vision_tower', None)
        cfg.mm_vision_module = getattr(cfg, 'mm_vision_tower', None)
        cfg.select_feature_type = 'patch'
        cfg.image_aspect = 'square'
        cfg.image_grid_points = None
        cfg.tune_mlp_adapter = False
        cfg.freeze_mlp_adapter = True
        cfg.pretrain_mm_mlp_adapter = None
        cfg.use_image_patch_token = False


class GroveForCausalLM(LlavaLlamaForCausalLM):
    config_class = GroveConfig

    def __init__(self, config: GroveConfig, **kwargs):
        vision_pretrained = kwargs.get('vision_pretrained', None)

        if 'with_region' in kwargs:
            config.with_region = kwargs['with_region']
        if 'num_level_reg_features' in kwargs:
            config.num_level_reg_features = kwargs['num_level_reg_features']
        if not hasattr(config, 'num_reg_features'):
            config.num_reg_features = getattr(config, 'num_level_reg_features', 4)
        if 'temp_objectness_threshold' in kwargs:
            config.temp_objectness_threshold = kwargs['temp_objectness_threshold']
        if 'use_temp_objectness' in kwargs:
            config.use_temp_objectness = kwargs['use_temp_objectness']
        if 'num_frames' in kwargs:
            config.num_frames = kwargs['num_frames']
        if 'mm_use_image_start_end' in kwargs:
            config.mm_use_image_start_end = kwargs['mm_use_image_start_end']
        if 'mm_vision_module' in kwargs:
            config.mm_vision_module = kwargs['mm_vision_module']
        if 'mm_vision_select_layer' in kwargs:
            config.mm_vision_select_layer = kwargs['mm_vision_select_layer']
        if 'pretrain_mm_mlp_adapter' in kwargs:
            config.pretrain_mm_mlp_adapter = kwargs['pretrain_mm_mlp_adapter']
        if 'tune_mm_mlp_adapter' in kwargs:
            config.tune_mm_mlp_adapter = kwargs['tune_mm_mlp_adapter']
        if 'freeze_mm_mlp_adapter' in kwargs:
            config.freeze_mm_mlp_adapter = kwargs['freeze_mm_mlp_adapter']
        if 'train_mask_decoder' in kwargs:
            config.train_mask_decoder = kwargs['train_mask_decoder']
        if 'out_dim' in kwargs:
            config.out_dim = kwargs['out_dim']

        ce_loss_weight = kwargs.pop('ce_loss_weight', getattr(config, 'ce_loss_weight', 1.0))
        giou_loss_weight = kwargs.pop('giou_loss_weight', getattr(config, 'giou_loss_weight', 1.0))
        temp_objectness_loss_weight = kwargs.pop('temp_objectness_loss_weight', getattr(config, 'temp_objectness_loss_weight', 1.0))
        if not hasattr(config, 'mm_hidden_size'):
            try:
                clip_cfg = CLIPVisionConfig.from_pretrained(getattr(config, 'mm_vision_tower', ''))
                config.mm_hidden_size = clip_cfg.hidden_size
            except Exception:
                config.mm_hidden_size = getattr(config, 'hidden_size', 1024)
        super().__init__(config)
        self.model = GroveBaseModel(config, vision_pretrained=vision_pretrained)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.det_token_idx = None
        self.ce_loss_weight = ce_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.temp_objectness_loss_weight = temp_objectness_loss_weight
        self.post_init()

    def apply_tokenizer_special_ids(self, tokenizer):
        """Resolve and attach all GROVE special token IDs from the tokenizer.

        Must be called exactly once after tokenizer.add_grove_tokens().
        """
        mapping = getattr(tokenizer, 'grove_special_token_ids', None)
        if mapping is None:
            raise ValueError("Tokenizer lacks grove_special_token_ids. Ensure add_grove_tokens() was invoked.")
        self.det_token_idx = mapping.get('det_token_idx')
        for k in ('bbox_token_idx', 'bop_token_idx', 'eop_token_idx', 'pad_token_id', 'bos_token_id', 'eos_token_id'):
            v = mapping.get(k)
            if v is not None:
                setattr(self.config, k, v)
        return mapping

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """Load model and auto-inject tokenizer special ids.

        Args:
            pretrained_model_name_or_path: path or hub id.
            **kwargs: forwarded to base from_pretrained (e.g., device_map, torch_dtype).

        Returns:
            GroveForCausalLM instance with det_token_idx set if tokenizer supplied.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Auto-load tokenizer
        use_mm = getattr(model.config, 'mm_use_image_start_end', True)
        tokenizer = GroveTokenizer.from_pretrained(pretrained_model_name_or_path, use_mm_start_end=use_mm)
        model.apply_tokenizer_special_ids(tokenizer)

        return model

    def get_grounding_encoder_embs(self, images: torch.FloatTensor):
        return self.model.grounding_encoder.image_encoder(images)

    def get_global_encoder_features(self, images: torch.FloatTensor):
        """Return (image_features, image_forward_outs) from vision tower.
        """
        return self.encode_images(images)

    def forward(self, **kwargs):
        if 'past_key_values' in kwargs:
            return super().forward(**kwargs)
        mode = kwargs.get('mode', None)
        if mode == 'encode_images':
            return self.encode_images(kwargs['images'])
        if mode == 'get_grounding_encoder_embs':
            return self.get_grounding_encoder_embs(kwargs['images'])
        if mode == 'get_dense_pe':
            return self.model.grounding_encoder.prompt_encoder.get_dense_pe()
        if mode == 'evaluate':
            return self.evaluate(
                kwargs['image_features'], kwargs['image_forward_outs'], kwargs['images_dtype'],
                kwargs['image_embeddings'], kwargs['input_ids'], kwargs['original_size_list'],
                max_tokens_new=kwargs['max_tokens_new'], bboxes=kwargs.get('bboxes'),
                token_embeddings=kwargs.get('token_embeddings'), dense_pe=kwargs.get('dense_pe'),
                device=kwargs.get('device')
            )
        return self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes_region: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, bboxes_list: List[torch.FloatTensor],
                      temp_objectness_labels_list: List[torch.FloatTensor], original_size_list: List[torch.Tensor],
                      inference: bool = False, **kwargs):
        image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
        det_token_mask = self._create_det_token_mask(input_ids)
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images)
        else:
            output, output_hidden_states = self._training_path(global_enc_images, bboxes_region, input_ids, labels, attention_masks, offset)
        hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, det_token_mask, offset)
        dense_pe = self.model.grounding_encoder.prompt_encoder.get_dense_pe()
        if self.config.use_temp_objectness:
            pred_bboxes, logits_temp_objectness = self._generate_and_postprocess_masks(pred_embeddings, image_embeddings, original_size_list, dense_pe, infer=inference)
        else:
            pred_bboxes = self._generate_and_postprocess_masks(pred_embeddings, image_embeddings, original_size_list, dense_pe, infer=inference)
            logits_temp_objectness = None
        if inference:
            return {'pred_bboxes': pred_bboxes, 'logits_temp_objectness': logits_temp_objectness}
        return self._calculate_losses_video(pred_bboxes, logits_temp_objectness, bboxes_list, temp_objectness_labels_list, output)

    def _create_det_token_mask(self, input_ids):
        if self.det_token_idx is None:
            raise RuntimeError("det_token_idx not set. Call apply_tokenizer_special_ids(tokenizer) before forward().")
        mask = input_ids[:, 1:] == self.det_token_idx
        return torch.cat([torch.zeros((mask.shape[0], 575)).bool().to(input_ids.device), mask, torch.zeros((mask.shape[0], 1)).bool().to(input_ids.device)], dim=1)

    def _inference_path(self, input_ids, global_enc_images):
        output = super().forward(images=global_enc_images, input_ids=input_ids, output_hidden_states=True)
        output_hidden_states = output.hidden_states
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, bboxes_region, input_ids, labels, attention_masks, offset):
        bboxes_region_list = bboxes_region
        output = super().forward(images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels, output_hidden_states=True, bboxes=bboxes_region_list)
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _process_hidden_states(self, output_hidden_states, det_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        last_hidden_state = last_hidden_state.repeat_interleave(self.config.num_frames, dim=0)
        det_token_mask = det_token_mask.repeat_interleave(self.config.num_frames, dim=0)
        pred_embeddings = last_hidden_state[det_token_mask]
        det_token_counts = det_token_mask.int().sum(-1)
        det_token_offset = det_token_counts.cumsum(-1)
        det_token_offset = torch.cat([torch.zeros(1).long().to(det_token_mask.device), det_token_offset], dim=0)
        pred_embeddings_list = []
        for i in range(len(det_token_offset) - 1):
            start_i, end_i = det_token_offset[i], det_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, orig_sizes, dense_pe, infer=False):
        bs = len(pred_embeddings)
        num_masks_per_embed = [embed.shape[0] for embed in pred_embeddings]
        pred_embeddings_cat = torch.cat(pred_embeddings, dim=0).unsqueeze(1)
        sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=pred_embeddings_cat)
        sparse_embeddings = sparse_embeddings.to(pred_embeddings_cat.dtype)
        if self.config.use_temp_objectness:
            bbox_preds, temp_objectness_logits = self.model.grounding_encoder.mask_decoder(image_embeddings=image_embeddings, image_pe=dense_pe, sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False, reps=num_masks_per_embed)
        else:
            bbox_preds = self.model.grounding_encoder.mask_decoder(image_embeddings=image_embeddings, image_pe=dense_pe, sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False, reps=num_masks_per_embed)
            temp_objectness_logits = None
        start_idx = 0
        bbox_pred_list = []
        temp_objectness_logits_list = []
        for i in range(0, bs, self.config.num_frames):
            bbox_pred_list_sample = []
            if self.config.use_temp_objectness:
                temp_objectness_logits_list_sample = []
            for j in range(self.config.num_frames):
                if infer:
                    orig_size = orig_sizes[i // self.config.num_frames]
                    from grove_transformers.utils.bbox_utils import box_cxcywh_to_xyxy, unnormalize_bboxes
                    unnormalized_bboxes = unnormalize_bboxes(bbox_preds[start_idx:start_idx + num_masks_per_embed[i + j]], orig_size[0], orig_size[1], type='pt')
                    unnormalized_bboxes = box_cxcywh_to_xyxy(unnormalized_bboxes, type='pt')
                    if self.config.use_temp_objectness:
                        temp_objectness_preds = torch.sigmoid(temp_objectness_logits[start_idx:start_idx + num_masks_per_embed[i + j]])
                        temp_objectness_preds = temp_objectness_preds > self.config.temp_objectness_threshold
                        bbox_pred_list_sample.append(unnormalized_bboxes[temp_objectness_preds.bool()])
                        temp_objectness_logits_list_sample.append(temp_objectness_logits[start_idx:start_idx + num_masks_per_embed[i + j]])
                    else:
                        bbox_pred_list_sample.append(unnormalized_bboxes)
                    start_idx += num_masks_per_embed[i + j]
                else:
                    bbox_pred_list_sample.append(bbox_preds[start_idx:start_idx + num_masks_per_embed[i + j]])
                    if self.config.use_temp_objectness:
                        temp_objectness_logits_list_sample.append(temp_objectness_logits[start_idx:start_idx + num_masks_per_embed[i + j]])
                    start_idx += num_masks_per_embed[i + j]
            bbox_pred_list.append(bbox_pred_list_sample)
            if self.config.use_temp_objectness:
                temp_objectness_logits_list.append(temp_objectness_logits_list_sample)
        if self.config.use_temp_objectness:
            return bbox_pred_list, temp_objectness_logits_list
        return bbox_pred_list

    def _calculate_losses_video(self, pred_bboxes, logits_temp_objectness, gt_bboxes_list, gt_temp_objectness_list, output):
        return self._compute_loss_components_video(pred_bboxes, logits_temp_objectness, gt_bboxes_list, gt_temp_objectness_list, output)

    def _compute_loss_components_video(self, pred_bboxes, logits_temp_objectness, gt_bboxes_list, gt_temp_objectness_list, output):
        ce_loss = output.loss * self.ce_loss_weight
        giou_loss = torch.tensor(0.0, device=ce_loss.device)
        l1_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.config.use_temp_objectness and logits_temp_objectness is not None:
            temporal_objectness_loss = torch.tensor(0.0, device=ce_loss.device)
        num_bboxes = 0
        num_max_bboxes = 0
        if self.config.use_temp_objectness and logits_temp_objectness is not None:
            for batch_idx, (pred_bboxes_video, logits_temp_objectness_video) in enumerate(zip(pred_bboxes, logits_temp_objectness)):
                for frame_idx, (pred_bboxes_frame, logits_temp_objectness_frame) in enumerate(zip(pred_bboxes_video, logits_temp_objectness_video)):
                    gt_bboxes_frame = gt_bboxes_list[batch_idx][frame_idx].to(pred_bboxes_frame.device)
                    gt_temp_objectness_frame = gt_temp_objectness_list[batch_idx][frame_idx].to(logits_temp_objectness_frame.device)
                    assert gt_bboxes_frame.shape[0] == gt_temp_objectness_frame.sum()
                    if gt_bboxes_frame.shape[0] != 0:
                        from utils.bbox_utils import box_cxcywh_to_xyxy
                        giou_loss += torch.ops.torchvision.generalized_box_iou_loss(
                            box_cxcywh_to_xyxy(pred_bboxes_frame[gt_temp_objectness_frame.bool()], type='pt'),
                            box_cxcywh_to_xyxy(gt_bboxes_frame, type='pt'), reduction='sum'
                        )
                        l1_loss += torch.nn.functional.l1_loss(pred_bboxes_frame[gt_temp_objectness_frame.bool()], gt_bboxes_frame, reduction='sum')
                    temporal_objectness_loss += torch.nn.functional.binary_cross_entropy_with_logits(logits_temp_objectness_frame, gt_temp_objectness_frame, reduction='sum')
                    num_bboxes += gt_bboxes_frame.shape[0]
                    num_max_bboxes += pred_bboxes_frame.shape[0]
            giou_loss = self.giou_loss_weight * giou_loss / (num_bboxes + 1e-8)
            l1_loss = self.giou_loss_weight * l1_loss / (num_bboxes + 1e-8)
            temporal_objectness_loss = self.temp_objectness_loss_weight * temporal_objectness_loss / (num_max_bboxes + 1e-8)
            total_loss = ce_loss + giou_loss + l1_loss + temporal_objectness_loss
            return {'loss': total_loss, 'ce_loss': ce_loss, 'giou_loss': giou_loss, 'l1_loss': l1_loss, 'temp_objectness_loss': temporal_objectness_loss}
        else:
            for batch_idx, pred_bboxes_video in enumerate(pred_bboxes):
                for frame_idx, pred_bboxes_frame in enumerate(pred_bboxes_video):
                    gt_bboxes_frame = gt_bboxes_list[batch_idx][frame_idx].to(pred_bboxes_frame.device)
                    gt_temp_objectness_frame = gt_temp_objectness_list[batch_idx][frame_idx].to(pred_bboxes_frame.device)
                    assert gt_bboxes_frame.shape[0] == gt_temp_objectness_frame.sum()
                    if gt_bboxes_frame.shape[0] != 0:
                        from utils.bbox_utils import box_cxcywh_to_xyxy
                        giou_loss += torch.ops.torchvision.generalized_box_iou_loss(
                            box_cxcywh_to_xyxy(pred_bboxes_frame[gt_temp_objectness_frame.bool()], type='pt'),
                            box_cxcywh_to_xyxy(gt_bboxes_frame, type='pt'), reduction='sum'
                        )
                        l1_loss += torch.nn.functional.l1_loss(pred_bboxes_frame[gt_temp_objectness_frame.bool()], gt_bboxes_frame, reduction='sum')
                    num_bboxes += gt_bboxes_frame.shape[0]
            giou_loss = self.giou_loss_weight * giou_loss / (num_bboxes + 1e-8)
            l1_loss = self.giou_loss_weight * l1_loss / (num_bboxes + 1e-8)
            total_loss = ce_loss + giou_loss + l1_loss
            return {'loss': total_loss, 'ce_loss': ce_loss, 'giou_loss': giou_loss, 'l1_loss': l1_loss}

    def evaluate(self, image_features, image_forward_outs, images_dtype,
                 image_embeddings, input_ids, original_size_list, max_tokens_new=32,
                 bboxes=None, token_embeddings=None, dense_pe=None, device=None):
        """Evaluate for inference-time generation + bbox decoding.

        Arguments:
          image_features, image_forward_outs, images_dtype: outputs from global encoder (encode_images)
          image_embeddings: SAM grounding image embeddings (from grounding encoder)
          input_ids: prompt input ids
          original_size_list: list of (H,W) original frame sizes
          max_tokens_new: generation length
          bboxes: optional region bboxes passed into generation (currently unused)
          token_embeddings: optional precomputed token embeddings
          dense_pe: dense positional embeddings from SAM prompt encoder
          device: torch device for temporary tensors
        Returns: (generated_ids, pred_bboxes[, logits_temp_objectness])
        """
        with torch.no_grad():
            generation_outputs = self.generate(
                images=None,
                input_ids=input_ids,
                bboxes=bboxes,
                image_features=image_features,
                image_forward_outs=image_forward_outs,
                images_dtype=images_dtype,
                token_embeddings=token_embeddings,
                max_new_tokens=max_tokens_new,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=True,
                synced_gpus=False,
            )
            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences
            # Concatenate layer hidden states into single list element
            output_hidden_states = [torch.cat([output_hidden_states[i] for i in range(len(output_hidden_states))], dim=1)]
            det_token_mask = generated_output_ids[:, 1:] == self.det_token_idx
            if self.det_token_idx is None:
                raise RuntimeError("det_token_idx not set. Call apply_tokenizer_special_ids(tokenizer) before evaluate().")
            # Pad front (575 pre-text tokens incl. image token span)
            det_token_mask = torch.cat([
                torch.zeros((det_token_mask.shape[0], 575), dtype=torch.bool, device=device or generated_output_ids.device),
                det_token_mask
            ], dim=1)
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, det_token_mask, None, infer=True
            )
            if self.config.use_temp_objectness:
                pred_bboxes, logits_temp_objectness = self._generate_and_postprocess_masks(
                    predicted_embeddings, image_embeddings, original_size_list, dense_pe, infer=True
                )
            else:
                pred_bboxes = self._generate_and_postprocess_masks(
                    predicted_embeddings, image_embeddings, original_size_list, dense_pe, infer=True
                )
        if self.config.use_temp_objectness:
            return generated_output_ids, pred_bboxes, logits_temp_objectness
        return generated_output_ids, pred_bboxes
