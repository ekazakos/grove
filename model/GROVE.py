import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from utils.bbox_utils import box_cxcywh_to_xyxy, unnormalize_bboxes


def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class GROVEBaseModel:
    def __init__(self, config, **kwargs):
        super(GROVEBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_grove_model(self.config)

    def initialize_grove_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained, use_temp_objectness=config.use_temp_objectness)
        # self._configure_grounding_encoder(config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        # self.text_hidden_fcs.train()
        # self.text_hidden_fcs.train()


class GROVEModel(GROVEBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GROVEModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class GROVEForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GROVEModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        config.num_frames = kwargs.get("num_frames", None)
        config.temp_objectness_threshold = kwargs.get("temp_objectness_threshold", 0.5)
        config.use_temp_objectness = kwargs.get("use_temp_objectness", True)
        self.det_token_idx = kwargs.pop("det_token_idx")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.giou_loss_weight = kwargs.pop("giou_loss_weight", None)
        self.temp_objectness_loss_weight = kwargs.pop("temp_objectness_loss_weight", None)

    # def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
    #     with torch.no_grad():
    #         return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    # def _encode_single_image(self, image):
    #     # torch.cuda.empty_cache()
    #     return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))
    def get_grounding_encoder_embs(self, images: torch.FloatTensor):
        # torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(images)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)  
        elif "mode" in kwargs and kwargs["mode"] == "encode_images":
            return self.encode_images(kwargs["images"])
        elif "mode" in kwargs and kwargs["mode"] == "get_grounding_encoder_embs":
            return self.get_grounding_encoder_embs(kwargs["images"])
        elif "mode" in kwargs and kwargs["mode"] == "get_dense_pe":
            return self.model.grounding_encoder.prompt_encoder.get_dense_pe()
        elif "mode" in kwargs and kwargs["mode"] == "evaluate":
            return self.evaluate(kwargs["image_features"], kwargs["image_forward_outs"], kwargs["images_dtype"], 
                                 kwargs["image_embeddings"], kwargs["input_ids"], kwargs["original_size_list"], 
                                 max_tokens_new=kwargs["max_tokens_new"], bboxes=kwargs["bboxes"], 
                                 token_embeddings=kwargs["token_embeddings"], dense_pe=kwargs["dense_pe"], 
                                 device=kwargs["device"])
        else:
            return self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes_region: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, bboxes_list: List[torch.FloatTensor],
                      temp_objectness_labels_list: List[torch.FloatTensor], 
                      original_size_list: List[torch.Tensor], inference: bool = False, **kwargs, ):
        # Extract grounding encoder image embeddings
        image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
        # Tile the offset for each frame
        # offset = offset.tile(self.config.num_frames)
        # assert image_embeddings.shape[0] / self.config.num_frames == len(offset) / self.config.num_frames - 1
        # Create segmentation token mask
        det_token_mask = self._create_det_token_mask(input_ids)

        # Handle inference or training paths
        if inference:
            # output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)
            output_hidden_states = self._inference_path(input_ids, global_enc_images)
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, bboxes_region, input_ids, labels, attention_masks, offset
            )

        # Process hidden states
        hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, det_token_mask, offset)

        # Generate and post-process masks
        dense_pe = self.model.grounding_encoder.prompt_encoder.get_dense_pe()
        if self.config.use_temp_objectness:
            pred_bboxes, logits_temp_objectness = self._generate_and_postprocess_masks(
            pred_embeddings, image_embeddings, original_size_list, dense_pe, infer=inference
            )
        else:
            pred_bboxes = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, original_size_list, dense_pe, infer=inference
            )
            logits_temp_objectness = None
        if inference:
            # return {"pred_bboxes": pred_bboxes, "logits_temp_objectness": logits_temp_objectness, 
            #         "gt_bboxes": bboxes_list, "gt_temp_objectness": temp_objectness_labels_list, }
            return {"pred_bboxes": pred_bboxes, "logits_temp_objectness": logits_temp_objectness,}

        # Calculate losses
        return self._calculate_losses_video(pred_bboxes, logits_temp_objectness, bboxes_list, temp_objectness_labels_list, output)

    def _create_det_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.det_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images):
        # length = input_ids.shape[0]
        # global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # # Process and return inference output
        # output_hidden_states = []
        # for i in range(input_ids.shape[0]):
        #     output_i = super().forward(
        #         images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
        #         input_ids=input_ids[i:i + 1], output_hidden_states=True, )
        #     output_hidden_states.append(output_i.hidden_states)
        #     torch.cuda.empty_cache()

        # output_hidden_states = torch.cat(output_hidden_states, dim=0)
        # output_hidden_states = [output_hidden_states]
        # return output_hidden_states
        output = super().forward(
            images=global_enc_images, input_ids=input_ids, output_hidden_states=True,)
        output_hidden_states = output.hidden_states
        output_hidden_states = [output_hidden_states]
        return output_hidden_states


    def _training_path(self, global_enc_images, bboxes_region, input_ids, labels, attention_masks, offset):
        # global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_region_list = bboxes_region

        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, bboxes=bboxes_region_list, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, det_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # For video, repeat the hidden states and masks for each frame to match image embeddings from the grounding encoder.
        # This effectively means that in the decoder, we will perform cross-attention with each frame's image embeddings
        # and the video-level text embeddings.
        last_hidden_state = last_hidden_state.repeat_interleave(self.config.num_frames, dim=0)
        det_token_mask = det_token_mask.repeat_interleave(self.config.num_frames, dim=0)

        pred_embeddings = last_hidden_state[det_token_mask]
        det_token_counts = det_token_mask.int().sum(-1)
        det_token_offset = det_token_counts.cumsum(-1)
        det_token_offset = torch.cat([torch.zeros(1).long().cuda(), det_token_offset], dim=0)
        # if not infer:
            # det_token_offset = det_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(det_token_offset) - 1):
            start_i, end_i = det_token_offset[i], det_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, orig_sizes, dense_pe, infer=False):        
        bs = len(pred_embeddings)
        num_masks_per_embed = [embed.shape[0] for embed in pred_embeddings]
        pred_embeddings = torch.cat(pred_embeddings, dim=0).unsqueeze(1)

        sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
            points=None, boxes=None, masks=None, text_embeds=pred_embeddings
        )
        sparse_embeddings = sparse_embeddings.to(pred_embeddings.dtype)

        bbox_pred_list = []
        temp_objectness_logits_list = []
        if self.config.use_temp_objectness:
            bbox_preds, temp_objectness_logits = self.model.grounding_encoder.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
            multimask_output=False, 
            reps=num_masks_per_embed)
        else:
            bbox_preds = self.model.grounding_encoder.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
            multimask_output=False, 
            reps=num_masks_per_embed)

        start_idx = 0
        for i in range(0, bs, self.config.num_frames):
            bbox_pred_list_sample = []
            if self.config.use_temp_objectness:
                temp_objectness_logits_list_sample = []
            for j in range(self.config.num_frames):
                # During inference, unnormalize the bounding boxes and select the boxes per frame
                # where the objectness is greater than 0.5
                if infer:
                    # Unnormalize the bounding boxes
                    orig_size = orig_sizes[i//self.config.num_frames]
                    unnormalized_bboxes = unnormalize_bboxes(bbox_preds[start_idx:start_idx+num_masks_per_embed[i+j]], 
                                                             orig_size[0], orig_size[1], type='pt')
                    unnormalized_bboxes = box_cxcywh_to_xyxy(unnormalized_bboxes, type="pt")
                    # Select the object bboxes based on objectness
                    if self.config.use_temp_objectness:
                        temp_objectness_preds = F.sigmoid(temp_objectness_logits[start_idx:start_idx+num_masks_per_embed[i+j]])
                        temp_objectness_preds = temp_objectness_preds > self.config.temp_objectness_threshold
                        bbox_pred_list_sample.append(unnormalized_bboxes[temp_objectness_preds.bool()])
                        temp_objectness_logits_list_sample.append(temp_objectness_logits[start_idx:start_idx+num_masks_per_embed[i+j]])
                    else:
                        bbox_pred_list_sample.append(unnormalized_bboxes)
                    start_idx += num_masks_per_embed[i+j]
                else:
                    bbox_pred_list_sample.append(bbox_preds[start_idx:start_idx+num_masks_per_embed[i+j]])
                    if self.config.use_temp_objectness:
                        temp_objectness_logits_list_sample.append(temp_objectness_logits[start_idx:start_idx+num_masks_per_embed[i+j]])
                    start_idx += num_masks_per_embed[i+j]
            bbox_pred_list.append(bbox_pred_list_sample)
            if self.config.use_temp_objectness:
                temp_objectness_logits_list.append(temp_objectness_logits_list_sample)
        if self.config.use_temp_objectness:
            return bbox_pred_list, temp_objectness_logits_list
        else:
            return bbox_pred_list

    
    def _calculate_losses_video(self, pred_bboxes, logits_temp_objectness, gt_bboxes_list, gt_temp_objectness_list, output):
        loss_components = self._compute_loss_components_video(pred_bboxes, logits_temp_objectness, gt_bboxes_list, 
                                                              gt_temp_objectness_list, output)
        return loss_components
    
    def _compute_loss_components_video(self, pred_bboxes, logits_temp_objectness, gt_bboxes_list, gt_temp_objectness_list,  output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        giou_loss = torch.tensor(0.0, device=ce_loss.device)
        l1_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.config.use_temp_objectness and logits_temp_objectness is not None:
            temporal_objectness_loss = torch.tensor(0.0, device=ce_loss.device)
        num_bboxes = 0
        num_max_bboxes = 0
        if self.config.use_temp_objectness and logits_temp_objectness is not None:
            # Iterate over batch and compute bbox-related losses
            for batch_idx, (pred_bboxes_video, logits_temp_objectness_video)  in enumerate(zip(pred_bboxes, logits_temp_objectness)):
                for frame_idx, (pred_bboxes_frame, logits_temp_objectness_frame) in enumerate(zip(pred_bboxes_video, logits_temp_objectness_video)):
                    gt_bboxes_frame = gt_bboxes_list[batch_idx][frame_idx].to(pred_bboxes_frame.device)
                    gt_temp_objectness_frame = gt_temp_objectness_list[batch_idx][frame_idx].to(logits_temp_objectness_frame.device)

                    assert gt_bboxes_frame.shape[0] == gt_temp_objectness_frame.sum(), f"Number of ground truth bboxes and objectness labels do not match: {gt_bboxes_frame.shape[0]} vs {gt_temp_objectness_frame.sum()}"

                    # Compute Generalized IoU Loss
                    # For each frame, we use the objectness to use only the object bboxes that are present in that frame
                    if gt_bboxes_frame.shape[0] != 0:
                        # Torchvision's generalized_box_iou_loss expects the boxes in (x1, y1, x2, y2) format
                        giou_loss += generalized_box_iou_loss(box_cxcywh_to_xyxy(pred_bboxes_frame[gt_temp_objectness_frame.bool()], type="pt"), 
                                                            box_cxcywh_to_xyxy(gt_bboxes_frame, type="pt"),
                                                            reduction="sum")
                        l1_loss += F.l1_loss(pred_bboxes_frame[gt_temp_objectness_frame.bool()], 
                                            gt_bboxes_frame, 
                                            reduction="sum")
                    # Compute Temporal Objectness Loss
                    temporal_objectness_loss += F.binary_cross_entropy_with_logits(logits_temp_objectness_frame, 
                                                                                gt_temp_objectness_frame, 
                                                                                reduction="sum")
                    num_bboxes += gt_bboxes_frame.shape[0]
                    num_max_bboxes += pred_bboxes_frame.shape[0]
            # Normalize the losses
            giou_loss = self.giou_loss_weight * giou_loss / (num_bboxes + 1e-8)
            l1_loss = self.giou_loss_weight * l1_loss / (num_bboxes + 1e-8)
            temporal_objectness_loss = self.temp_objectness_loss_weight * temporal_objectness_loss / (num_max_bboxes + 1e-8)

            # Aggregate all loss components
            total_loss = ce_loss + giou_loss + l1_loss + temporal_objectness_loss
            return {"loss": total_loss, "ce_loss": ce_loss, "giou_loss": giou_loss, "l1_loss": l1_loss,
                    "temp_objectness_loss": temporal_objectness_loss,}
        else:    
            # Iterate over batch and compute bbox-related losses
            for batch_idx, pred_bboxes_video  in enumerate(pred_bboxes):
                for frame_idx, pred_bboxes_frame in enumerate(pred_bboxes_video):
                    gt_bboxes_frame = gt_bboxes_list[batch_idx][frame_idx].to(pred_bboxes_frame.device)
                    gt_temp_objectness_frame = gt_temp_objectness_list[batch_idx][frame_idx].to(pred_bboxes_frame.device)

                    assert gt_bboxes_frame.shape[0] == gt_temp_objectness_frame.sum(), f"Number of ground truth bboxes and objectness labels do not match: {gt_bboxes_frame.shape[0]} vs {gt_temp_objectness_frame.sum()}"

                    # Compute Generalized IoU Loss
                    # For each frame, we use the objectness to use only the object bboxes that are present in that frame
                    if gt_bboxes_frame.shape[0] != 0:
                        # Torchvision's generalized_box_iou_loss expects the boxes in (x1, y1, x2, y2) format
                        giou_loss += generalized_box_iou_loss(box_cxcywh_to_xyxy(pred_bboxes_frame[gt_temp_objectness_frame.bool()], type="pt"), 
                                                            box_cxcywh_to_xyxy(gt_bboxes_frame, type="pt"),
                                                            reduction="sum")
                        l1_loss += F.l1_loss(pred_bboxes_frame[gt_temp_objectness_frame.bool()], 
                                            gt_bboxes_frame, 
                                            reduction="sum")
                    num_bboxes += gt_bboxes_frame.shape[0]
            # Normalize the losses
            giou_loss = self.giou_loss_weight * giou_loss / (num_bboxes + 1e-8)
            l1_loss = self.giou_loss_weight * l1_loss / (num_bboxes + 1e-8)

            # Aggregate all loss components
            total_loss = ce_loss + giou_loss + l1_loss
            return {"loss": total_loss, "ce_loss": ce_loss, "giou_loss": giou_loss, "l1_loss": l1_loss,}

    # def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
    #              bboxes=None, token_embeddings=None):
    def evaluate(self, image_features, image_forward_outs, images_dtype, 
                 image_embeddings, input_ids, orig_sizes, max_tokens_new=32, bboxes=None, 
                 token_embeddings=None, dense_pe=None, device=None):
        with torch.no_grad():
            # image_features, image_forward_outs = self.encode_images(global_enc_images)
            # images_dtype = global_enc_images.dtype
            generation_outputs = self.generate(
                images=None, input_ids=input_ids, bboxes=bboxes, image_features=image_features, 
                image_forward_outs=image_forward_outs, images_dtype=images_dtype, token_embeddings=token_embeddings,
                max_new_tokens=max_tokens_new, num_beams=1, output_hidden_states=True, return_dict_in_generate=True, 
                do_sample=False, use_cache=True, synced_gpus=False,)
            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences
            # TODO: Add argument for use_cache in evaluate() and put an if condition here         
            output_hidden_states = [torch.cat([output_hidden_states[i] for i in range(len(output_hidden_states))], dim=1)]
            det_token_mask = generated_output_ids[:, 1:] == self.det_token_idx
            # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
            det_token_mask = torch.cat(
                [torch.zeros((det_token_mask.shape[0], 575), dtype=torch.bool).to(device=device), det_token_mask], dim=1, ) 
            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, det_token_mask, None, infer=True
            )
            # image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            if self.config.use_temp_objectness:
                pred_bboxes, logits_temp_objectness = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, orig_sizes, dense_pe, infer=True
            )
            else:
                pred_bboxes = self._generate_and_postprocess_masks(
                    predicted_embeddings, image_embeddings, orig_sizes, dense_pe, infer=True
            )
            # generated_output_ids = F.pad(generated_output_ids, (0, 300 - generated_output_ids.shape[1]), value=-500)
            # pred_masks = [F.pad(pred_mask, (10 - pred_mask.shape[0], 0, 0), value=-500) for pred_mask in pred_masks]
            # pred_masks = torch.stack(pred_masks, dim=0)
        if self.config.use_temp_objectness:
            return generated_output_ids, pred_bboxes, logits_temp_objectness
        else:
            return generated_output_ids, pred_bboxes
