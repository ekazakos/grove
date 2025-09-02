from transformers import LlamaConfig


class GroveConfig(LlamaConfig):
    model_type = 'grove'

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        vocab_size: int = 32000,
        attn_implementation: str | None = None,
        mm_vision_tower: str = 'openai/clip-vit-large-patch14-336',
        mm_vision_select_layer: int = -2,
        mm_vision_select_feature: str = 'patch',
        out_dim: int = 512,
        train_mask_decoder: bool = False,
        use_temp_objectness: bool = True,
        temp_objectness_threshold: float = 0.5,
        num_frames: int | None = None,
        ce_loss_weight: float = 1.0,
        giou_loss_weight: float = 1.0,
        temp_objectness_loss_weight: float = 1.0,
        with_region: bool = True,
        special_tokens: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        self.mm_vision_tower = mm_vision_tower
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.out_dim = out_dim
        self.train_mask_decoder = train_mask_decoder
        self.use_temp_objectness = use_temp_objectness
        self.temp_objectness_threshold = temp_objectness_threshold
        self.num_frames = num_frames if num_frames is not None else 1
        self.ce_loss_weight = ce_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.temp_objectness_loss_weight = temp_objectness_loss_weight
        self.with_region = with_region
        self.special_tokens = special_tokens or {}

        if attn_implementation is not None:
            setattr(self, '_attn_implementation', attn_implementation)
