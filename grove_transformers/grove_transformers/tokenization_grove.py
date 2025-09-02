from transformers import PreTrainedTokenizerFast, AutoTokenizer
from .utils.utils import DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN

REGION_TOKENS = ['<bbox>', '<point>']
DETECTION_TOKENS = ['[DET]']
PHRASE_TOKENS = ['<p>', '</p>']
VIDEO_MM_TOKENS = [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN]


class GroveTokenizer(PreTrainedTokenizerFast):
    """Tokenizer that starts from a pretrained base and adds GROVE tokens.

    Use GroveTokenizer.from_pretrained(...). Then call add_grove_tokens().
    """
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_grove_tokens()

    def add_grove_tokens(self, use_mm_start_end: bool = True):
        tokens = []
        if use_mm_start_end:
            tokens.extend(VIDEO_MM_TOKENS)
        tokens.extend(REGION_TOKENS + DETECTION_TOKENS + PHRASE_TOKENS)

        if self.pad_token is None and self.unk_token is not None:
            self.pad_token = self.unk_token

        existing = set(self.get_vocab())
        new_tokens = [t for t in tokens if t not in existing]
        if new_tokens:
            self.add_tokens(new_tokens, special_tokens=True)

        # Infer core special token ids directly from tokenizer state; we do not persist them in config.
        self.grove_special_token_ids = {
            'bbox_token_idx': self.convert_tokens_to_ids('<bbox>'),
            'det_token_idx': self.convert_tokens_to_ids('[DET]'),
            'bop_token_idx': self.convert_tokens_to_ids('<p>'),
            'eop_token_idx': self.convert_tokens_to_ids('</p>'),
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
        }

    def get_special_token_id(self, name: str):
        return self.grove_special_token_ids.get(name)

