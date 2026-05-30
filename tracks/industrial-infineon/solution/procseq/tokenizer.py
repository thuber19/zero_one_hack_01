"""Custom atomic-step WordLevel tokenizer (one token per process step)."""
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from procseq import vocab as _vocab

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer"

def build_tokenizer(save_dir: Path | None = None) -> PreTrainedTokenizerFast:
    v = _vocab.build_vocab()
    raw = Tokenizer(WordLevel(vocab=v, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    tok.add_special_tokens({"additional_special_tokens":
        ["[FAM_MOSFET]", "[FAM_IGBT]", "[FAM_IC]"]})
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(save_dir))
    return tok

def load_tokenizer(save_dir: Path = DEFAULT_DIR) -> PreTrainedTokenizerFast:
    if (save_dir / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(str(save_dir))
    return build_tokenizer(save_dir)

def _steps_to_text(steps: list[str]) -> str:
    return " ".join(_vocab.step_to_token(s) for s in steps)

def encode_sequence(tok, steps, family=None, add_bos_eos=True) -> list[int]:
    pieces = []
    if add_bos_eos:
        pieces.append(tok.bos_token)
    if family:
        pieces.append(_vocab.FAMILY_TOKEN[family])
    pieces.append(_steps_to_text(steps))
    if add_bos_eos:
        pieces.append(tok.eos_token)
    text = " ".join(p for p in pieces if p)
    return tok.encode(text)

def decode_to_steps(tok, ids) -> list[str]:
    """Inverse of encode_sequence body: drop specials, map tokens->steps."""
    out = []
    specials = set(_vocab.SPECIAL_TOKENS)
    for t in tok.convert_ids_to_tokens(ids):
        if t in specials or t is None:
            continue
        out.append(_vocab.token_to_step(t))
    return out
