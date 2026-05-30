# tests/test_decoder_build.py
from procseq.tokenizer import build_tokenizer
from procseq.models.decoder import build_decoder

def test_build_decoder_tiny_has_correct_vocab():
    tok = build_tokenizer()
    model = build_decoder(size="tiny", tokenizer=tok)
    assert model.config.vocab_size == len(tok)
    assert model.config.hidden_size == 128
