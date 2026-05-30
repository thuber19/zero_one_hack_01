from procseq.tokenizer import build_tokenizer
from procseq.datasets import ClmDataset, clm_collate

def test_clm_dataset_item_has_input_ids_and_family():
    tok = build_tokenizer()
    seqs = [(["RECEIVE WAFER LOT", "SHIP LOT"], "MOSFET")]
    ds = ClmDataset(seqs, tok, max_len=16)
    item = ds[0]
    assert "input_ids" in item and item["input_ids"][0] == tok.bos_token_id

def test_clm_collate_pads_and_masks_labels():
    tok = build_tokenizer()
    seqs = [(["RECEIVE WAFER LOT", "SHIP LOT"], "MOSFET"),
            (["RECEIVE WAFER LOT", "THERMAL OXIDATION", "SHIP LOT"], "IC")]
    ds = ClmDataset(seqs, tok, max_len=16)
    batch = clm_collate([ds[0], ds[1]], pad_id=tok.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    # labels == -100 wherever input is pad
    import torch
    assert (batch["labels"][batch["input_ids"] == tok.pad_token_id] == -100).all()
