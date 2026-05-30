"""Score all three submission files against the internal mirror ground truth."""
import argparse, csv, json
from pathlib import Path
from procseq.config import load_config
from procseq import eval_metrics as em

def _gt_nextstep(valid_gt_path):
    """Derive NEXT_STEP gold from full sequence + partial length per example."""
    gold = {}
    with open(valid_gt_path) as f:
        for r in csv.DictReader(f):
            gold[r["EXAMPLE_ID"]] = r["FULL_SEQUENCE"].split("|")
    return gold

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config); art = Path(cfg["artifacts"])
    results = {}
    # build next-step + completion gold from the mirror input+gt
    full = {}
    with open(art / "eval_valid_groundtruth.csv") as f:
        for r in csv.DictReader(f):
            full[r["EXAMPLE_ID"]] = r["FULL_SEQUENCE"].split("|")
    partial = {}
    with open(art / "eval_input_valid.csv") as f:
        for r in csv.DictReader(f):
            partial[r["EXAMPLE_ID"]] = r["PARTIAL_SEQUENCE"].split("|")
    # Task1 gold = the single next step after the cut
    ns_gold = {eid: full[eid][len(partial[eid])] for eid in partial
               if len(full[eid]) > len(partial[eid])}
    if (art / "submission_task1.csv").exists():
        preds = {}
        with open(art / "submission_task1.csv") as f:
            for r in csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = [r[f"RANK_{k}"] for k in range(1,6)]
        results["task1_nextstep"] = em.score_nextstep(preds, ns_gold)
    # Task2 gold = suffix after cut
    comp_gold = {eid: full[eid][len(partial[eid]):] for eid in partial}
    if (art / "submission_task2.csv").exists():
        preds = {}
        with open(art / "submission_task2.csv") as f:
            for r in csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = (r["PREDICTED_SEQUENCE"].split("|")
                                          if r["PREDICTED_SEQUENCE"] else [])
        results["task2_completion"] = em.score_completion(preds, comp_gold)
        # logic probe: are reconstructed FULL sequences rule-valid?
        recon = [partial[eid] + preds.get(eid, []) for eid in partial]
        results["task2_logic_validity"] = em.logic_validity_rate(recon)
    # Task3
    if (art / "submission_task3.csv").exists():
        gold = {}
        with open(art / "eval_anomaly_labels.csv") as f:
            for r in csv.DictReader(f):
                gold[r["EXAMPLE_ID"]] = (int(r["IS_VALID"]), r["PREDICTED_RULE"])
        preds = {}
        with open(art / "submission_task3.csv") as f:
            for r in csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = (int(r["IS_VALID"]),
                    float(r["SCORE"] or 0.5), r["PREDICTED_RULE"])
        results["task3_anomaly"] = em.score_anomaly(preds, gold)
    out = art / "metrics.json"; out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)); print(f"-> {out}")

if __name__ == "__main__":
    main()
