"""Merge per-setting outputs (…_s0 … _s4) into the single files the analysis expects:
   output/paper/infant_child_{lgbm,enet}{.json,_best_params.json,_pred.jsonl}"""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "output" / "paper"
ok = True
for tag in ["lgbm", "enet"]:
    parts = [OUT / f"infant_child_{tag}_s{i}" for i in range(5)]
    missing = [str(p) for p in parts if not Path(str(p) + "_pred.jsonl").exists()]
    if missing:
        print(f"[{tag}] SKIPPED, missing settings: {missing}"); ok = False; continue
    metrics, params = [], {}
    with open(OUT / f"infant_child_{tag}_pred.jsonl", "w") as fo:
        for p in parts:
            metrics += json.load(open(str(p) + ".json"))
            params.update(json.load(open(str(p) + "_best_params.json")))
            with open(str(p) + "_pred.jsonl") as fi:
                fo.write(fi.read())
    json.dump(metrics, open(OUT / f"infant_child_{tag}.json", "w"), indent=2)
    json.dump(params, open(OUT / f"infant_child_{tag}_best_params.json", "w"), indent=2)
    n = sum(1 for _ in open(OUT / f"infant_child_{tag}_pred.jsonl"))
    print(f"[{tag}] merged 5 settings -> {n} prediction rows (expect 5 x 252 = 1260)")
sys.exit(0 if ok else 1)
