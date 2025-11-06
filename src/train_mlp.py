import argparse, json, time
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, pdrop=0.1):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(pdrop)]
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --------------- IO helpers ---------------
def _load_np(art_dir: Path, name: str):
    p = art_dir / name
    return np.load(p) if p.exists() else np.array([])

def load_artifacts(art_dir: Path):
    x_train = _load_np(art_dir, "train_X.npy")
    x_test  = _load_np(art_dir, "test_X.npy")
    y_train = {
        "event": _load_np(art_dir, "train_y_event.npy"),
        "area":  _load_np(art_dir, "train_y_area.npy"),
        "sub":   _load_np(art_dir, "train_y_sub.npy"),
    }
    y_test = {
        "event": _load_np(art_dir, "test_y_event.npy"),
        "area":  _load_np(art_dir, "test_y_area.npy"),
        "sub":   _load_np(art_dir, "test_y_sub.npy"),
    }
    test_bus_ids = _load_np(art_dir, "test_bus_ids.npy")  # optional

    inverse_lm = {"event": {}, "area": {}, "sub": {}}
    lm = art_dir / "label_maps.json"
    if lm.exists():
        with open(lm) as f:
            maps = json.load(f)
        for k_json, k_local in [("event","event"),("area","area"),("substation","sub")]:
            if k_json in maps:
                inverse_lm[k_local] = {v: k for k, v in maps[k_json].items()}
    return x_train, x_test, y_train, y_test, test_bus_ids, inverse_lm

# --------------- Train + Eval ---------------
def _train_and_eval(x_train, y_train, x_test, y_test, inv_names,
                    epochs=30, hidden=[256,128], lr=1e-3, batch_size=64):
    if x_train.size == 0 or y_train.size == 0:
        print("empty training data")
        return None
    if len(np.unique(y_train)) < 2:
        print("only one class in training data")
        return None

    input_dim = x_train.shape[1]
    num_classes = int(y_train.max() + 1)

    model = MLP(input_dim, hidden=hidden, num_classes=num_classes, pdrop=0.1)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

    preds = probs = None
    if x_test.size != 0 and y_test.size != 0:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32))
            probs  = torch.softmax(logits, dim=1).numpy()
            preds  = probs.argmax(1)

        # report
        if inv_names:
            cls_names = [inv_names[i] for i in sorted(inv_names)]
            print(classification_report(
                y_test, preds,
                labels=list(range(len(cls_names))),
                target_names=cls_names,
                zero_division=0
            ))
        else:
            print(classification_report(y_test, preds, zero_division=0))
    else:
        print("test split empty; skipping report")

    return preds, probs, model

# --------------- Main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="stats_roc_mt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--emit_dashboard", action="store_true",
                    help="write predictions_mlp.csv once after eval")
    ap.add_argument("--stream_test", action="store_true",
                    help="emit test predictions row-by-row for the dashboard")
    ap.add_argument("--dash_out", default="results/dashboard/data")
    ap.add_argument("--stream_delay", type=float, default=0.75,
                    help="seconds to sleep between streamed rows")
    ap.add_argument("--reset_stream", action="store_true",
                    help="truncate predictions_mlp.csv before streaming")
    args = ap.parse_args()

    art = Path(args.artifacts)
    x_train, x_test, y_train, y_test, test_bus_ids, inverse_lm = load_artifacts(art)

    if x_train.size == 0:
        raise SystemExit("Empty training features. Re-run preprocessing with valid CSVs.")

    # ---- EVENT ----
    print("\n=== EVENT level ===")
    ev_res = _train_and_eval(x_train, y_train["event"], x_test, y_test["event"],
                             inverse_lm["event"], epochs=args.epochs)
    if ev_res is None:
        raise SystemExit("Event model failed to train.")
    preds_event, probs_event, event_model = ev_res
    torch.save(event_model.state_dict(), art / "mlp_event.pt")
    print("[saved] mlp_event.pt")

    # ---- AREA ----
    print("\n=== AREA level ===")
    ar_res = _train_and_eval(x_train, y_train["area"], x_test, y_test["area"],
                             inverse_lm["area"], epochs=args.epochs)
    if ar_res is not None:
        _, _, area_model = ar_res
        torch.save(area_model.state_dict(), art / "mlp_area.pt")
        print("[saved] mlp_area.pt")

    # ---- SUBSTATION / REGION ----
    print("\n=== SUBSTATION (REGION) level ===")
    sb_res = _train_and_eval(x_train, y_train["sub"], x_test, y_test["sub"],
                             inverse_lm["sub"], epochs=args.epochs)
    if sb_res is not None:
        _, _, sub_model = sb_res
        torch.save(sub_model.state_dict(), art / "mlp_sub.pt")
        print("[saved] mlp_sub.pt")

    # ------------- DASHBOARD OUTPUTS -------------
    out_dir = Path(args.dash_out); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions_mlp.csv"

    if args.emit_dashboard and (preds_event is not None) and (probs_event is not None):
        conf_event = probs_event.max(axis=1)
        area_ids = y_test["area"]
        if test_bus_ids.size != conf_event.size:
            bus_ids = np.full_like(area_ids, -1)
        else:
            bus_ids = test_bus_ids
        inv_event = inverse_lm["event"]
        names = [inv_event.get(int(i), f"class_{int(i)}") for i in preds_event]
        t0 = pd.Timestamp.now().floor("s") - timedelta(seconds=2*len(names))
        ts = [t0 + timedelta(seconds=2*i) for i in range(len(names))]
        pd.DataFrame({
            "timestamp": ts,
            "event_type": names,
            "bus_id": bus_ids.astype(int),
            "area_id": area_ids.astype(int),
            "confidence": conf_event.astype(float),
        }).to_csv(out_file, index=False)
        print("[dashboard] wrote:", out_file)

    if args.stream_test and (preds_event is not None) and (probs_event is not None):
        conf_event = probs_event.max(axis=1)
        area_ids = y_test["area"]
        if area_ids.size != conf_event.size:
            raise SystemExit("Area ids not aligned with test size; cannot stream.")
        if test_bus_ids.size != conf_event.size:
            bus_ids = np.full_like(area_ids, -1)
        else:
            bus_ids = test_bus_ids

        if args.reset_stream and out_file.exists():
            out_file.unlink()
        header_needed = not out_file.exists()

        inv_event = inverse_lm["event"]
        for i in range(conf_event.size):
            row = {
                "timestamp": pd.Timestamp.now().floor("s"),
                "event_type": inv_event.get(int(preds_event[i]), f"class_{int(preds_event[i])}"),
                "bus_id": int(bus_ids[i]),
                "area_id": int(area_ids[i]),
                "confidence": float(conf_event[i]),
            }
            pd.DataFrame([row]).to_csv(out_file, mode="a", header=header_needed, index=False)
            header_needed = False
            time.sleep(max(0.0, args.stream_delay))
        print("[dashboard] streaming complete:", out_file)

if __name__ == "__main__":
    main()
