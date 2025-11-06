import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd

EVENT_FOLDERS = ["Fault", "LoadChange", "TopologyChange"]
SPLITS = ["train", "test"]

def numeric(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.apply(pd.to_numeric, errors="coerce").select_dtypes(include="number")

def roc(data_frame: pd.DataFrame) -> pd.DataFrame:
    time_col = next((c for c in data_frame.columns if c.strip().lower() == "time"), None)
    if time_col is None:
        return data_frame.diff().fillna(0)
    data = data_frame[time_col].diff().to_numpy()
    data[data == 0] = np.nan
    core = data_frame.drop(columns=[time_col]).diff().divide(data, axis=0)
    return core.fillna(0)

def combine_stats(data_frame: pd.DataFrame) -> np.ndarray:
    cols = [c for c in data_frame.columns if c.strip().lower() != "time"]
    if not cols:
        return np.array([])
    arr = data_frame[cols].to_numpy()
    mean  = np.nanmean(arr, 0)
    std   = np.nanstd(arr, 0)
    vmin  = np.nanmin(arr, 0)
    vmax  = np.nanmax(arr, 0)
    delta = (arr[-1] - arr[0]) if len(arr) >= 2 else np.zeros(arr.shape[1])
    rms   = np.sqrt(np.nanmean(arr ** 2, 0))
    return np.concatenate([mean, std, vmin, vmax, delta, rms])

def encode_labels(names):
    return {k: i for i, k in enumerate(sorted(set(names)))}

def normalize_area(a: str | None) -> str | None:
    if not a:
        return None
    s = str(a).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", "", s)
    if "area1and2" in s or "area12" in s:
        return "Area 1"
    if "area1" in s:
        return "Area 1"
    if "area2" in s:
        return "Area 2"
    m = re.match(r"area(\d+)$", s)
    if m:
        return f"Area {m.group(1)}"
    return a.strip()

def normalize_substation_to_region(s: str | None) -> str | None:
    if not s:
        return None
    t = str(s).replace("Bus", "").replace("bus", "").replace(" ", "")
    m = re.search(r"\d+", t)
    if not m:
        return "Unknown"
    n = int(m.group(0))
    if n < 30:
        return "Region 1 (Bus <30)"
    if n < 60:
        return "Region 2 (Bus 30–59)"
    return "Region 3 (Bus ≥60)"

BUS_RE = re.compile(r"bus\s*(\d+)(?:[-_]\s*\d+)?", re.IGNORECASE)
def extract_primary_bus_id(name_stem: str) -> int | None:
    m = BUS_RE.search(name_stem)
    return int(m.group(1)) if m else None

def load_desc(desc_path: Path):
    data_frame = pd.read_excel(desc_path, sheet_name=0)
    files  = data_frame.get("File")
    events = data_frame.get("Lables")
    areas  = data_frame.get("Unnamed: 4")
    subs   = data_frame.get("Unnamed: 5")

    if files is None:
        raise SystemExit("Spreadsheet must contain a 'File' column.")

    mapping = {}
    for i in range(len(data_frame)):
        f = files.iloc[i]
        if pd.isna(f):
            continue
        e = None if events is None or pd.isna(events.iloc[i]) else str(events.iloc[i]).strip()
        a = None if areas  is None or pd.isna(areas.iloc[i])  else str(areas.iloc[i]).strip()
        s = None if subs   is None or pd.isna(subs.iloc[i])   else str(subs.iloc[i]).strip()

        base = str(f).strip()
        if base.lower().endswith(".csv"):
            base = base[:-4]

        area_clean = normalize_area(a) or "unknown"
        substation_region  = normalize_substation_to_region(s) or "Unknown"
        event_label_final  = e or None
        mapping[base] = (event_label_final, area_clean, substation_region)
    return mapping

def split_data(split_dir: Path, file2labels: dict):
    feature_matrix_list, y_event, y_area, y_sub = [], [], [], []
    bus_ids = []
    num_feat = None

    for event_name in EVENT_FOLDERS:
        evt_dir = split_dir / event_name
        if not evt_dir.exists():
            continue
        for csv in evt_dir.glob("*.csv"):
            raw_df = pd.read_csv(csv)
            data_frame = numeric(raw_df)
            raw_f = combine_stats(data_frame)
            roc_f = combine_stats(roc(data_frame))
            if raw_f.size == 0:
                continue
            feature_vector = np.concatenate([raw_f, roc_f])
            if num_feat is None:
                num_feat = feature_vector.size
            feature_matrix_list.append(feature_vector)

            base = csv.stem
            if base in file2labels:
                e, a, s = file2labels[base]
                event_label = (e or event_name)
                area_label  = a or "unknown"
                sub_label   = s or "Unknown"
            else:
                event_label = event_name
                area_label  = "unknown"
                sub_label   = "Unknown"

            y_event.append(event_label)
            y_area.append(area_label)
            y_sub.append(sub_label)

            bid = extract_primary_bus_id(base)
            if bid is None:
                bid = extract_primary_bus_id(csv.name) or -1
            bus_ids.append(int(bid) if bid is not None else -1)

    lm_event = encode_labels(y_event) if y_event else {}
    lm_area  = encode_labels(y_area)  if y_area  else {}
    lm_sub   = encode_labels(y_sub)   if y_sub   else {}

    if not feature_matrix_list:
        X = np.zeros((0, 0)) if num_feat is None else np.zeros((0, num_feat))
        y_event_encoded = np.zeros((0,), dtype=np.int64)
        y_area_encoded = np.zeros((0,), dtype=np.int64)
        y_sub_encoded = np.zeros((0,), dtype=np.int64)
        bus_ids_arr   = np.zeros((0,), dtype=np.int64)
    else:
        X   = np.vstack(feature_matrix_list)
        y_event_encoded = np.array([lm_event[e] for e in y_event], dtype=np.int64)
        y_area_encoded  = np.array([lm_area[a]  for a in y_area],  dtype=np.int64)
        y_sub_encoded   = np.array([lm_sub[s]   for s in y_sub],   dtype=np.int64)
        bus_ids_arr     = np.array(bus_ids, dtype=np.int64)

    label_maps = {"event": lm_event, "area": lm_area, "substation": lm_sub}
    return X, y_event_encoded, y_area_encoded, y_sub_encoded, bus_ids_arr, label_maps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_r", required=True)
    ap.add_argument("--desc", default="../Data_Description.xlsx")
    ap.add_argument("--output", default="stats_roc_mt")
    args = ap.parse_args()

    data_r, output, desc = Path(args.data_r), Path(args.output), Path(args.desc)
    output.mkdir(parents=True, exist_ok=True)
    if not desc.exists():
        raise SystemExit(f"Description file not found: {desc}")

    file2labels = load_desc(desc)

    all_maps = {"event": {}, "area": {}, "substation": {}}
    for split in SPLITS:
        X, y_event_encoded, y_area_encoded, y_sub_encoded, bus_ids_arr, label_maps = split_data(data_r / split, file2labels)

        for k in all_maps:
            if not all_maps[k] and label_maps[k]:
                all_maps[k] = label_maps[k]

        np.save(output / f"{split}_X.npy",         X)
        np.save(output / f"{split}_y_event.npy",   y_event_encoded)
        np.save(output / f"{split}_y_area.npy",    y_area_encoded)
        np.save(output / f"{split}_y_sub.npy",     y_sub_encoded)
        np.save(output / f"{split}_bus_ids.npy",   bus_ids_arr)    

        print(f"  - {split}: X={X.shape}, y_event={y_event_encoded.shape}, y_area={y_area_encoded.shape}, y_sub={y_sub_encoded.shape}, bus_ids={bus_ids_arr.shape}")

    with open(output / "label_maps.json", "w") as f:
        json.dump(all_maps, f, indent=2)

if __name__ == "__main__":
    main()
