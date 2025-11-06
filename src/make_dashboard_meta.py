from pathlib import Path
import re, math, json
import numpy as np
import pandas as pd

# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "stats_roc_mt"
OUT  = ROOT / "results" / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)

DESC_XLSX = ROOT / "Data_Description.xlsx"

# Helpers
# -----------------------------------------
def normalize_area(a: str | None) -> int | None:
    """Map a free-form area string to an integer area_id (1,2,...) or None."""
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return None
    s = str(a).strip().lower().replace("&", "and").replace(" ", "")
    if "area1and2" in s or "area12" in s:
        return None  
    if s in ("area1", "area01", "1", "a1"):
        return 1
    if s in ("area2", "area02", "2", "a2"):
        return 2
    m = re.match(r"area(\d+)$", s)
    return int(m.group(1)) if m else None

BUS_RE = re.compile(r"\bbus\s*(\d+)\b", re.IGNORECASE)
VB_RE  = re.compile(r"V_B(\d+)")  

def parse_bus_from_text(s: str | None) -> int | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = BUS_RE.search(str(s))
    return int(m.group(1)) if m else None

# Coordinates for layout 
# -----------------------------------------
CENTERS_CYCLE = [(0.25,0.75),(0.75,0.75),(0.25,0.25),(0.75,0.25)]

def area_center_for_index(i: int) -> tuple[float,float]:
    return CENTERS_CYCLE[i % len(CENTERS_CYCLE)]

SUB_POS = {
    1: 0.15,  
    2: 0.35,  
}
def sub_xy(area_idx_zero_based: int, substation_idx: int) -> tuple[float,float]:
    cx, cy = area_center_for_index(area_idx_zero_based)
    x = SUB_POS.get(substation_idx, cx)
    return (x if cx < 0.5 else 1 - (1 - x), cy)  

def build_from_description():
    if not DESC_XLSX.exists():
        return None
    try:
        df = pd.read_excel(DESC_XLSX, sheet_name=0)
    except Exception as e:
        print(f"[warn] Could not read {DESC_XLSX}: {e}")
        return None

    file_col = "File"
    area_col = "Unnamed: 4"
    bus_col  = "Unnamed: 5"

    missing = [c for c in [file_col, area_col, bus_col] if c not in df.columns]
    if missing:
        print(f"[warn] XLSX missing columns {missing}; has {list(df.columns)}")
        return None

    rows = []
    for _, r in df.iterrows():
        bus_id  = parse_bus_from_text(r.get(bus_col))
        area_id = normalize_area(r.get(area_col))
        if bus_id is None or area_id is None:
            continue
        rows.append((int(bus_id), int(area_id)))

    if not rows:
        print("[warn] description had no usable (bus, area) rows.")
        return None

    md = pd.DataFrame(rows, columns=["bus_id","area_id"]).drop_duplicates()
    print(f"[source] Using Data_Description.xlsx → {len(md)} unique buses")
    return md

def build_from_artifacts():
    tb = ART / "test_bus_ids.npy"
    ta = ART / "test_y_area.npy"
    if not tb.exists() or not ta.exists():
        return None
    B = np.load(tb)
    A = np.load(ta)
    if B.size == 0 or A.size == 0:
        return None
    bus_area = []
    for b, a in zip(B, A):
        try:
            bi, ai = int(b), int(a)
            if bi >= 0:
                bus_area.append((bi, ai))
        except Exception:
            pass
    if not bus_area:
        return None
    md = pd.DataFrame(bus_area, columns=["bus_id", "area_id"]).drop_duplicates()
    print(f"[source] Using artifacts (.npy) → {len(md)} unique buses")
    return md


def build_from_csv_columns():
    sample = None
    for cand in ["Data_1.csv", "Data_10.csv", "Data_23.csv"]:
        p = ROOT / cand
        if p.exists():
            sample = p
            break
    if sample is None:
        print("[warn] No sample CSV found to infer bus columns.")
        return None
    try:
        df = pd.read_csv(sample, nrows=3)
    except Exception as e:
        print(f"[warn] Could not read {sample}: {e}")
        return None

    bus_ids = sorted({
        int(m.group(1)) for c in df.columns
        for m in [VB_RE.search(str(c))] if m
    })
    if not bus_ids:
        print("[warn] No V_Bxx columns found in sample CSV.")
        return None

    try:
        maps = json.load(open(ART / "label_maps.json", "r"))
        area_name_to_id = maps.get("area", {})
        area_ids = sorted(area_name_to_id.values()) or [0,1]
    except Exception:
        area_ids = [0,1]

    pairs = []
    for i, b in enumerate(bus_ids):
        pairs.append((int(b), int(area_ids[i % len(area_ids)])))
    md = pd.DataFrame(pairs, columns=["bus_id","area_id"]).drop_duplicates()
    print(f"[source] Using CSV columns fallback → {len(md)} buses")
    return md

def build_bus_area_table():
    def _is_good(x):
        return isinstance(x, pd.DataFrame) and not x.empty

    md = build_from_description()
    if _is_good(md):
        return md

    md = build_from_artifacts()
    if _is_good(md):
        return md

    md = build_from_csv_columns()
    if _is_good(md):
        return md

    raise SystemExit("Could not build bus↔area mapping from any source.")


def main():
    md = build_bus_area_table()

    md["bus_id"] = md["bus_id"].astype(int)
    md["area_id"] = md["area_id"].astype(int)
    md = md.drop_duplicates().sort_values(["area_id","bus_id"])

    uniq_areas = sorted(md["area_id"].unique().tolist())


    md["substation_idx"] = md["bus_id"].apply(lambda b: 2 if (b % 2 == 0) else 1)
    md["substation"] = md["substation_idx"].apply(lambda i: f"Substation {i}")


    area_rank = {a: i for i, a in enumerate(uniq_areas)}
    xy = md.apply(
        lambda r: sub_xy(area_rank[int(r["area_id"])], int(r["substation_idx"])),
        axis=1
    )
    md["x"] = [round(t[0], 4) for t in xy]
    md["y"] = [round(t[1], 4) for t in xy]

    bus_csv = OUT / "bus_metadata.csv"
    md_out = md[["bus_id","area_id","substation","x","y"]].sort_values(["area_id","substation","bus_id"])
    md_out.to_csv(bus_csv, index=False)
    print(f"[ok] wrote {bus_csv} rows: {len(md_out)}")

    try:
        maps = json.load(open(ART / "label_maps.json", "r"))
        area_name_to_id = maps.get("area", {})
        id_to_name = {int(v): str(k) for k, v in area_name_to_id.items()}
    except Exception:
        id_to_name = {}

    area_rows = []
    for a in uniq_areas:
        idx = area_rank[a]
        cx, cy = area_center_for_index(idx)
        nm = id_to_name.get(int(a), f"Area {a}")
        area_rows.append({"area_id": int(a), "area_name": nm, "center_x": cx, "center_y": cy})

    area_csv = OUT / "area_metadata.csv"
    pd.DataFrame(area_rows).to_csv(area_csv, index=False)
    print(f"[ok] wrote {area_csv} rows: {len(area_rows)}")

if __name__ == "__main__":
    main()
