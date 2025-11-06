from pathlib import Path
import os, re, json
import numpy as np
import pandas as pd
import streamlit as st
import torch, torch.nn as nn

# ========= THEME & STYLES =========
APP_TITLE = "Smart Grid Event Display"
EVENT_COLORS = {"TopologyChange":"#3FA7F5","LoadChange":"#F59E0B","Fault":"#EF4444","Normal":"#22C55E"}
EVENT_BADGE = {
    "TopologyChange": {"emoji": "🧭", "color": EVENT_COLORS["TopologyChange"]},
    "LoadChange":     {"emoji": "⚙️",  "color": EVENT_COLORS["LoadChange"]},
    "Fault":          {"emoji": "🚨", "color": "#EF4444"},
    "Normal":         {"emoji": "✅",  "color": EVENT_COLORS["Normal"]},
}
def inject_local_css(path: str = "src/dashboard_layout.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at: {path}")

def hero_html():
    return f"""
    <div class="sg-hero">
      <div class="logo">⚡</div>
      <div>
        <div class="title">{APP_TITLE}</div>
        <div class="subtitle">System → Area → Substation</div>
      </div>
    </div>
    """
def event_badge_html():
    chips = []
    for name, meta in EVENT_BADGE.items():
        chips.append(
            f'<span class="sg-badge"><span class="sg-dot" style="background:{meta["color"]}"></span>'
            f'{meta["emoji"]} <b>{name}</b></span>'
        )
    return '<div class="sg-legend">' + "".join(chips) + '</div>'

# ========= FEATURE HELPERS =========
def numeric(df): return df.apply(pd.to_numeric, errors="coerce").select_dtypes(include="number")
def roc(df):
    time_col = next((c for c in df.columns if c.strip().lower()=="time"), None)
    if time_col is None: return df.diff().fillna(0)
    dt = df[time_col].diff().to_numpy(); dt[dt==0] = np.nan
    return df.drop(columns=[time_col]).diff().divide(dt, axis=0).fillna(0)
def combine_stats(df):
    cols = [c for c in df.columns if c.strip().lower()!="time"]
    if not cols: return np.array([])
    arr = df[cols].to_numpy()
    mean  = np.nanmean(arr,0); std=np.nanstd(arr,0)
    vmin  = np.nanmin(arr,0); vmax=np.nanmax(arr,0)
    delta = (arr[-1]-arr[0]) if len(arr)>=2 else np.zeros(arr.shape[1])
    rms   = np.sqrt(np.nanmean(arr**2,0))
    return np.concatenate([mean,std,vmin,vmax,delta,rms])

def infer_primary_bus_from_df(df: pd.DataFrame, buses_df: pd.DataFrame) -> int | None:

    if df is None or df.empty or buses_df is None or buses_df.empty:
        return None
    valid_ids = set(buses_df["bus_id"].dropna().astype(int).unique().tolist())
    if not valid_ids:
        return None

    candidates = []
    for col in df.columns:
        name = str(col).strip()
        if name.lower() == "time": 
            continue
        nums = re.findall(r"\d+", name)
        picked = None
        for n in nums:
            try:
                nid = int(n)
                if nid in valid_ids:
                    picked = nid
                    break
            except ValueError:
                continue
        if picked is not None:
            candidates.append((col, picked))
    if not candidates:
        return None

    tv_best, bus_best = -1.0, None
    for col, bid in candidates:
        s = pd.to_numeric(df[col], errors="coerce")
        tv = s.diff().abs().sum()
        if pd.notna(tv) and float(tv) > tv_best:
            tv_best, bus_best = float(tv), int(bid)
    return bus_best

# ========= MODEL =========
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

def load_event_model_and_maps(art_dir: Path, hidden=(256,128)):
    Xtr = np.load(art_dir/"train_X.npy")
    yev = np.load(art_dir/"train_y_event.npy")
    in_dim = Xtr.shape[1]; ncls = int(yev.max()+1)
    with open(art_dir/"label_maps.json") as f:
        maps = json.load(f)
    inv_event = {v:k for k,v in maps["event"].items()}
    model = MLP(in_dim, hidden, ncls)
    model.load_state_dict(torch.load(art_dir/"mlp_event.pt", map_location="cpu"))
    model.eval()
    return model, inv_event

# ========= UI HELPERS =========
def display_area_label(aid: int, areas_df: pd.DataFrame | None, area_rank: dict[int, int]) -> str:
    """Return a clean, unique display label for an area id using a 1..N rank."""
    try:
        ia = int(aid)
    except Exception:
        return f"Area {aid}"

    rank = area_rank.get(ia)
    if rank is not None:
        return f"Area {rank}"
    if areas_df is not None and not areas_df.empty and "area_id" in areas_df.columns:
        row = areas_df[areas_df["area_id"].astype(int) == ia]
        if not row.empty:
            nm = (row.iloc[0].get("area_name") or "").strip()
            if nm:
                return nm
    return f"Area {ia + 1}"


def render_cards(
    buses_df: pd.DataFrame,
    area_id: int,
    substation: str,
    event_type: str,
    bus_id: int | None,
    areas_df: pd.DataFrame | None = None,
    area_rank: dict[int, int] | None = None,
):
    import re
    area_rank = area_rank or {}
    area_ids = sorted(buses_df["area_id"].dropna().astype(int).unique().tolist())
    if not area_ids:
        return

    etype_norm = re.sub(r"\s+", "", str(event_type)).lower()
    cols = st.columns(len(area_ids))

    for idx, a in enumerate(area_ids):
        with cols[idx]:
            parts = []

            title = display_area_label(a, areas_df, area_rank)
            parts.append(f"<h3>{title}</h3>")

            if a == area_id and bus_id is not None:
                parts.append(
                    f'<div class="sg-alert">'
                    f'In <b>{title}</b> · '
                    f'At <b>Bus {bus_id}</b> · '
                    f'Event <span class="ev"><b>{event_type}</b></span>'
                    f'</div>'
                )

            subs = (
                buses_df[buses_df["area_id"].astype(int) == a]
                ["substation"].dropna().unique().tolist()
            )

            if not subs:
                if len(parts) == 1:
                    continue
                parts.append('<div class="sg-alert">No substations in this area.</div>')
            else:
                for ss in subs:
                    label = str(ss)
                    if label.upper().startswith("SS-"):
                        label = label.replace("SS-", "Substation ")

                    active = (a == area_id and str(ss) == str(substation))
                    row_classes = "sg-sub sg-active" if active else "sg-sub"

                    top_on    = active and (etype_norm == "topologychange")
                    load_on   = active and (etype_norm == "loadchange")
                    fault_on  = active and (etype_norm == "fault")
                    normal_on = not active

                    dots_html = (
                        '<div class="sg-dots">'
                        + dot(EVENT_COLORS["TopologyChange"], off=not top_on)
                        + dot(EVENT_COLORS["LoadChange"],     off=not load_on)
                        + dot(EVENT_COLORS["Fault"],          off=not fault_on)
                        + dot(EVENT_COLORS["Normal"],         off=not normal_on)
                        + '</div>'
                    )
                    tag_html = f'<span class="sg-tag">Bus {bus_id}</span>' if active and bus_id is not None else ''

                    parts.append(
                        f'<div class="{row_classes}">'
                        f'<div class="sg-subtitle">{label}</div>'
                        f'{dots_html}{tag_html}'
                        f'</div>'
                    )

            inner = "".join(parts).strip()
            if not inner:
                continue

            st.markdown('<div class="sg-card">', unsafe_allow_html=True)
            st.markdown(inner, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def dot(color, off=False):
    return f'<span class="sg-dot" style="background:{("#CBD5E1" if off else color)}"></span>'


def render_sidebar(roots, files, labels, default_index=0):
    st.sidebar.markdown(
        '<div class="sg-sidecard">'
        '<div class="sg-side-title">🧪 Test a file</div>'
        '<div style="font-size:12px;opacity:.8;margin:-4px 0 8px;">Choose a CSV from your test sets</div>'
        '</div>', unsafe_allow_html=True
    )

    counts = {}
    for p in files:
        key = p.parent.name
        counts[key] = counts.get(key, 0) + 1

    chips_html = ""
    for k, v in sorted(counts.items()):
        if k.lower().startswith("topology"):
            dot_class = "topology"
        elif k.lower().startswith("load"):
            dot_class = "load"
        elif k.lower().startswith("fault"):
            dot_class = "fault"
        else:
            dot_class = "normal"
        chips_html += f'<span class="sg-chip"><span class="dot {dot_class}"></span>{k}: {v}</span>'

    st.sidebar.markdown(f'<div class="sg-sidecard">{chips_html}</div>', unsafe_allow_html=True)

    chosen_label = st.sidebar.selectbox("Test files", labels, index=default_index)
    c1, c2 = st.sidebar.columns(2)
    run = c1.button("Classify ▶", use_container_width=True)
    reset = c2.button("Reset view", use_container_width=True)
    return chosen_label, run, reset


# ========= APP =========
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    data_dir = Path(os.environ.get("DASH_DATA_DIR", "results/dashboard/data"))
    art_dir  = Path(os.environ.get("ART_DIR", "stats_roc_mt"))
    last_file = data_dir/"predictions_mlp.csv"

    inject_local_css("src/dashboard_layout.css")
    st.markdown(hero_html(), unsafe_allow_html=True)
    st.markdown(event_badge_html(), unsafe_allow_html=True)

    try:
        buses = pd.read_csv(data_dir/"bus_metadata.csv")
        buses["area_id"] = buses["area_id"].astype(int)
        unique_area_ids = sorted(buses["area_id"].astype(int).unique())
        area_rank = {aid: i + 1 for i, aid in enumerate(unique_area_ids)}

    except Exception as e:
        st.error(f"Could not read {data_dir/'bus_metadata.csv'}: {e}")
        return
    try:
        areas = pd.read_csv(data_dir/"area_metadata.csv")
    except Exception:
        areas = pd.DataFrame()  

    roots = [Path("test/Fault"), Path("test/LoadChange"), Path("test/TopologyChange")]
    files = [p for r in roots if r.exists() for p in sorted(r.glob("*.csv"), key=lambda p: (p.parent.name, p.name))]
    labels = [f"{p.parent.name} / {p.name}" for p in files]

    selected, run, reset = render_sidebar(
    roots, files, labels, default_index=(0 if labels else 0)
)


    if reset:
        try:
            if last_file.exists(): last_file.unlink()
        except Exception:
            pass
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    pred = None
    if run and selected:
        chosen = files[labels.index(selected)]
        try:
            df = pd.read_csv(chosen)
            fv = np.concatenate([combine_stats(numeric(df)), combine_stats(roc(numeric(df)))]).astype(np.float32)
            x = torch.tensor(fv)[None, :]
            bus_id = infer_primary_bus_from_df(df, buses)
            model, inv_event = load_event_model_and_maps(art_dir)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).numpy()[0]
                pred_idx  = int(probs.argmax())
            event_name = inv_event.get(pred_idx, f"class_{pred_idx}")
            conf = float(probs[pred_idx])

            area_id, substation = None, "Unknown"
            if bus_id is not None:
                hit = buses[buses["bus_id"].astype(int)==int(bus_id)]
                if not hit.empty:
                    area_id = int(hit.iloc[0]["area_id"])
                    substation = str(hit.iloc[0]["substation"])

            pred = {
                "timestamp": pd.Timestamp.now().floor("s"),
                "file": chosen.name,
                "event": event_name,
                "confidence": conf,
                "bus_id": bus_id,
                "area_id": area_id,
                "substation": substation,
            }
            data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([pred]).to_csv(last_file, index=False)

        except Exception as e:
            st.error(f"Classification failed: {e}")
            return
    elif last_file.exists():
        try:
            last = pd.read_csv(last_file)
            if not last.empty: pred = last.iloc[0].to_dict()
        except Exception:
            pred = None

    if pred is None:
        st.info("Pick a file and click **Classify ▶** to see results.")
        return

    st.success(f"{pred['file']} → {pred['event']} (conf {pred['confidence']:.2f})")
    if pred.get("bus_id") is None or pred.get("area_id") is None:
        st.warning("Could not map this file to a bus/area; check bus column names in the CSV.")


    render_cards(
        buses_df=buses,
        area_id=int(pred["area_id"]) if pred.get("area_id") is not None else -1,
        substation=str(pred.get("substation") or "Unknown"),
        event_type=str(pred["event"]),
        bus_id=int(pred["bus_id"]) if pred.get("bus_id") is not None else None,
        areas_df=areas,
        area_rank=area_rank, 
    )


    with st.expander("Debug details"):
        st.write(pred)
        st.write("Unique areas from buses:", list(map(lambda x: int(x), sorted(buses['area_id'].unique()))))

if __name__ == "__main__":
    main()
