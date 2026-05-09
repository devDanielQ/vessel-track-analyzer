"""
Vessel Track Analyzer — Streamlit Web App
Detects transshipment events by visualizing two AIS vessel tracks.

Key optimizations over the original tkinter app:
  1. Basemap rendered ONCE as a numpy array; reused across all frames
  2. Distances pre-computed in a single vectorised pass before animation
  3. Frames captured with fig.canvas.buffer_rgba() (no per-frame file I/O)
  4. imageio writes directly to the output container (no ffmpeg subprocess per frame)
  5. NumPy arrays used for coordinate access instead of repeated .iloc[] calls
"""

import io
import os
import tempfile
import time
from groq import Groq

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")                         # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
import numpy as np
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
from PIL import Image
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vessel Track Analyzer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px !important;        /* base size — all rem values scale from here */
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050E1A 0%, #0B1E33 100%);
    border-right: 1px solid #1B3557;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FBBF24 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Main background ── */
.main .block-container {
    background: #060F1C;
    color: #E2E8F0;
    padding-top: 1.5rem;
    max-width: 100% !important;
}
.stApp { background: #060F1C; }

/* ── Page title ── */
.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #F0F9FF;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 0.3rem;
    margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0D1E31;
    border: 1px solid #1B3557;
    border-radius: 10px;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] {
    color: #FBBF24 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.15rem !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.04em;
    color: #FFFFFF !important;
    border-bottom: 2px solid transparent;
}
button[data-baseweb="tab"] p {
    font-size: 0.78rem !important;
    font-family: 'Space Mono', monospace !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FBBF24 !important;
    border-bottom-color: #FBBF24 !important;
}
button[data-baseweb="tab"][aria-selected="true"] p {
    color: #FBBF24 !important;
}

/* ── Generate button ── */
.stButton > button {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
    color: #F0F9FF !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #065F46 0%, #064E3B 100%);
    color: #ECFDF5 !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 8px;
    width: 100%;
    padding: 0.6rem 2rem !important;
}

/* ── Info / warning boxes ── */
[data-testid="stInfo"], .stAlert {
    background: #0D1E31 !important;
    border: 1px solid #1B3557 !important;
    color: #94A3B8 !important;
    border-radius: 10px;
    font-size: 0.82rem !important;
}

/* ── Code blocks ── */
.stCodeBlock {
    background: #0D1E31 !important;
    border: 1px solid #1B3557 !important;
    border-radius: 8px;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1EBD1E, #1EBD1E) !important;
}

/* ── AI Analysis output ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td {
    font-size: 0.88rem !important;
    line-height: 1.75 !important;
}
[data-testid="stMarkdownContainer"] h1 { font-size: 1.2rem !important; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.05rem !important; }
[data-testid="stMarkdownContainer"] h3 { font-size: 0.95rem !important; }

/* ── Streamlit default widget labels ── */
[data-testid="stWidgetLabel"] p { font-size: 0.82rem !important; }
.stSelectbox label, .stSlider label, .stRadio label { font-size: 0.82rem !important; }

/* ── Section dividers ── */
hr { border-color: #1B3557 !important; }

/* ── Chart labels ── */
.stVegaLiteChart { background: #0D1E31; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

RESAMPLE_MAP = {
    "1 min":  "1min",
    "5 min":  "5min",
    "10 min": "10min",
    "15 min": "15min",
    "30 min": "30min",
    "1 hour": "1h",
}


# ── Safety limits ─────────────────────────────────────────────────────────────
MAX_FILE_MB  = 50
MAX_ROWS     = 100_000


# ── Sample data generator ──────────────────────────────────────────────────────
def generate_sample_csv() -> tuple[bytes, bytes]:
    """
    Generate two realistic AIS vessel tracks in the Gulf of Guinea
    that demonstrate a transshipment event — ready to use as demo data.
    """
    rng  = np.random.default_rng(42)
    base = pd.Timestamp("2024-12-08 00:00:00", tz="UTC")
    freq = pd.Timedelta(minutes=5)
    n    = 72 * 12   # 3 days at 5-min intervals

    timestamps = [base + freq * i for i in range(n)]

    # Vessel 1 — fishing vessel, slow, loitering near coast
    lat1 = np.cumsum(rng.normal(0, 0.002, n)) + 4.20
    lon1 = np.cumsum(rng.normal(0.001, 0.002, n)) + 7.10
    # Force rendezvous with Vessel 2 around frame 500
    lat1[480:560] = np.linspace(lat1[479], 4.18, 80) + rng.normal(0, 0.0005, 80)
    lon1[480:560] = np.linspace(lon1[479], 7.35, 80) + rng.normal(0, 0.0005, 80)

    # Vessel 2 — cargo/reefer, faster approach then departure
    lat2 = np.cumsum(rng.normal(-0.001, 0.003, n)) + 4.60
    lon2 = np.cumsum(rng.normal(-0.002, 0.003, n)) + 7.80
    lat2[440:560] = np.linspace(lat2[439], 4.18, 120) + rng.normal(0, 0.0005, 120)
    lon2[440:560] = np.linspace(lon2[439], 7.35, 120) + rng.normal(0, 0.0005, 120)
    lat2[560:620] = np.linspace(4.18, lat2[560] + 0.4, 60)
    lon2[560:620] = np.linspace(7.35, lon2[560] - 0.3, 60)

    def _to_csv(ts, lats, lons) -> bytes:
        df = pd.DataFrame({"dt_pos_utc": ts, "latitude": lats, "longitude": lons})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    return _to_csv(timestamps, lat1, lon1), _to_csv(timestamps, lat2, lon2)


@st.cache_data(show_spinner=False)
def load_vessel_data(file_bytes: bytes, filename: str, resample_key: str) -> pd.DataFrame:
    """Load, clean and resample an AIS CSV file."""

    # ── File size guard ────────────────────────────────────────────────────────
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"**{filename}** is {size_mb:.1f} MB — the maximum allowed is {MAX_FILE_MB} MB.  \n"
            f"Try exporting a shorter date range from your AIS provider, "
            f"or use a coarser resampling interval."
        )

    df = pd.read_csv(io.BytesIO(file_bytes))

    # ── Row count guard ────────────────────────────────────────────────────────
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"**{filename}** contains {len(df):,} rows — the maximum is {MAX_ROWS:,}.  \n"
            f"Please export a shorter date range or filter the data before uploading."
        )

    # ── Column validation with friendly message ────────────────────────────────
    required = {"dt_pos_utc", "longitude", "latitude"}
    missing  = required - set(df.columns)
    if missing:
        found = ", ".join(sorted(df.columns.tolist())[:8])
        extra = "…" if len(df.columns) > 8 else ""
        raise ValueError(
            f"**{filename}** is missing required columns: `{'`, `'.join(sorted(missing))}`.  \n\n"
            f"Required columns are: `dt_pos_utc`, `longitude`, `latitude`.  \n"
            f"Columns found in your file: {found}{extra}.  \n\n"
            f"Rename your columns to match the required names and re-upload."
        )

    df = df[["dt_pos_utc", "longitude", "latitude"]].copy()
    df["dt_pos_utc"] = pd.to_datetime(df["dt_pos_utc"], utc=True)
    df = df.dropna(subset=["longitude", "latitude"])

    if df.empty:
        raise ValueError(
            f"**{filename}** has no valid position data after removing rows with "
            f"missing coordinates. Check that your latitude/longitude columns contain numbers."
        )

    df = (
        df.set_index("dt_pos_utc")
        .resample(RESAMPLE_MAP[resample_key])
        .first()
        .reset_index()
        .dropna(subset=["longitude", "latitude"])
    )
    return df


@st.cache_data(show_spinner=False)
def compute_distances(
    v1_bytes: bytes, v2_bytes: bytes, resample_key: str
) -> list[float]:
    """Pre-compute geodesic distances for every shared timestep (vectorised)."""
    v1 = load_vessel_data(v1_bytes, "v1", resample_key)
    v2 = load_vessel_data(v2_bytes, "v2", resample_key)
    n = min(len(v1), len(v2))
    dists = [
        geodesic(
            (v1["latitude"].iat[i], v1["longitude"].iat[i]),
            (v2["latitude"].iat[i], v2["longitude"].iat[i]),
        ).kilometers
        for i in range(n)
    ]
    return dists


# ─────────────────────────────────────────────────────────────────────────────
# Folium interactive map
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Basemap definitions
# ─────────────────────────────────────────────────────────────────────────────

BASEMAPS = {
    "🌊  Nautical (Esri Ocean)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri Ocean Basemap",
        "overlay": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri Ocean Reference",
            "name": "Ocean labels",
        },
        "track_colors": ("#FFD700", "#FF4500"),
    },
    "🛰️  Satellite (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
        "overlay": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri Boundaries & Places",
            "name": "Labels",
        },
        "track_colors": ("#00FFFF", "#FF4500"),
    },
    "🌑  Dark (CartoDB)": {
        "tiles": "CartoDB dark_matter",
        "attr": "CartoDB",
        "overlay": None,
        "track_colors": ("#38BDF8", "#FB923C"),
    },
    "🗺️  Street (OpenStreetMap)": {
        "tiles": "OpenStreetMap",
        "attr": "OpenStreetMap",
        "overlay": None,
        "track_colors": ("#1D4ED8", "#DC2626"),
    },
    "☁️  Light (CartoDB Positron)": {
        "tiles": "CartoDB positron",
        "attr": "CartoDB",
        "overlay": None,
        "track_colors": ("#1D4ED8", "#DC2626"),
    },
}

def build_folium_map(v1: pd.DataFrame, v2: pd.DataFrame, distances: list[float], basemap_key: str = '🌊  Nautical (Esri Ocean)') -> folium.Map:
    center_lat = (v1["latitude"].mean() + v2["latitude"].mean()) / 2
    center_lon = (v1["longitude"].mean() + v2["longitude"].mean()) / 2

    bm      = BASEMAPS.get(basemap_key, list(BASEMAPS.values())[0])
    c1, c2  = bm["track_colors"]

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=bm["tiles"],
        attr=bm["attr"],
    )

    if bm["overlay"]:
        ov = bm["overlay"]
        folium.TileLayer(
            tiles=ov["tiles"],
            attr=ov["attr"],
            name=ov["name"],
            overlay=True,
            control=True,
            opacity=0.85,
        ).add_to(m)

    folium.LayerControl().add_to(m)

    # Vessel track lines — colors chosen per basemap for contrast
    v1_coords = list(zip(v1["latitude"], v1["longitude"]))
    v2_coords = list(zip(v2["latitude"], v2["longitude"]))

    folium.PolyLine(v1_coords, color=c1, weight=3,
                    opacity=0.95, tooltip="Vessel 1 Track").add_to(m)
    folium.PolyLine(v2_coords, color=c2, weight=3,
                    opacity=0.95, tooltip="Vessel 2 Track").add_to(m)

    # Start / end markers
    def ship_icon(color: str) -> folium.Icon:
        return folium.Icon(color=color, icon="ship", prefix="fa")

    folium.Marker(v1_coords[0],  popup="🔵 Vessel 1 — Start",
                  icon=ship_icon("blue")).add_to(m)
    folium.Marker(v1_coords[-1], popup="🔵 Vessel 1 — End",
                  icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(m)
    folium.Marker(v2_coords[0],  popup="🟠 Vessel 2 — Start",
                  icon=ship_icon("orange")).add_to(m)
    folium.Marker(v2_coords[-1], popup="🟠 Vessel 2 — End",
                  icon=folium.Icon(color="orange", icon="flag", prefix="fa")).add_to(m)

    # Closest-approach hotspot
    if distances:
        idx = int(np.argmin(distances))
        min_dist = distances[idx]
        ts = v1["dt_pos_utc"].iat[idx]
        mid_lat = (v1["latitude"].iat[idx] + v2["latitude"].iat[idx]) / 2
        mid_lon = (v1["longitude"].iat[idx] + v2["longitude"].iat[idx]) / 2

        folium.CircleMarker(
            [mid_lat, mid_lon],
            radius=18,
            color="#EF4444",
            fill=True,
            fill_color="#EF4444",
            fill_opacity=0.25,
            popup=folium.Popup(
                f"<b>⚠️ Closest Approach</b><br>"
                f"Distance: <b>{min_dist:.2f} km</b><br>"
                f"Time: {ts}",
                max_width=220,
            ),
            tooltip=f"⚠️ Closest approach: {min_dist:.2f} km",
        ).add_to(m)

        # Connect vessels at closest point
        folium.PolyLine(
            [
                (v1["latitude"].iat[idx], v1["longitude"].iat[idx]),
                (v2["latitude"].iat[idx], v2["longitude"].iat[idx]),
            ],
            color="#EF4444",
            weight=1.5,
            dash_array="6",
            opacity=0.7,
            tooltip="Closest approach line",
        ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Optimized video generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_optimized_video(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    distances: list[float],
    output_path: str,
    fps: int = 10,
    progress_callback=None,
) -> bool:
    """
    Generates the vessel-track animation with three key speed improvements:

      A) Cartopy basemap rendered ONCE → stored as numpy array.
         All subsequent frames composite data on top without re-invoking cartopy.

      B) A single reusable matplotlib Figure is created; only the Line2D /
         PathCollection artists are mutated each frame (set_data).

      C) Frames are captured via fig.canvas.buffer_rgba() — an in-memory memoryview
         with zero file I/O — then handed directly to imageio.
    """

    def _cb(pct: float, msg: str):
        if progress_callback:
            progress_callback(min(pct, 1.0), msg)

    try:
        # ── Bounds ──────────────────────────────────────────────────────────
        lon_min = min(v1["longitude"].min(), v2["longitude"].min()) - 1
        lon_max = max(v1["longitude"].max(), v2["longitude"].max()) + 1
        lat_min = min(v1["latitude"].min(), v2["latitude"].min()) - 1
        lat_max = max(v1["latitude"].max(), v2["latitude"].max()) + 1
        extent  = [lon_min, lon_max, lat_min, lat_max]

        FIG_W, FIG_H, DPI = 14, 9, 100

        # ── A: Pre-render satellite basemap once ────────────────────────────
        _cb(0.04, "Fetching satellite tiles (once) …")

        import cartopy.io.img_tiles as cimgt
        import math

        class EsriSatellite(cimgt.GoogleWTS):
            """Esri World Imagery satellite tile source (free, no API key)."""
            def _image_url(self, tile):
                x, y, z = tile
                return (
                    "https://server.arcgisonline.com/ArcGIS/rest/services"
                    f"/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                )

        # Pick zoom level from extent — wider area = lower zoom
        lon_span = lon_max - lon_min
        zoom = max(4, min(10, int(math.log2(360 / lon_span)) + 2))

        imagery = EsriSatellite()

        bg_fig, bg_ax = plt.subplots(
            figsize=(FIG_W, FIG_H),
            subplot_kw={"projection": ccrs.PlateCarree()},
            facecolor="#000000",
        )
        bg_ax.set_extent(extent, crs=ccrs.PlateCarree())
        bg_ax.add_image(imagery, zoom)

        # Transparent label overlay — country names, cities, borders (no API key needed)
        class EsriLabels(cimgt.GoogleWTS):
            def _image_url(self, tile):
                x, y, z = tile
                return (
                    "https://server.arcgisonline.com/ArcGIS/rest/services"
                    f"/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                )

        bg_ax.add_image(EsriLabels(), zoom)

        # Thin coastline outline for extra clarity
        bg_ax.add_feature(cfeature.COASTLINE, edgecolor=(1, 1, 1, 0.4), linewidth=0.5)

        gl = bg_ax.gridlines(draw_labels=True, linewidth=0.2, color="white",
                             alpha=0.3, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"color": "white", "fontsize": 7, "fontfamily": "monospace"}
        gl.ylabel_style = {"color": "white", "fontsize": 7, "fontfamily": "monospace"}
        bg_fig.patch.set_facecolor("#000000")

        buf = io.BytesIO()
        bg_fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                       facecolor="#060F1C", pad_inches=0.05)
        buf.seek(0)
        basemap_arr = np.array(Image.open(buf).convert("RGB"))
        plt.close(bg_fig)

        img_h, img_w = basemap_arr.shape[:2]

        # ── B: Build the reusable animation figure ───────────────────────────
        _cb(0.12, "Building animation canvas …")

        fig, ax = plt.subplots(
            figsize=(img_w / DPI, img_h / DPI),
            subplot_kw={"projection": ccrs.PlateCarree()},
            facecolor="#000000",
        )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor("#000000")
        fig.patch.set_facecolor("#000000")

        # Show pre-rendered basemap as a static imshow layer
        ax.imshow(
            basemap_arr,
            origin="upper",
            extent=extent,
            transform=ccrs.PlateCarree(),
            aspect="auto",
            zorder=0,
            interpolation="bilinear",
        )

        # Dynamic artists (data updated each frame)
        (line1,) = ax.plot([], [], color="#FFD700", lw=2.5, alpha=0.95,
                           transform=ccrs.PlateCarree(), zorder=2)
        (line2,) = ax.plot([], [], color="#FF4500", lw=2.5, alpha=0.95,
                           transform=ccrs.PlateCarree(), zorder=2)

        # Directional arrows — removed and recreated each frame to show heading
        arrow1 = [None]
        arrow2 = [None]

        def _make_arrow(x, y, dx, dy, color):
            """Quiver arrow at (x,y) pointing in direction (dx,dy), normalised."""
            mag = np.sqrt(dx ** 2 + dy ** 2)
            if mag > 1e-9:
                dx, dy = dx / mag, dy / mag
            else:
                dx, dy = 0.0, 1.0          # point north when vessel is stationary
            arrow_len = (lon_max - lon_min) * 0.025
            return ax.quiver(
                x, y, dx * arrow_len, dy * arrow_len,
                color=color,
                transform=ccrs.PlateCarree(),
                scale=1, scale_units="xy",
                width=0.003,
                headwidth=5, headlength=6, headaxislength=4.5,
                zorder=5,
            )

        # Distance line between vessels
        (dist_line,) = ax.plot([], [], color="#EF4444", lw=1.2, linestyle="--",
                               alpha=0.65, transform=ccrs.PlateCarree(), zorder=3)

        # Info overlay
        info = ax.text(
            0.015, 0.97, "",
            transform=ax.transAxes, fontsize=8.5, color="#E2E8F0",
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#000000",
                      edgecolor=(1, 1, 1, 0.3), alpha=0.75),
            zorder=5,
        )

        # Title
        ax.set_title(
            "VESSEL TRACK ANALYSIS  //  TRANSSHIPMENT DETECTION",
            color="#7DD3FC", fontsize=9, fontfamily="monospace",
            pad=8, loc="left",
        )

        # Legend
        legend_handles = [
            Line2D([0], [0], color="#FFD700", lw=2, label="Vessel 1 Track"),
            Line2D([0], [0], color="#FF4500", lw=2, label="Vessel 2 Track"),
            Line2D([0], [0], marker=">", color="#FFD700", ms=9,
                   linestyle="None", label="V1 Heading"),
            Line2D([0], [0], marker=">", color="#FF4500", ms=9,
                   linestyle="None", label="V2 Heading"),
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  facecolor="#000000", edgecolor=(1, 1, 1, 0.3),
                  labelcolor="#FFFFFF", fontsize=8, framealpha=0.75)

        fig.tight_layout(pad=0.4)

        # Pre-extract numpy arrays — avoids repeated .iloc[] overhead
        v1_lons = v1["longitude"].to_numpy()
        v1_lats = v1["latitude"].to_numpy()
        v2_lons = v2["longitude"].to_numpy()
        v2_lats = v2["latitude"].to_numpy()
        v1_times = v1["dt_pos_utc"].dt.strftime("%Y-%m-%d %H:%M UTC").to_numpy()

        total_frames = max(len(v1), len(v2))

        # ── C: Write frames using buffer_rgba (no file I/O per frame) ────────
        _cb(0.18, "Writing frames …")

        _, ext = os.path.splitext(output_path)
        writer_kw: dict = {}
        if ext.lower() == ".mp4":
            writer_kw = {"codec": "libx264", "macro_block_size": None,
                         "ffmpeg_params": ["-crf", "22", "-preset", "fast"]}
        elif ext.lower() == ".gif":
            writer_kw = {"loop": 0}

        with imageio.get_writer(output_path, fps=fps, **writer_kw) as writer:
            for frame in range(total_frames):

                # Vessel 1
                end1 = min(frame + 1, len(v1_lons))
                line1.set_data(v1_lons[:end1], v1_lats[:end1])

                # Vessel 2
                end2 = min(frame + 1, len(v2_lons))
                line2.set_data(v2_lons[:end2], v2_lats[:end2])

                # Directional arrows — remove old, draw new pointing in heading direction
                if arrow1[0] is not None:
                    arrow1[0].remove()
                if arrow2[0] is not None:
                    arrow2[0].remove()

                if frame < len(v1_lons):
                    prev1 = frame - 1 if frame > 0 else 0
                    dx1 = v1_lons[frame] - v1_lons[prev1]
                    dy1 = v1_lats[frame] - v1_lats[prev1]
                    arrow1[0] = _make_arrow(v1_lons[frame], v1_lats[frame],
                                            dx1, dy1, "#FFD700")
                else:
                    arrow1[0] = None

                if frame < len(v2_lons):
                    prev2 = frame - 1 if frame > 0 else 0
                    dx2 = v2_lons[frame] - v2_lons[prev2]
                    dy2 = v2_lats[frame] - v2_lats[prev2]
                    arrow2[0] = _make_arrow(v2_lons[frame], v2_lats[frame],
                                            dx2, dy2, "#FF4500")
                else:
                    arrow2[0] = None

                # Dashed line between current positions
                if frame < len(v1_lons) and frame < len(v2_lons):
                    dist_line.set_data(
                        [v1_lons[frame], v2_lons[frame]],
                        [v1_lats[frame], v2_lats[frame]],
                    )

                # Info text
                dist_str = (
                    f"{distances[frame]:.1f} km"
                    if frame < len(distances) else "—"
                )
                ts_str = v1_times[frame] if frame < len(v1_times) else ""
                info.set_text(f"⏱  {ts_str}\n📍 Separation: {dist_str}")

                # Capture frame — zero file I/O
                fig.canvas.draw()
                rgba_buf  = fig.canvas.buffer_rgba()
                rgb_frame = np.asarray(rgba_buf)[:, :, :3]

                # libx264 requires width & height to be divisible by 2
                h, w = rgb_frame.shape[:2]
                rgb_frame = rgb_frame[: h - h % 2, : w - w % 2]

                writer.append_data(rgb_frame)

                if frame % 20 == 0:
                    _cb(0.18 + 0.80 * frame / total_frames,
                        f"Frame {frame + 1:,} / {total_frames:,}")

        plt.close(fig)
        _cb(1.0, "Complete!")
        return True

    except Exception as exc:
        _cb(0.0, f"Error: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# AI analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_vessel_metrics(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    distances: list[float],
) -> dict:
    """
    Compute a rich set of statistics from the vessel data to feed into the AI.
    Avoids sending raw rows — summarises into interpretable metrics instead.
    """
    metrics: dict = {}

    # ── Basic track info ─────────────────────────────────────────────────────
    metrics["v1_points"]   = len(v1)
    metrics["v2_points"]   = len(v2)
    metrics["v1_start"]    = str(v1["dt_pos_utc"].iat[0])
    metrics["v1_end"]      = str(v1["dt_pos_utc"].iat[-1])
    metrics["v2_start"]    = str(v2["dt_pos_utc"].iat[0])
    metrics["v2_end"]      = str(v2["dt_pos_utc"].iat[-1])
    v1_dur = v1["dt_pos_utc"].iat[-1] - v1["dt_pos_utc"].iat[0]
    v2_dur = v2["dt_pos_utc"].iat[-1] - v2["dt_pos_utc"].iat[0]
    metrics["v1_duration_hours"] = round(v1_dur.total_seconds() / 3600, 1)
    metrics["v2_duration_hours"] = round(v2_dur.total_seconds() / 3600, 1)

    # ── Geographic extent ────────────────────────────────────────────────────
    metrics["v1_lat_range"] = [round(v1["latitude"].min(), 4),  round(v1["latitude"].max(), 4)]
    metrics["v1_lon_range"] = [round(v1["longitude"].min(), 4), round(v1["longitude"].max(), 4)]
    metrics["v2_lat_range"] = [round(v2["latitude"].min(), 4),  round(v2["latitude"].max(), 4)]
    metrics["v2_lon_range"] = [round(v2["longitude"].min(), 4), round(v2["longitude"].max(), 4)]

    # ── Distance statistics ──────────────────────────────────────────────────
    if distances:
        dists_arr = np.array(distances)
        metrics["dist_min_km"]  = round(float(dists_arr.min()), 3)
        metrics["dist_max_km"]  = round(float(dists_arr.max()), 2)
        metrics["dist_mean_km"] = round(float(dists_arr.mean()), 2)
        metrics["dist_std_km"]  = round(float(dists_arr.std()), 2)

        min_idx = int(dists_arr.argmin())
        metrics["closest_approach_time"] = str(v1["dt_pos_utc"].iat[min_idx])
        metrics["closest_approach_lat"]  = round(float(v1["latitude"].iat[min_idx]), 4)
        metrics["closest_approach_lon"]  = round(float(v1["longitude"].iat[min_idx]), 4)

        # Proximity windows — how many minutes spent within thresholds
        resample_mins = (
            v1["dt_pos_utc"].iat[1] - v1["dt_pos_utc"].iat[0]
        ).total_seconds() / 60 if len(v1) > 1 else 5

        def _minutes_within(threshold_km: float) -> float:
            n = int((dists_arr < threshold_km).sum())
            return round(n * resample_mins, 1)

        metrics["minutes_within_500m"] = _minutes_within(0.5)
        metrics["minutes_within_1km"]  = _minutes_within(1.0)
        metrics["minutes_within_5km"]  = _minutes_within(5.0)

    # ── AIS dark periods (gaps > 2 h) ────────────────────────────────────────
    def _dark_periods(df: pd.DataFrame, gap_hours: float = 2.0) -> list[dict]:
        times = df["dt_pos_utc"].sort_values().reset_index(drop=True)
        gaps  = times.diff().dropna()
        dark  = gaps[gaps > pd.Timedelta(hours=gap_hours)]
        result = []
        for idx in dark.index:
            result.append({
                "start": str(times.iat[idx - 1]),
                "end":   str(times.iat[idx]),
                "gap_hours": round(dark[idx].total_seconds() / 3600, 2),
            })
        return result

    metrics["v1_dark_periods"] = _dark_periods(v1)
    metrics["v2_dark_periods"] = _dark_periods(v2)

    # ── Speed estimates (km/h between consecutive points) ────────────────────
    def _speed_stats(df: pd.DataFrame) -> dict:
        lats  = df["latitude"].to_numpy()
        lons  = df["longitude"].to_numpy()
        times = df["dt_pos_utc"].to_numpy()
        speeds = []
        for i in range(1, len(df)):
            dt_h = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds() / 3600
            if dt_h > 0:
                d_km = geodesic((lats[i - 1], lons[i - 1]), (lats[i], lons[i])).kilometers
                speeds.append(d_km / dt_h)
        if not speeds:
            return {}
        spd = np.array(speeds)
        return {
            "mean_knots":  round(float(spd.mean()) * 0.5399568, 1),
            "max_knots":   round(float(spd.max())  * 0.5399568, 1),
            "pct_stationary": round(float((spd < 0.5).mean()) * 100, 1),  # < ~1 knot
        }

    metrics["v1_speed"] = _speed_stats(v1)
    metrics["v2_speed"] = _speed_stats(v2)

    return metrics


def build_analysis_prompt(metrics: dict, vessel1_name: str, vessel2_name: str) -> str:
    """Build the analyst prompt sent to Claude."""
    import json
    return f"""You are an expert maritime analyst specialising in IUU (Illegal, Unreported, Unregulated) fishing detection and vessel transshipment analysis.

A user has uploaded AIS tracking data for two vessels and needs your professional assessment.

Vessel names provided by the user:
- Vessel 1: {vessel1_name}
- Vessel 2: {vessel2_name}

Computed metrics from the AIS data:
{json.dumps(metrics, indent=2)}

Please provide a structured analysis covering:

1. **Transshipment Risk Assessment** — Based on proximity events, time spent close together, and location, rate the transshipment risk as LOW / MEDIUM / HIGH / CRITICAL and explain your reasoning.

2. **Suspicious Events Timeline** — List specific timestamps or windows that warrant attention, with brief explanations of why each is suspicious.

3. **AIS Dark Period Analysis** — Assess any gaps in AIS transmission. Dark periods can indicate deliberate signal suppression to hide activities.

4. **Vessel Behaviour Patterns** — Comment on speed profiles, loitering, and movement patterns for each vessel that may indicate fishing activity, waiting, or rendezvous behaviour.

5. **Closest Approach Analysis** — Evaluate the significance of the closest approach event in the context of potential transshipment.

6. **Recommendations** — Concrete next steps for investigators or fisheries authorities.

Be concise, professional, and evidence-based. Reference specific timestamps and distances from the data. Avoid vague language — use the numbers.
"""


def stream_ai_analysis(
    api_key: str,
    metrics: dict,
    vessel1_name: str,
    vessel2_name: str,
):
    """Stream Groq (Llama 3.1 70B) analysis token-by-token into a Streamlit container."""
    client = Groq(api_key=api_key)
    prompt = build_analysis_prompt(metrics, vessel1_name, vessel2_name)
    output = st.empty()
    full   = ""

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        full += text
        output.markdown(full + "▌")   # blinking cursor effect

    output.markdown(full)   # final render without cursor
    return full


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🚢 Vessel Track Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Upload two AIS datasets to visualize vessel movement '
    'and detect potential transshipment events.</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂Data Upload")

    file1 = st.file_uploader("Vessel 1 CSV", type=["csv"], key="upload_v1")
    file2 = st.file_uploader("Vessel 2 CSV", type=["csv"], key="upload_v2")

    st.markdown("---")
    st.markdown("## ⚙️ Settings")

    resample_label = st.selectbox(
        "Resampling Interval",
        list(RESAMPLE_MAP.keys()),
        index=1,
        help="Coarser intervals = fewer frames = faster video generation.",
    )

    fps = st.slider("Frame Rate (fps)", min_value=2, max_value=30, value=10)

    output_format = st.radio("Output Format", ["mp4", "gif"], horizontal=True)

    st.markdown("---")
    st.markdown("### 🗺️ Map Style")
    basemap_key = st.selectbox(
        "Interactive Map Basemap",
        options=list(BASEMAPS.keys()),
        index=0,
        help="Choose the background map style for the interactive vessel track view.",
    )

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.caption("• Use **15 min** or **30 min** intervals for long tracks.")
    st.caption("• **MP4** is fastest to generate and smallest file size.")
    st.caption("• **GIF** is great for quick sharing but larger.")
    st.caption("• Higher fps → smoother video, bigger file.")

    st.markdown("---")
    st.markdown("### 📋 Required CSV columns")
    st.code("dt_pos_utc\nlongitude\nlatitude", language="text")

    st.markdown("---")
    st.markdown("### 🤖 AI Analysis")
    st.caption("Powered by Groq · Llama 3.3 70B")

# ── Resolve Groq API key (secrets → env var) ─────────────────────────────────
def _get_groq_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GROQ_API_KEY", "")

anthropic_api_key = _get_groq_key()

# ── Main area ─────────────────────────────────────────────────────────────────
if not file1 or not file2:

    # ── How it works ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1️⃣  Upload")
        st.markdown("Upload two AIS CSV files — one per vessel — using the sidebar.")
    with col2:
        st.markdown("#### 2️⃣  Explore")
        st.markdown("An interactive map shows tracks, distances, and closest approach point.")
    with col3:
        st.markdown("#### 3️⃣  Export")
        st.markdown("Generate an MP4 or GIF animation and download it instantly.")
    with col4:
        st.markdown("#### 4️⃣  Analyse")
        st.markdown("Run AI-powered transshipment risk assessment on the vessel data.")

    st.markdown("---")

    # ── Sample data loader ────────────────────────────────────────────────────
    st.markdown("#### 🧪  Try with sample data")
    st.caption(
        "No AIS data yet? Download these two demo CSV files — two vessels meeting "
        "in the Gulf of Guinea with a simulated transshipment event — then upload "
        "them via the sidebar."
    )

    # Generate once and always show — avoids Streamlit button rerun collapse
    _v1_bytes, _v2_bytes = generate_sample_csv()
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label="⬇️  Download Sample Vessel 1 CSV",
        data=_v1_bytes,
        file_name="sample_vessel1.csv",
        mime="text/csv",
        key="dl_sample_v1",
    )
    dl2.download_button(
        label="⬇️  Download Sample Vessel 2 CSV",
        data=_v2_bytes,
        file_name="sample_vessel2.csv",
        mime="text/csv",
        key="dl_sample_v2",
    )

    st.markdown("---")

    # ── CSV format guide ──────────────────────────────────────────────────────
    st.markdown("#### 📋  Required CSV format")
    st.markdown(
        "Your AIS export must contain these three columns "
        "(extra columns are ignored):"
    )
    st.code(
        "dt_pos_utc,longitude,latitude\n"
        "2024-12-08 00:00:00,7.102,4.201\n"
        "2024-12-08 00:05:00,7.104,4.203\n"
        "2024-12-08 00:10:00,7.107,4.206\n"
        "...",
        language="csv",
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("**`dt_pos_utc`** — UTC datetime of the position fix")
    col_b.markdown("**`longitude`** — decimal degrees (WGS-84), e.g. `7.102`")
    col_c.markdown("**`latitude`** — decimal degrees (WGS-84), e.g. `4.201`")

    st.markdown("---")

    # ── What is transshipment ─────────────────────────────────────────────────
    with st.expander("ℹ️  What is transshipment and why does it matter?"):
        st.markdown("""
**Transshipment** is the transfer of catch between vessels at sea — typically from a
fishing vessel to a refrigerated cargo vessel (reefer). While not always illegal,
unmonitored transshipment is a major vector for IUU (Illegal, Unreported and
Unregulated) fishing because it allows vessels to stay at sea longer and obscure
the origin of their catch.

**What this tool detects:**
- Vessels coming within close proximity (< 500 m) for extended periods
- AIS signal gaps (dark periods) that may indicate deliberate transponder shutdown
- Loitering behaviour — a vessel holding position waiting for a rendezvous
- Speed anomalies that suggest cargo transfer operations

**Risk ratings** produced by the AI analysis:
| Rating | Meaning |
|---|---|
| 🟢 LOW | Normal vessel interaction, no suspicious indicators |
| 🟡 MEDIUM | Some proximity or AIS gaps — warrants monitoring |
| 🟠 HIGH | Multiple indicators present — recommend investigation |
| 🔴 CRITICAL | Strong evidence of transshipment — immediate action advised |
        """)

    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    file1_bytes = file1.read()
    file2_bytes = file2.read()

    with st.spinner("Loading vessel data …"):
        v1 = load_vessel_data(file1_bytes, file1.name, resample_label)
        v2 = load_vessel_data(file2_bytes, file2.name, resample_label)
        distances = compute_distances(file1_bytes, file2_bytes, resample_label)

except ValueError as e:
    st.error(str(e), icon="⚠️")
    st.markdown(
        "**Need help?** Check the required CSV format in the sidebar "
        "or load the sample data on the welcome screen."
    )
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading files: {e}")
    st.caption("If this keeps happening, try re-exporting your AIS data or contact support.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d, col_e = st.columns(5)

v1_duration = v1["dt_pos_utc"].max() - v1["dt_pos_utc"].min()
min_dist     = min(distances) if distances else float("nan")
min_idx      = int(np.argmin(distances)) if distances else 0

col_a.metric("Vessel 1 Points", f"{len(v1):,}")
col_b.metric("Vessel 2 Points", f"{len(v2):,}")
col_c.metric("Closest Approach", f"{min_dist:.2f} km")
col_d.metric("Track Duration", f"{v1_duration.days}d {v1_duration.seconds // 3600}h")
col_e.metric("Total Frames", f"{max(len(v1), len(v2)):,}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
# Vessel name inputs for AI analysis
name_col1, name_col2 = st.columns(2)
vessel1_name = name_col1.text_input("Vessel 1 name / MMSI", placeholder="e.g. MV Atlantic Star")
vessel2_name = name_col2.text_input("Vessel 2 name / MMSI", placeholder="e.g. MV Pacific Dawn")

tab_map, tab_chart, tab_video, tab_ai, tab_help = st.tabs([
    "🗺️  Interactive Map",
    "📈  Distance Chart",
    "🎬  Generate Video",
    "🤖  AI Analysis",
    "❓  Help",
])

# ── Tab 1: Folium map ─────────────────────────────────────────────────────────
with tab_map:
    st.markdown("### Vessel Tracks — Interactive Map")
    st.caption(
        "🔵 Vessel 1 (blue)  ·  🟠 Vessel 2 (orange)  ·  "
        "🔴 Closest approach (red ring)  ·  Scroll to zoom, click markers for detail."
    )

    fmap = build_folium_map(v1, v2, distances, basemap_key)
    st_folium(fmap, width=None, height=560, returned_objects=[])

# ── Tab 2: Distance chart ─────────────────────────────────────────────────────
with tab_chart:
    st.markdown("### Distance Between Vessels Over Time")

    if distances:
        dist_df = pd.DataFrame({
            "Time": v1["dt_pos_utc"].iloc[: len(distances)],
            "Distance (km)": distances,
        }).set_index("Time")

        # Annotate closest approach
        ca_time = v1["dt_pos_utc"].iat[min_idx]
        st.line_chart(dist_df, color="#FFD700", height=380) # Gold color
        #st.line_chart(dist_df, color="#4ADE80", height=380) # Emerald green
        #st.line_chart(dist_df, color="#F87171", height=380) # coral red

        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Minimum distance", f"{min_dist:.2f} km")
        col_y.metric("Maximum distance", f"{max(distances):.2f} km")
        col_z.metric("Mean distance",    f"{np.mean(distances):.2f} km")

        #st.caption(f"⚠️  Closest approach at **{ca_time}**")
        ca_time_str = pd.Timestamp(ca_time).strftime("%Y-%m-%d %H:%M UTC")
        st.markdown(
            f'<p style="font-size:1.5rem; color:#FFFFFF; margin-top:0.5rem;">'
            f'⚠️  Closest approach at <b>{ca_time_str}</b></p>',
            unsafe_allow_html=True
        )
    else:
        st.warning("No overlapping timestamps found to compute distances.")

# ── Tab 3: Video generation ───────────────────────────────────────────────────
with tab_video:
    st.markdown("### Generate Animation Video")

    total_frames   = max(len(v1), len(v2))
    est_duration_s = total_frames / fps

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Frames to render", f"{total_frames:,}")
    info_col2.metric("Video duration",   f"{est_duration_s:.0f}s  ({est_duration_s / 60:.1f} min)")
    info_col3.metric("Format",           output_format.upper())

    st.markdown("")

    # Time estimate
    secs_per_frame = 0.25   # conservative estimate including satellite tile render
    est_secs = total_frames * secs_per_frame
    if est_secs < 60:
        est_str = f"~{est_secs:.0f} seconds"
    else:
        est_str = f"~{est_secs/60:.0f} minutes"

    if total_frames > 500:
        st.warning(
            f"⏱  **Estimated generation time: {est_str}** for {total_frames:,} frames.  \n"
            f"Consider switching to **15 min** or **30 min** resampling in the sidebar "
            f"to reduce frames and speed up generation."
        )
    else:
        st.info(f"⏱  Estimated generation time: {est_str} for {total_frames:,} frames.")

    if st.button(f"🎬  Generate {output_format.upper()} Video"):

        progress_bar  = st.progress(0.0)
        status_text   = st.empty()

        def _update(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(f"▸  {msg}")

        suffix = f".{output_format}"
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            t0 = time.time()
            generate_optimized_video(
                v1, v2, distances, tmp_path,
                fps=fps,
                progress_callback=_update,
            )
            elapsed = time.time() - t0

            status_text.success(f"✅  Video generated in {elapsed:.1f}s")
            progress_bar.progress(1.0)

            with open(tmp_path, "rb") as fh:
                video_bytes = fh.read()

            mime = "video/mp4" if output_format == "mp4" else "image/gif"
            st.download_button(
                label=f"⬇️  Download {output_format.upper()}",
                data=video_bytes,
                file_name=f"vessel_track.{output_format}",
                mime=mime,
            )

            if output_format == "gif":
                st.image(video_bytes, caption="Preview (GIF)")

        except Exception as e:
            status_text.error(f"❌  Error: {e}")
            st.exception(e)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ── Tab 5: Help ──────────────────────────────────────────────────────────────
with tab_help:
    st.markdown("### ❓ Help & Documentation")

    with st.expander("📡  What is AIS data?", expanded=True):
        st.markdown("""
The **Automatic Identification System (AIS)** is a tracking system used on ships.
Vessels broadcast their position, speed, heading and identity every few seconds via VHF radio.
These signals are picked up by coastal stations and satellites and stored as CSV or database records.

Your AIS CSV should have at minimum:
- **`dt_pos_utc`** — timestamp of the position fix in UTC
- **`latitude`** — decimal degrees, e.g. `4.201` (positive = North)
- **`longitude`** — decimal degrees, e.g. `7.102` (positive = East)
        """)

    with st.expander("🔄  What is transshipment?"):
        st.markdown("""
**Transshipment** is the transfer of fish or cargo between vessels at sea.
A fishing vessel offloads its catch to a refrigerated cargo vessel (reefer),
allowing it to stay fishing without returning to port.

While legal when reported, unmonitored transshipment is a major pathway for
**IUU (Illegal, Unreported and Unregulated) fishing** because it:
- Obscures the origin and volume of the catch
- Allows vessels to operate in restricted zones and transfer catch undetected
- Bypasses port inspections

This tool helps identify potential transshipment events by detecting when two
vessels come into very close proximity, particularly when combined with AIS dark periods.
        """)

    with st.expander("⚙️  Settings guide"):
        st.markdown("""
| Setting | What it does | Recommendation |
|---|---|---|
| **Resampling interval** | How frequently positions are sampled | 5–15 min for most tracks |
| **Frame rate (fps)** | Smoothness of the generated video | 10 fps is a good balance |
| **Output format** | MP4 for sharing, GIF for embedding | MP4 recommended |
| **Basemap** | Background map style | Nautical for maritime work |

**Resampling interval affects speed:**
A 3-day track at 5-min intervals = ~864 frames.
The same track at 30-min intervals = ~144 frames — 6× faster to generate.
        """)

    with st.expander("🤖  Understanding the AI risk ratings"):
        st.markdown("""
The AI analysis uses Llama 3.3 70B (via Groq) to assess transshipment risk
based on computed metrics from your AIS data. Risk ratings mean:

| Rating | Meaning | Suggested action |
|---|---|---|
| 🟢 **LOW** | Normal vessel interaction | No action required |
| 🟡 **MEDIUM** | Some proximity or AIS gaps | Monitor and log |
| 🟠 **HIGH** | Multiple indicators present | Flag for investigation |
| 🔴 **CRITICAL** | Strong evidence of transshipment | Immediate action / report |

The AI considers: proximity duration, AIS dark periods, speed profiles,
loitering behaviour, and the geographic context of the closest approach.
        """)

    with st.expander("📋  CSV format & troubleshooting"):
        st.markdown("**Required columns:**")
        st.code("dt_pos_utc,longitude,latitude", language="text")
        st.markdown("""
**Common errors and fixes:**

| Error | Cause | Fix |
|---|---|---|
| Missing columns | Column names don't match | Rename columns in your export |
| No valid position data | All coordinates are blank/null | Check your AIS export settings |
| File too large | CSV exceeds 50 MB | Export a shorter date range |
| Too many rows | Over 100,000 rows | Use a coarser resample or shorter range |
| Datetime parse error | Wrong timestamp format | Ensure format is `YYYY-MM-DD HH:MM:SS` |
        """)

    with st.expander("🎬  Video generation tips"):
        st.markdown("""
- **Satellite basemap** requires internet to fetch tiles and adds ~10–20s to generation time
- For long tracks (> 2 days) use **30 min** resampling to keep generation under 3 minutes
- **MP4** encodes faster and produces smaller files than GIF
- If video generation fails, try switching from Satellite to Dark or Nautical basemap
- The progress bar updates every 20 frames — it is working even if it looks slow
        """)

    st.markdown("---")
    st.markdown(
        "Built with Streamlit · Folium · Cartopy · Groq — For support or feedback, contact your system administrator."
    )


# ── Tab 4: AI Analysis ────────────────────────────────────────────────────────
with tab_ai:
    st.markdown("### 🤖 AI-Powered Transshipment Analysis")
    st.caption(
        "Claude analyses proximity events, dark periods, speed profiles and movement "
        "patterns to assess transshipment risk and flag suspicious behaviour."
    )

    if not anthropic_api_key:
        st.error(
            "Groq API key not configured. Add it to `.streamlit/secrets.toml`:  \n"
            "```\nGROQ_API_KEY = \"gsk_your_key_here\"\n```  \n"
            "Get a free key at [console.groq.com](https://console.groq.com)."
        )
    else:
        # Pre-compute and display key metrics as a preview
        metrics = compute_vessel_metrics(v1, v2, distances)

        with st.expander("📊  Data summary sent to AI", expanded=False):
            ma, mb, mc, md = st.columns(4)
            ma.metric("Minutes within 500 m", metrics.get("minutes_within_500m", "—"))
            mb.metric("Minutes within 1 km", metrics.get("minutes_within_1km", "—"))
            mc.metric("V1 dark periods", len(metrics.get("v1_dark_periods", [])))
            md.metric("V2 dark periods", len(metrics.get("v2_dark_periods", [])))

            sp1 = metrics.get("v1_speed", {})
            sp2 = metrics.get("v2_speed", {})
            se, sf, sg, sh = st.columns(4)
            se.metric("V1 avg speed", f"{sp1.get('mean_knots', '—')} kn")
            sf.metric("V1 % stationary", f"{sp1.get('pct_stationary', '—')} %")
            sg.metric("V2 avg speed", f"{sp2.get('mean_knots', '—')} kn")
            sh.metric("V2 % stationary", f"{sp2.get('pct_stationary', '—')} %")

            if metrics.get("v1_dark_periods"):
                st.markdown("**Vessel 1 — AIS dark periods**")
                st.dataframe(
                    pd.DataFrame(metrics["v1_dark_periods"]),
                    use_container_width=True, hide_index=True,
                )
            if metrics.get("v2_dark_periods"):
                st.markdown("**Vessel 2 — AIS dark periods**")
                st.dataframe(
                    pd.DataFrame(metrics["v2_dark_periods"]),
                    use_container_width=True, hide_index=True,
                )

        st.markdown("")

        # Session state to persist the report across reruns
        if "ai_report" not in st.session_state:
            st.session_state["ai_report"] = ""

        run_col, clear_col = st.columns([3, 1])
        run_btn = run_col.button("🔍  Run AI Analysis", type="primary")
        clear_btn = clear_col.button("🗑  Clear")

        if clear_btn:
            st.session_state["ai_report"] = ""

        if run_btn:
            v1_label = vessel1_name or "Vessel 1"
            v2_label = vessel2_name or "Vessel 2"
            try:
                with st.spinner("Analysing vessel behaviour …"):
                    report = stream_ai_analysis(
                        anthropic_api_key, metrics, v1_label, v2_label
                    )
                st.session_state["ai_report"] = report
            except Exception as e:
                err_str = str(e).lower()
                if "401" in str(e) or "invalid_api_key" in err_str or "authentication" in err_str:
                    st.error("❌  Invalid Groq API key — check your `.streamlit/secrets.toml`.")
                elif "decommissioned" in err_str or "model" in err_str:
                    st.error(f"❌  Model error: {e}. Update the model name in `stream_ai_analysis()`.")
                else:
                    st.error(f"❌  Analysis failed: {e}")
                    st.exception(e)

        elif st.session_state["ai_report"]:
            # Show cached report
            st.markdown(st.session_state["ai_report"])

            # Download report as text
            st.download_button(
                label="⬇️  Download Report",
                data=st.session_state["ai_report"],
                file_name="transshipment_analysis.md",
                mime="text/markdown",
            )