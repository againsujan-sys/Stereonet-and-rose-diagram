"""Structural Geology Stereonet & Rose Diagram Generator.

Run from the project directory with:
    streamlit run app.py

Coordinate and structural conventions
------------------------------------
- Azimuths are degrees clockwise from north.
- Planes are stored as RIGHT-HAND-RULE strike / dip (dip direction = strike + 90).
- Linear data are trend / downward plunge, in degrees.
- Plotted lines and poles use the LOWER hemisphere.
- Fold-axis (pi/beta estimate) is the MINIMUM eigenvector of the orientation
  tensor of bedding POLES. The principal/cluster pole is the MAXIMUM eigenvector.

The application is intentionally kept in one self-contained app.py, but the data
validation, structural calculations, plotting, and Streamlit UI are separate
functions to make the code straightforward to present or extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless, safe for Streamlit / servers.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplstereonet
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = APP_DIR / "sample_field_data.csv"
BASIC_COLUMNS = ("Station_ID", "Structure_Type", "Dip", "Trend", "Plunge")
COLORS = {
    "bedding": "#1976A3",
    "joint": "#D88938",
    "other_plane": "#676A9C",
    "line": "#7A50A2",
    "axis": "#C62828",
    "principal": "#14845B",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. File loading and strict, user-friendly validation
# ─────────────────────────────────────────────────────────────────────────────


def read_measurements(source: str | Path | BytesIO, name: str) -> pd.DataFrame:
    """Read CSV or .xlsx. Caller catches exceptions and displays UI error."""
    extension = Path(name).suffix.lower()
    if extension == ".csv":
        return pd.read_csv(source)
    if extension in {".xlsx", ".xlsm"}:
        return pd.read_excel(source, engine="openpyxl")
    raise ValueError("Unsupported file type. Please provide a CSV or .xlsx Excel file.")


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Treat blanks as missing but reject text such as 'NE' or '45 degrees'."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    raw = frame[column]
    stripped = raw.astype("string").str.strip()
    empty = raw.isna() | stripped.eq("").fillna(False)
    numbers = pd.to_numeric(raw.where(~empty), errors="coerce")
    bad = (~empty) & numbers.isna()
    if bad.any():
        sample_rows = (np.flatnonzero(bad.to_numpy())[:6] + 2).tolist()
        raise ValueError(
            f"Non-numeric value in '{column}' at file row(s) {sample_rows}. "
            "Use decimal degrees and leave inapplicable cells blank."
        )
    return numbers.astype(float)


def _check_range(values: pd.Series, label: str, low: float, high: float) -> None:
    bad = values.notna() & (~np.isfinite(values) | (values < low) | (values > high))
    if bad.any():
        rows = (np.flatnonzero(bad.to_numpy())[:6] + 2).tolist()
        raise ValueError(
            f"Invalid {label} at file row(s) {rows}: expected {low}°–{high}°. "
            "Strike/trend of 360° is accepted as 0°."
        )


def prepare_measurements(raw: pd.DataFrame, convention: str) -> pd.DataFrame:
    """Validate rows and convert plane orientations to canonical RHR.

    In 'Dip Direction / Dip' mode, Dip_Direction is preferred when present.
    Otherwise, the legacy Strike column is interpreted as DIP DIRECTION.
    Lineations use Trend/Plunge unchanged, regardless of plane convention.
    """
    if raw.empty:
        raise ValueError("The uploaded file has no measurement rows.")

    frame = raw.copy().reset_index(drop=True)
    frame.columns = [str(col).strip().replace(" ", "_") for col in frame.columns]

    missing = [col for col in BASIC_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    dipdir_mode = convention == "Dip Direction / Dip"
    if not dipdir_mode and "Strike" not in frame.columns:
        raise ValueError("RHR mode requires a 'Strike' column.")
    if dipdir_mode and not {"Strike", "Dip_Direction"}.intersection(frame.columns):
        raise ValueError(
            "Dip Direction / Dip mode requires 'Dip_Direction' or 'Strike' "
            "(used as dip direction) column."
        )

    for key in ("Station_ID", "Structure_Type"):
        blank = frame[key].isna() | frame[key].astype(str).str.strip().eq("")
        if blank.any():
            rows = (np.flatnonzero(blank.to_numpy())[:6] + 2).tolist()
            raise ValueError(f"'{key}' is missing at file row(s) {rows}.")
        frame[key] = frame[key].astype(str).str.strip()

    dip = _numeric_column(frame, "Dip")
    trend = _numeric_column(frame, "Trend")
    plunge = _numeric_column(frame, "Plunge")
    strike_or_dir = _numeric_column(frame, "Strike")

    if dipdir_mode and "Dip_Direction" in frame.columns:
        preferred_dipdir = _numeric_column(frame, "Dip_Direction")
        azimuth = preferred_dipdir.combine_first(strike_or_dir)
    else:
        azimuth = strike_or_dir

    _check_range(azimuth, "strike or dip direction", 0, 360)
    _check_range(dip, "dip", 0, 90)
    _check_range(trend, "trend", 0, 360)
    _check_range(plunge, "plunge", 0, 90)

    plane_has_azimuth = azimuth.notna()
    plane_has_dip = dip.notna()
    line_has_trend = trend.notna()
    line_has_plunge = plunge.notna()

    plane_incomplete = plane_has_azimuth ^ plane_has_dip
    line_incomplete = line_has_trend ^ line_has_plunge
    if plane_incomplete.any() or line_incomplete.any():
        rows = (
            np.flatnonzero((plane_incomplete | line_incomplete).to_numpy())[:10] + 2
        ).tolist()
        raise ValueError(
            f"Incomplete strike/dip or trend/plunge pair at row(s) {rows}. "
            "A measurement needs both angles of its pair, or neither."
        )

    is_plane = plane_has_azimuth & plane_has_dip
    is_line = line_has_trend & line_has_plunge
    if (~(is_plane | is_line)).any():
        rows = (np.flatnonzero((~(is_plane | is_line)).to_numpy())[:10] + 2).tolist()
        raise ValueError(
            f"No usable plane or line measurement at row(s) {rows}. "
            "Provide Strike/Dip (or Dip_Direction/Dip) and/or Trend/Plunge."
        )

    # A clockwise dip direction is 90° to the RIGHT of RHR strike.
    strike_rhr = ((azimuth - 90) if dipdir_mode else azimuth) % 360
    frame["Strike_RHR"] = strike_rhr.where(is_plane)
    frame["Dip"] = dip
    frame["Dip_Direction"] = ((strike_rhr + 90) % 360).where(is_plane)
    frame["Trend"] = (trend % 360).where(is_line)
    frame["Plunge"] = plunge
    frame["Is_Plane"] = is_plane
    frame["Is_Line"] = is_line
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 2. Structural geology calculations (azimuth clockwise from N, z positive down)
# ─────────────────────────────────────────────────────────────────────────────


def trend_plunge_to_vector(trend: np.ndarray, plunge: np.ndarray) -> np.ndarray:
    """Return normalized N-East-Down unit vectors for geological lines."""
    az = np.deg2rad(np.asarray(trend, dtype=float))
    pl = np.deg2rad(np.asarray(plunge, dtype=float))
    return np.column_stack((
        np.cos(pl) * np.cos(az),
        np.cos(pl) * np.sin(az),
        np.sin(pl),
    ))


def lower_hemisphere_trend_plunge(vector: np.ndarray) -> tuple[float, float]:
    """Turn an undirected eigenvector into a lower-hemisphere axial line."""
    v = np.asarray(vector, dtype=float).copy()
    v = v / np.linalg.norm(v)
    if v[2] < -1e-10 or (abs(v[2]) <= 1e-10 and (v[0] < 0 or (abs(v[0]) <= 1e-10 and v[1] < 0))):
        v *= -1
    horizontal = np.hypot(v[0], v[1])
    trend = float(np.degrees(np.arctan2(v[1], v[0])) % 360) if horizontal > 1e-10 else 0.0
    plunge = float(np.degrees(np.arctan2(max(v[2], 0.0), horizontal)))
    return trend, plunge


def pole_trend_plunge_to_plane(pole_trend: float, pole_plunge: float) -> tuple[float, float]:
    """Lower-hemisphere pole (trend, plunge) -> plane RHR (strike, dip)."""
    return (pole_trend + 90) % 360, 90 - pole_plunge


@dataclass(frozen=True)
class FoldAnalysis:
    fold_trend: float
    fold_plunge: float
    girdle_strike: float
    girdle_dip: float
    principal_trend: float
    principal_plunge: float
    mean_strike: float
    mean_dip: float
    eigenvalues: np.ndarray  # Descending λ1 >= λ2 >= λ3; sum ≈ 1.
    n_bedding: int


def analyze_fold(bedding: pd.DataFrame) -> FoldAnalysis:
    """Fit π-girdle/beta-axis from bedding POLES using axial eigenanalysis.

    Tensor T = Σ(n nᵀ)/N is sign-invariant. Smallest eigenvector is normal
    to best-fit plane through poles (the π-girdle), i.e. the fold-axis estimate.
    Largest eigenvector is principal POLE orientation, not fold axis.
    """
    if len(bedding) < 3:
        raise ValueError("At least 3 bedding planes are required for fold analysis.")

    strike = bedding["Strike_RHR"].to_numpy(dtype=float)
    dip = bedding["Dip"].to_numpy(dtype=float)
    # RHR: dip azimuth = strike+90; DOWNWARD pole points OPPOSITE the dip.
    pole_trend = (strike + 270) % 360
    pole_plunge = 90 - dip
    vectors = trend_plunge_to_vector(pole_trend, pole_plunge)
    tensor = (vectors.T @ vectors) / len(vectors)

    values, eigenvectors = np.linalg.eigh(tensor)  # Ascending by definition.
    ordering = np.argsort(values)[::-1]
    values = values[ordering]
    eigenvectors = eigenvectors[:, ordering]

    beta_trend, beta_plunge = lower_hemisphere_trend_plunge(eigenvectors[:, 2])
    principal_trend, principal_plunge = lower_hemisphere_trend_plunge(eigenvectors[:, 0])
    girdle_strike, girdle_dip = pole_trend_plunge_to_plane(beta_trend, beta_plunge)
    mean_strike, mean_dip = pole_trend_plunge_to_plane(principal_trend, principal_plunge)

    return FoldAnalysis(
        beta_trend, beta_plunge, girdle_strike, girdle_dip,
        principal_trend, principal_plunge, mean_strike, mean_dip,
        values, len(vectors),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plotting routines (independent of the Streamlit layout)
# ─────────────────────────────────────────────────────────────────────────────


def draw_stereonet(
    planes: pd.DataFrame,
    lines: pd.DataFrame,
    density_planes: pd.DataFrame,
    fold: FoldAnalysis | None,
    projection: str = "equal_area",
    show_circles: bool = True,
    show_poles: bool = True,
    show_lines: bool = True,
    show_density: bool = True,
    show_fold: bool = True,
    density_method: str = "exponential_kamb",
    colormap: str = "viridis",
    density_sigma: float = 3.0,
    density_levels: int = 12,
    density_alpha: float = 0.76,
    show_colorbar: bool = True,
    colorbar_label: str = "Pole density above uniform (σ)",
) -> plt.Figure:
    """Draw lower-hemisphere Schmidt or Wulff stereonet."""
    fig, ax = mplstereonet.subplots(figsize=(9, 8), projection=projection)
    fig.patch.set_facecolor("white")

    if show_density and len(density_planes) >= 3:
        contour = ax.density_contourf(
            density_planes["Strike_RHR"].to_numpy(),
            density_planes["Dip"].to_numpy(),
            measurement="poles",
            method=density_method,
            cmap=colormap,
            sigma=density_sigma,
            gridsize=100,
            levels=density_levels,
            alpha=density_alpha,
            zorder=1,
        )
        if show_colorbar:
            bar = fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.11, shrink=0.74)
            bar.set_label(colorbar_label, fontsize=9)
            bar.ax.tick_params(labelsize=8)

    if show_circles and not planes.empty:
        ax.plane(
            planes["Strike_RHR"].to_numpy(),
            planes["Dip"].to_numpy(),
            color="#60687B", alpha=0.26, linewidth=0.75, zorder=2,
        )

    if show_poles and not planes.empty:
        for type_name, group in planes.groupby("Structure_Type", sort=True):
            kind = type_name.casefold()
            col = (COLORS["bedding"] if "bedding" in kind else
                   COLORS["joint"] if "joint" in kind else
                   COLORS["other_plane"])
            ax.pole(
                group["Strike_RHR"].to_numpy(),
                group["Dip"].to_numpy(),
                marker="o", linestyle="None", markersize=5.0,
                markeredgecolor="white", markeredgewidth=0.55,
                color=col, alpha=0.95, label=f"{type_name} (n={len(group)})", zorder=5,
            )

    if show_lines and not lines.empty:
        for type_name, group in lines.groupby("Structure_Type", sort=True):
            ax.line(
                group["Plunge"].to_numpy(),  # mplstereonet order: PLUNGE, BEARING
                group["Trend"].to_numpy(),
                marker="^", linestyle="None", markersize=7,
                markeredgecolor="white", markeredgewidth=0.55,
                color=COLORS["line"], label=f"{type_name} (n={len(group)})", zorder=6,
            )

    if fold is not None and show_fold:
        # Great circle containing the fitted bedding-pole girdle.
        ax.plane(
            fold.girdle_strike, fold.girdle_dip,
            color=COLORS["axis"], linestyle="--", linewidth=2.1,
            zorder=7, label="Best-fit π-girdle",
        )
        ax.line(
            fold.fold_plunge, fold.fold_trend,
            color=COLORS["axis"], marker="*", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8,
            linestyle="None", zorder=9, label="Estimated π/β fold axis",
        )
        ax.line(
            fold.principal_plunge, fold.principal_trend,
            color=COLORS["principal"], marker="D", markersize=7,
            markeredgecolor="white", markeredgewidth=0.7,
            linestyle="None", zorder=8, label="Principal pole (λ₁)",
        )

    ax.grid(True, color="#8F97A5", linewidth=0.65, alpha=0.45)
    title = "Schmidt • equal-area" if projection == "equal_area" else "Wulff • equal-angle"
    ax.set_title(f"Structural stereonet | {title}", fontsize=13, pad=22, fontweight="bold")

    # Use explicit compact legend elements: mplstereonet's ax.line can create
    # one artist per point, which would otherwise make the legend very large.
    handles: list[Line2D] = []
    if show_circles and not planes.empty:
        handles.append(Line2D([0], [0], color="#60687B", lw=1.0, label="Measured planes"))
    if show_poles:
        for type_name in planes["Structure_Type"].unique():
            group = planes.loc[planes["Structure_Type"] == type_name]
            kind = type_name.casefold()
            col = (COLORS["bedding"] if "bedding" in kind else
                   COLORS["joint"] if "joint" in kind else COLORS["other_plane"])
            handles.append(Line2D([0], [0], marker="o", linestyle="None", color=col,
                                  label=f"{type_name}: {len(group)}"))
    if show_lines:
        for type_name in lines["Structure_Type"].unique():
            group = lines.loc[lines["Structure_Type"] == type_name]
            handles.append(Line2D([0], [0], marker="^", linestyle="None",
                                  color=COLORS["line"], label=f"{type_name}: {len(group)}"))
    if show_fold and fold is not None:
        handles.extend([
            Line2D([0], [0], linestyle="--", color=COLORS["axis"], lw=2,
                   label="Fitted π-girdle"),
            Line2D([0], [0], marker="*", linestyle="None", markersize=11,
                   color=COLORS["axis"], label="π/β axis"),
            Line2D([0], [0], marker="D", linestyle="None", markersize=6,
                   color=COLORS["principal"], label="Principal pole (λ₁)"),
        ])
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.012),
                   ncol=2, fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.87, bottom=0.19)
    return fig


def rose_histogram(
    angles_deg: np.ndarray,
    bin_size: int,
    bidirectional: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directional bin counts and edges on 0..360 degrees.

    For axial strikes, each plane contributes to TWO 180°-opposed rays.
    """
    values = np.asarray(angles_deg, dtype=float) % 360
    if bidirectional:
        values = np.concatenate([values % 180, (values % 180) + 180])
    edges = np.arange(0, 360 + bin_size, bin_size, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    return counts, edges


def draw_rose(
    angles_deg: np.ndarray,
    bin_size: int,
    bidirectional: bool,
    title: str,
    fill_color: str = "#216A93",
) -> plt.Figure:
    """Polar rose with north at 0° and clockwise-positive bearings."""
    counts, edges = rose_histogram(angles_deg, bin_size, bidirectional)
    fig, ax = plt.subplots(figsize=(8.2, 7.5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(
        np.deg2rad(edges[:-1]), counts,
        width=np.deg2rad(np.diff(edges)), align="edge",
        color=fill_color, edgecolor="white", linewidth=0.7, alpha=0.85,
    )
    ax.set_thetagrids(np.arange(0, 360, 30), labels=[f"{a}°" for a in range(0, 360, 30)])
    ax.set_ylim(0, max(1, int(counts.max())) * 1.15)
    ax.grid(alpha=0.30)
    ax.set_title(title, va="bottom", fontsize=13, fontweight="bold", pad=25)
    description = ("Axial (bidirectional): each measurement plotted twice"
                   if bidirectional else "Directional (unidirectional): each measurement plotted once")
    fig.text(0.50, 0.045, f"n = {len(angles_deg)} | bin = {bin_size}° | {description}",
             ha="center", fontsize=9, color="#525C67")
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    return fig


def figure_bytes(fig: plt.Figure, fmt: str) -> bytes:
    """Save 300 DPI PNG or vector SVG to a Streamlit download buffer."""
    if fmt not in {"png", "svg"}:
        raise ValueError(f"Unsupported figure format: {fmt}")
    output = BytesIO()
    fig.savefig(output, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
    output.seek(0)
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Streamlit interface only: no structural math here
# ─────────────────────────────────────────────────────────────────────────────


def download_figure_controls(fig: plt.Figure, base_name: str) -> None:
    png_col, svg_col = st.columns(2)
    png_col.download_button(
        "⬇ Export PNG · 300 DPI",
        data=figure_bytes(fig, "png"), file_name=f"{base_name}.png",
        mime="image/png", use_container_width=True, key=f"{base_name}_png",
    )
    svg_col.download_button(
        "⬇ Export SVG · vector",
        data=figure_bytes(fig, "svg"), file_name=f"{base_name}.svg",
        mime="image/svg+xml", use_container_width=True, key=f"{base_name}_svg",
    )


def main() -> None:
    st.set_page_config(
        page_title="Structural Stereonet & Rose Diagram Generator",
        page_icon="🧭", layout="wide",
    )
    st.title("🧭 Structural Geology Studio")
    st.caption("Lower-hemisphere stereonets, bedding-pole eigenanalysis and rose diagrams")

    with st.sidebar:
        st.header("1 · Load field data")
        if "data_source" not in st.session_state:
            st.session_state["data_source"] = "Sample Field Data"
        # Put the button BEFORE the keyed radio: Streamlit does not allow
        # changing a widget key after that widget is instantiated in a run.
        if st.button("Load Sample Field Data", use_container_width=True):
            st.session_state["data_source"] = "Sample Field Data"
        st.radio("Data source", ["Sample Field Data", "Upload CSV / Excel"],
                 key="data_source")

        uploaded = None
        if st.session_state["data_source"] == "Upload CSV / Excel":
            uploaded = st.file_uploader("CSV or .xlsx", type=["csv", "xlsx", "xlsm"])
        convention = st.radio(
            "Plane input convention",
            ["RHR Strike / Dip", "Dip Direction / Dip"],
            help="Dip direction is 90° clockwise from RHR strike. "
                 "In Dip Direction mode use 'Dip_Direction' if present; otherwise "
                 "the Strike column is interpreted as dip direction.",
        )
        st.caption("Sample field data use **RHR Strike/Dip**. "
                   "Switch to Dip Direction mode only for a file recorded that way.")

    try:
        if st.session_state["data_source"] == "Sample Field Data":
            raw = read_measurements(SAMPLE_PATH, SAMPLE_PATH.name)
            st.info("Using the built-in synthetic Himalayan-style fold-limb dataset.")
        elif uploaded is not None:
            raw = read_measurements(uploaded, uploaded.name)
        else:
            st.info("Upload a CSV / Excel file, or press 'Load Sample Field Data'.")
            st.stop()
        data = prepare_measurements(raw, convention)
    except Exception as exc:
        st.error(f"Cannot load measurements: {exc}")
        st.stop()

    all_planes = data.loc[data["Is_Plane"]].copy()
    all_lines = data.loc[data["Is_Line"]].copy()
    plane_types = sorted(all_planes["Structure_Type"].unique().tolist())
    line_types = sorted(all_lines["Structure_Type"].unique().tolist())
    bedding_defaults = [name for name in plane_types if "bedding" in name.casefold()]

    with st.sidebar:
        st.divider()
        st.header("2 · Stereonet")
        chosen_planes = st.multiselect("Show planar structure types", plane_types,
                                      default=plane_types)
        chosen_lines = st.multiselect("Show linear structure types", line_types,
                                     default=line_types)
        projection_name = st.selectbox("Projection", ["Equal-Area (Schmidt)",
                                                       "Equal-Angle (Wulff)"])
        projection = "equal_area" if projection_name.startswith("Equal-Area") else "equal_angle"
        show_great_circles = st.checkbox("Measured great circles", value=True)
        show_poles = st.checkbox("Poles to planes", value=True)
        show_lineations = st.checkbox("Lineations", value=True)
        show_density = st.checkbox("Density contours", value=True)
        density_method_ui = st.selectbox("Density estimation", ["Exponential Kamb", "Kamb"])
        density_method = "exponential_kamb" if density_method_ui == "Exponential Kamb" else "kamb"
        density_map = st.selectbox("Density colormap",
                                   ["viridis", "magma", "plasma", "cividis", "YlOrRd", "inferno"])
        density_sigma = st.slider("Kamb smoothing σ", 1.0, 5.0, 3.0, 0.5)
        density_levels = st.slider("Density contour levels", 5, 24, 12, 1)
        density_alpha = st.slider("Density opacity", 0.3, 1.0, 0.76, 0.05)
        show_colorbar = st.checkbox("Show density colorbar", value=True)
        colorbar_label = st.text_input("Colorbar label", "Pole density above uniform (σ)")
        st.caption("Contour values are significance relative to uniform density (σ).")

        st.divider()
        st.header("3 · Fold analysis")
        fold_types = st.multiselect(
            "Bedding group(s) for fold fit", plane_types, default=bedding_defaults,
            help="Select bedding from BOTH limbs; do not fit a fold using joints.",
        )
        draw_fold = st.checkbox("Show fitted π-girdle, π/β axis and principal pole", value=True)
        density_basis = st.radio("Contour data", ["Fold-fit bedding", "All visible planes"],
                                 help="Keep separate from fold fitting; mixed joint populations "
                                      "can obscure bedding-pole girdles.")

        st.divider()
        st.header("4 · Rose diagram")
        rose_options = ["Strike", "Dip direction", "Line trend"]
        rose_source = st.selectbox("Rose orientation", rose_options)
        rose_types = st.multiselect("Rose structure types", plane_types if rose_source != "Line trend" else line_types,
                                    default=plane_types if rose_source != "Line trend" else line_types)
        rose_bidirectional = st.checkbox(
            "Bidirectional (axial) rose", value=(rose_source == "Strike"),
            help="Use for strikes and joints. Uncheck for paleocurrents or "
                 "other directional azimuths.",
            key=f"rose_axial_{rose_source}",
        )
        bin_size = st.select_slider("Rose bin size (degrees)", options=[5, 10, 15, 20, 30], value=10)

    planes = all_planes.loc[all_planes["Structure_Type"].isin(chosen_planes)]
    lines = all_lines.loc[all_lines["Structure_Type"].isin(chosen_lines)]
    bedding = all_planes.loc[all_planes["Structure_Type"].isin(fold_types)]
    density_planes = bedding if density_basis == "Fold-fit bedding" else planes

    fold = None
    if len(bedding) >= 3:
        fold = analyze_fold(bedding)

    metric_row = st.columns(4)
    metric_row[0].metric("Total records", f"{len(data)}")
    metric_row[1].metric("Planar records", f"{len(all_planes)}")
    metric_row[2].metric("Linear records", f"{len(all_lines)}")
    metric_row[3].metric("Bedding for fit", f"{len(bedding)}")

    if fold is None:
        st.warning("Select at least three bedding planes in 'Bedding group(s) for fold fit' "
                   "to calculate a π/β fold axis. The plots remain available.")
    else:
        st.subheader("Bedding-pole eigenanalysis")
        axis_col, pole_col, plane_col = st.columns(3)
        axis_col.metric("π/β fold-axis trend / plunge",
                        f"{fold.fold_trend:05.1f}° / {fold.fold_plunge:04.1f}°")
        pole_col.metric("Principal pole trend / plunge (λ₁)",
                        f"{fold.principal_trend:05.1f}° / {fold.principal_plunge:04.1f}°")
        plane_col.metric("Plane represented by principal pole (RHR)",
                         f"{fold.mean_strike:05.1f}° / {fold.mean_dip:04.1f}°")
        st.caption("Normalized tensor eigenvalues: "
                   f"λ₁ = {fold.eigenvalues[0]:.3f}, "
                   f"λ₂ = {fold.eigenvalues[1]:.3f}, "
                   f"λ₃ = {fold.eigenvalues[2]:.3f}. "
                   f"Fitted π-girdle plane (RHR): "
                   f"{fold.girdle_strike:.1f}° / {fold.girdle_dip:.1f}°.")
        if fold.eigenvalues[1] < 1.5 * max(fold.eigenvalues[2], 1e-12) or fold.eigenvalues[1] < 0.10:
            st.warning("The pole distribution may not define a robust girdle: the "
                       "minimum eigenvector is reported but should not be treated "
                       "as a confidently constrained fold axis.")

    stereo_tab, rose_tab, data_tab = st.tabs(["◉ Stereonet", "◉ Rose diagram", "▤ Data & methods"])

    with stereo_tab:
        if planes.empty and lines.empty and fold is None:
            st.info("Select plane or line structures in the sidebar.")
        else:
            if show_density and len(density_planes) < 3:
                st.warning("Density contours need at least three selected planar measurements.")
            try:
                stereo = draw_stereonet(
                    planes=planes, lines=lines, density_planes=density_planes,
                    fold=fold, projection=projection,
                    show_circles=show_great_circles, show_poles=show_poles,
                    show_lines=show_lineations, show_density=show_density,
                    show_fold=draw_fold, density_method=density_method,
                    colormap=density_map, density_sigma=density_sigma,
                    density_levels=density_levels, density_alpha=density_alpha,
                    show_colorbar=show_colorbar, colorbar_label=colorbar_label,
                )
                st.pyplot(stereo, clear_figure=False, use_container_width=True)
                download_figure_controls(stereo, "structural_stereonet")
                plt.close(stereo)
            except Exception as exc:
                st.error(f"Could not render stereonet: {exc}")

    with rose_tab:
        rose_table = (all_lines if rose_source == "Line trend" else all_planes)
        rose_table = rose_table.loc[rose_table["Structure_Type"].isin(rose_types)]
        column = {"Strike": "Strike_RHR", "Dip direction": "Dip_Direction",
                  "Line trend": "Trend"}[rose_source]
        angles = rose_table[column].dropna().to_numpy(dtype=float)
        if not len(angles):
            st.info("No measurements for the selected rose diagram source and structure types.")
        else:
            if rose_source != "Strike" and rose_bidirectional:
                st.caption("You selected axial duplication for a directional variable; "
                           "consider unchecking 'Bidirectional (axial) rose'.")
            rose = draw_rose(
                angles_deg=angles, bin_size=bin_size,
                bidirectional=rose_bidirectional,
                title=f"{rose_source} rose | {'axial' if rose_bidirectional else 'directional'}",
            )
            st.pyplot(rose, clear_figure=False, use_container_width=True)
            download_figure_controls(rose, "structural_rose_diagram")
            plt.close(rose)

    with data_tab:
        st.markdown("#### Validated measurements (internally converted to RHR)")
        columns = ["Station_ID", "Structure_Type", "Strike_RHR", "Dip",
                   "Dip_Direction", "Trend", "Plunge"]
        st.dataframe(data[columns], use_container_width=True, hide_index=True)
        st.download_button(
            "Download validated orientations as CSV",
            data=data[columns].to_csv(index=False).encode("utf-8"),
            file_name="validated_structural_data.csv", mime="text/csv",
        )
        with st.expander("Data schema and structural methods", expanded=True):
            st.markdown(
                """
**Required columns:** `Station_ID`, `Structure_Type`, `Dip`, `Trend`, `Plunge`,
plus `Strike` for RHR input or either `Dip_Direction` / `Strike` for dip-direction input.
Leave unused plane or line fields **blank** rather than putting zero in them.

**Input modes:** RHR strike `S` dips toward `(S + 90) % 360`.
If the file stores dip direction `D`, RHR strike is `(D - 90) % 360`.
In Dip Direction mode, a present `Dip_Direction` column overrides `Strike` on
rows where it has a value; otherwise the `Strike` column holds dip direction.

**Fold fit:** Lower-hemisphere bedding-pole vectors build the axial orientation
tensor `T = sum(n nᵀ)/N`. Its **smallest** eigenvector (`λ₃`) is normal to
the best-fit plane through poles: the **π/β fold-axis estimate**.
The **largest** eigenvector (`λ₁`) gives a principal pole orientation and
an associated representative plane. It is **not** an arithmetic mean of strike
angles and is not necessarily a unique 'mean bedding' for a folded population.
A weak or incomplete girdle makes the fold-axis estimate unreliable.

**Rose:** Axial mode duplicates each measurement at +180° to represent the
same unoriented strike at both ends. Dip direction and paleocurrent trend
are generally **directional**, so use the unidirectional mode.

**Contouring:** `mplstereonet` Kamb and exponential-Kamb values are deviations
from uniform point density in standard-deviation (σ) units. Synthetic sample
measurements illustrate methods and are **not field observations**.
"""
            )
        if st.session_state["data_source"] == "Sample Field Data":
            st.download_button(
                "Download sample_field_data.csv",
                data=SAMPLE_PATH.read_bytes(), file_name="sample_field_data.csv",
                mime="text/csv",
            )

    st.caption("Built for reproducible structural-geology demonstrations • lower hemisphere • "
               "300 DPI PNG / scalable SVG • synthetic sample data")


if __name__ == "__main__":
    main()
