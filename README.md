# Structural Geology Studio

An interactive Streamlit stereonet and rose-diagram app built for a classroom demonstration. Supports RHR strike/dip and dip-direction/dip, lower-hemisphere Schmidt and Wulff projections, Kamb and exponential-Kamb contours, planar and linear structures, eigenvector fold fitting, axial/directional rose diagrams, and high-resolution PNG/SVG downloads.

## Included files

- **`app.py`** — full Streamlit app, data validation, structural math and plots (modular functions in one file).
- **`sample_field_data.csv`** — **40 synthetic** Himalayan-style structural measurements: 28 folded bedding observations (14 per limb), 10 joint planes, and 2 fold-hinge lineations. Generated, **not measured in the field**.
- **`requirements.txt`** — version-pinned dependencies.
- **`README.md`** — these setup and usage instructions.
- **`test_core.py`** — repeatable unit tests of RHR conversion, fold-axis fitting, input validation and exports.

## Installation (Windows PowerShell)

Install **Python 3.11 or 3.12** and ensure `py` launches it. In a PowerShell terminal:

```powershell
cd path\to\structural_geology_stereonet
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is blocked by the local execution policy, skip the activation line and run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```bash
cd /path/to/structural_geology_stereonet
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If your computer uses a different supported Python executable, replace `python3.11` / `py -3.11` with `python3.12` / `py -3.12`.

Open the **localhost** URL that Streamlit prints (usually `http://localhost:8501`). Press **Ctrl+C** in the terminal to stop.

Optional self-test (from the activated environment):

```bash
python -m unittest test_core.py -v
```

## Classroom demonstration (about five actions)

1. App opens on **Sample Field Data**; choose **RHR Strike / Dip** for the included file.
2. Under **Fold analysis**, leave both **Bedding (North Limb)** and **Bedding (South Limb)** selected. The best-fit π/β axis should be close to **115° trend / 10° plunge** (small noise is deliberately included).
3. Switch from **Equal-Area (Schmidt)** to **Equal-Angle (Wulff)**, then toggle great circles, poles, or density.
4. Compare **Kamb** and **Exponential Kamb** density or alter the colormap and smoothing.
5. In the rose tab, compare **Strike** + bidirectional (axial) with **Dip direction** + unidirectional; export figures in **300 DPI PNG** and **vector SVG**.

## Data schema and conventions

Required headings: `Station_ID, Structure_Type, Dip, Trend, Plunge`, plus `Strike` (for RHR) or one of `Dip_Direction` / `Strike` (for dip-direction mode). A `.xlsx` file may use the same headings.

- Plane-only row: fill `Strike`/`Dip` and leave `Trend`/`Plunge` empty.
- Line-only row: leave `Strike`/`Dip` empty and fill `Trend`/`Plunge`.
- Both pairs may be supplied if both features are observed at the same station.
- **RHR**: dip direction = `(strike + 90) % 360`.
- **Dip direction / dip** mode: if `Dip_Direction` exists its values take precedence; otherwise `Strike` is interpreted as **dip direction**, *not strike*. Convert to RHR with `(dip_direction - 90) % 360`.
- Bearing and trend must be 0°–360°; dips and plunges must be 0°–90°. Angles of 360° wrap to 0°.
- Empty / nonnumeric / out-of-range measurements are rejected with a useful row number. Blanks are allowed only when the corresponding pair is also blank.

### Example RHR rows

```csv
Station_ID,Structure_Type,Strike,Dip,Trend,Plunge
P01,Bedding (North Limb),112,45,,
P02,Joint Set A,040,75,,
L01,Fold Hinge Lineation,,,115,10
```

### Dip-direction Excel or CSV input

```csv
Station_ID,Structure_Type,Dip_Direction,Dip,Trend,Plunge
P01,Bedding (North Limb),202,45,,
```

Switch the input-convention radio to **Dip Direction / Dip** before uploading this file. It will be plotted as RHR `112/45`.

## Methodology and limitations

For each bedding observation of RHR strike `s` and dip `d`, the lower-hemisphere pole is `trend = (s + 270) % 360`, `plunge = 90-d`. Convert all poles to N/E/D unit vectors `n`, then build the **axial orientation tensor**

`T = (Σ n·nᵀ) / N`.

Diagonalize `T` with `numpy.linalg.eigh`; sort `λ1 >= λ2 >= λ3`:

- `λ3` eigenvector = pole to the **best-fit girdle** of bedding poles, i.e. π/β cylindrical fold-axis **estimate** (trend/plunge).
- `λ1` eigenvector = principal **pole** orientation (plus associated RHR plane), which is not the fold axis and should **not** be called a simple mean strike/dip for folded populations.
- A clear pole girdle is necessary for reliable fold inference. A bimodal or scattered joint population should not be mixed into a bedding-only fold fit.
- All stereonets use the **lower hemisphere** and compass azimuths clockwise from north.
- The π/β axis is an approximation to a cylindrical fold's axis; real folds may be conical, noncylindrical, or overprinted.
- Kamb significance is reported in **standard deviations above uniform distribution**; equal-area is often preferred for density interpretation.
- Bidirectional rose duplicates strike directions at +180°, so the histogram counts include both directional rays for each planar measurement.

This sample is **synthetic**, for demonstration and code testing only. Do not report sample orientation estimates as measured Himalayan results.

## Troubleshooting

- **`ModuleNotFoundError`**: activate the virtual environment and install requirements using the *same* interpreter that launches Streamlit (`python -m pip ...`, `python -m streamlit ...`).
- **Empty rose diagram**: choose planar types for strike or dip-direction roses; choose lineation types for a trend rose.
- **No fold axis**: select at least 3 valid bedding planes; for a meaningful girdle use observations from both fold limbs.
- **Incorrect sample orientation after input conversion**: the packaged CSV uses RHR; keep RHR mode for it.
- **Excel fails to read**: save a standard `.xlsx` file (not `.xls`) and install `openpyxl` via requirements.
