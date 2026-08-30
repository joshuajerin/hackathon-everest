# Real calibration datasets

No public dataset contains synchronized G1 crampon force, moving-probe travel, foot IMU, FMCW radar, contact
outcome, and slip labels. Public data is used for component priors and parser/physics checks. End-to-end packet
training remains simulation-generated until the project collects shared-clock device data.

## MOSAiC SnowMicroPen — anonymous numeric pilot

- Record: <https://doi.org/10.1594/PANGAEA.935554>
- Index: <https://doi.pangaea.de/10.1594/PANGAEA.935554?format=textfile>
- License: CC BY 4.0
- Format: vendor `.pnt`, 5 mm cone, about 4 µm sampling and 20 mm/s nominal probe speed
- Use: snow force-depth distributions, layer signatures, density parameterization checks, parser tests
- Do not use as: direct crampon force scale, ice traction label, or synchronized radar/force truth

Download a small deterministic subset and fit a summary:

```bash
uv sync --extra calibration
uv run python scripts/download_mosaic_smp.py --per-month 2
uv run python scripts/calibrate_mosaic_smp.py
```

PANGAEA cold files can return HTTP 503 while being restored from tape. The downloader records failures and
keeps a usable subset. The checked-in derived summary under `calibration/mosaic_smp/` currently uses five
profiles from December 2019, January 2020, May 2020, and September 2020. It observed:

- SMP force median `0.958 N`, p95 `9.805 N`, p99 `17.926 N`;
- P2015-derived density p05 `131.6 kg/m³`, median `296.8 kg/m³`, p95 `532.2 kg/m³`.

Five profiles are a pipeline proof, not population coverage. The generated 6 mm force translation uses an
explicit projected-area ratio of `1.44`; it is marked weak because tip geometry, speed, compaction zone, shear,
and fracture differ.

Citation: Macfarlane, A. R. et al. (2021), *Snowpit SnowMicroPen (SMP) force profiles collected during the
MOSAiC expedition*, PANGAEA, <https://doi.org/10.1594/PANGAEA.935554>.

## SnowEx20 SnowMicroPen — next authenticated import

- Landing: <https://nsidc.org/data/snex20_smp/versions/1>
- DOI: <https://doi.org/10.5067/ZYW6IHFRYDSE>
- User guide: <https://nsidc.org/sites/default/files/snex20_smp-v001-userguide.pdf>
- Public field/QC log: <https://nsidc.org/sites/default/files/snex20_smp_fieldnotes.xlsx>
- Granule API: <https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C3271568365-NSIDC_CPRD&page_size=2000>

The numerical CSV/PNT files require a free NASA Earthdata Login. Keep that authentication local. A small
quality-checked candidate is `SNEX20_SMP_S19M0949_2S11_20200201.CSV` at about 54 KB. The product contains raw,
not fully validated force-depth profiles. Negative values, ice on the tip, probe motion, different instruments,
and manually derived interfaces require QC.

Use it to expand snow condition diversity and validate SnowEx joins. Do not randomly split neighboring
profiles across train and test.

## SnowEx20 BSU GPR

- Landing: <https://nsidc.org/data/snex20_bsu_gpr/versions/1>
- DOI: <https://doi.org/10.5067/Q2LFK0QSVGS2>
- User guide: <https://nsidc.org/sites/default/files/snex20_bsu_gpr-v001-userguide.pdf>
- Compact 10 m product: `SNEX20_BSU_GPR_pE_01282020_01292020_02042020_downsampled.csv`, about 1.84 MB

The downsampled table contains time, location, two-way travel time, derived depth, and SWE. It requires
Earthdata Login. It is useful for radar preprocessing and travel-time/depth validation. It is not the waveform
or antenna configuration of the planned foot radar and does not provide traction labels.

A small raw-to-picked-to-derived validation chain is also available as:

- raw: <https://doi.org/10.5067/CL5ZRBCEF8G3>
- picked travel time: <https://doi.org/10.5067/SBFHOZ7F5WHS>
- derived depth/density/SWE: <https://doi.org/10.5067/SOFEX3867ECJ>

## Data needed from the device

Use one hardware clock or measured clock transforms for:

- four calibrated axial loads;
- four probe/collar positions;
- foot IMU;
- raw radar data and decoded five-value frontend;
- G1 pose, joint state, command, and estimated foot wrench;
- an independent force plate or six-axis rig load;
- slip displacement and failure outcome;
- temperature, surface preparation, geometry revision, and repeated-location ID.

Split evaluation by ice/snow batch, day, site, and route. Nearby traces from the same prepared block are not
independent. Store raw readings, calibration coefficients, firmware/software versions, units, timestamp source,
validity flags, and checksums.
