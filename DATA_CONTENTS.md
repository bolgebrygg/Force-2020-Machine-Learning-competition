# FORCE 2020 — Data Contents

> **Path:** `/home/ubuntu/projects/force2020/lithology_competition/data/`
> **Date recorded:** 2026-05-27

---

## Root Files

| File | Size | Description |
|------|------|-------------|
| `train.zip` | 91.6 MB | Training data utama (belum di-extract) |
| `hidden_test.csv` | 30.3 MB | Test set hidden (11 wells, 29 kolom) |
| `leaderboard_test_features.csv` | 31.7 MB | Test features leaderboard (11 wells) |
| `leaderboard_test_target.csv` | 3.7 MB | Test target leaderboard (label litologi) |
| `penalty_matrix.npy` | 1.3 KB | Skoring matrix kompetisi |
| `starter_notebook.ipynb` | 367 KB | Baseline notebook dari penyelenggara |
| `test_code.py` | 1.2 KB | Skoring script |

---

## LAS Files

**Path:** `las_files_Lithostrat_data/`
**Total:** 119 file LAS
**Size per file:** ~3–6 MB

### Daftar Wells (119 wells)

- 15_9-13, 15_9-14, 15_9-15, 15_9-17, 15_9-23
- 16_1-2, 16_1-6 A
- 16_2-6, 16_2-7, 16_2-11 A, 16_2-16
- 16_4-1, 16_5-3
- 16_7-4, 16_7-5, 16_7-6, 16_8-1
- 16_10-1, 16_10-2, 16_10-3, 16_10-5
- 16_11-1 ST3
- 17_4-1, 17_11-1
- 25_10-9, 25_10-10
- 25_11-5, 25_11-15, 25_11-19 S, 25_11-24
- ... (dan 89 wells lainnya)

### LAS Format Info

- **Source:** PETREL export
- **Depth unit:** meter
- **Step:** 0.152 m (= 0.5 ft)
- **Null value:** -999.25

### Kurva di LAS Files

| Curve | Unit | Description |
|-------|------|-------------|
| DEPT | m | Depth |
| FORCE_2020_LITHOFACIES_CONFIDENCE | — | Label confidence (dari kompetisi) |
| FORCE_2020_LITHOFACIES_LITHOLOGY | — | Litologi label (dari kompetisi) |
| CALI | in | Caliper |
| MUDWEIGHT | — | Mud weight |
| ROP | m/h | Rate of penetration |
| RDEP | ohm.m | Deep resistivity |
| RSHA | ohm.m | Shallow resistivity |
| RMED | ohm.m | Medium resistivity |
| RXO | ohm.m | Flushed zone resistivity |
| SP | mV | Spontaneous potential |
| DTC | us/ft | Compressional sonic |
| NPHI | m3/m3 | Neutron porosity |
| PEF | b/e | Photoelectric factor |
| GR | gAPI | Gamma ray |
| RHOB | g/cm3 | Bulk density |
| DRHO | g/cm3 | Density correction |

---

## CSV Kolom (hidden_test.csv / leaderboard_test)

| Kolom | Tipe | Description |
|-------|------|-------------|
| WELL | string | Well name (e.g. 15/9-23) |
| DEPTH_MD | float | Measured depth (m) |
| X_LOC | float | X coordinate (UTM) |
| Y_LOC | float | Y coordinate (UTM) |
| Z_LOC | float | Z coordinate (TVD) |
| GROUP | string | Lithostratigraphic group |
| FORMATION | string | Formation name (50 unique values) |
| CALI | float | Caliper (in) |
| RSHA | float | Shallow resistivity (ohm.m) |
| RMED | float | Medium resistivity (ohm.m) |
| RDEP | float | Deep resistivity (ohm.m) |
| RHOB | float | Bulk density (g/cm3) |
| GR | float | Gamma ray (gAPI) |
| SGR | float | Spectral gamma ray |
| NPHI | float | Neutron porosity (v/v) |
| PEF | float | Photoelectric factor (b/e) |
| DTC | float | Compressional sonic (us/ft) |
| SP | float | Spontaneous potential (mV) |
| BS | float | Bit size (in) |
| ROP | float | Rate of penetration (m/h) |
| DTS | float | Shear sonic (us/ft) |
| DCAL | float | Differential caliper |
| DRHO | float | Density correction (g/cm3) |
| MUDWEIGHT | float | Mud weight |
| RMIC | float | Micro-resistivity |
| ROPA | float | ROP average |
| RXO | float | Flushed zone resistivity |
| FORCE_2020_LITHOFACIES_LITHOLOGY | int | Litologi code (target) |
| FORCE_2020_LITHOFACIES_CONFIDENCE | float | Confidence label |

---

## Lithostratigraphy

### Groups (13)

| Group | Keterangan |
|-------|------------|
| NORDLAND GP. | Pliocene–Pleistocene |
| HORDALAND GP. | Oligocene–Miocene |
| ROGALAND GP. | Paleocene–Eocene |
| SHETLAND GP. | Late Cretaceous |
| CROMER KNOLL GP. | Early Cretaceous |
| VIKING GP. | Late Jurassic |
| VESTLAND GP. | Middle Jurassic |
| DUNLIN GP. | Early Jurassic |
| HEGRE GP. | Triassic |
| BAAT GP. | Triassic |
| ROTLIEGENDES GP. | Permian |
| ZECHSTEIN GP. | Permian |
| SKAGERRAK GP. | Triassic |

### Formations (50)

**Brent Group (Middle Jurassic):**
- Etive Fm, Rannoch Fm, Ness Fm, Oseberg Fm, Tarbert Fm

**Viking Group (Late Jurassic):**
- Draupne Fm, Heather Fm, Fensfjord Fm, Sognefjord Fm

**Shetland Group (Cretaceous):**
- Hod Fm, Kyrre Fm, Blodoeks Fm, Ekofisk Fm

**Reservoir sands:**
- Hugin Fm, Sleipner Fm, Skagerrak Fm, Smith Bank Fm, Johansen Fm, Intra Draupne Fm. Sst.

**Shales/seals:**
- Heather Fm, Draupne Fm, Hod Fm

---

## Data Source

- **Origin:** Norwegian Continental Shelf (North Sea, Norwegia)
- **Released by:** FORCE (Forum for Reservoir Characterization, Evaluation and Monitoring) + NPD (Norwegian Petroleum Directorate)
- **Operators:** Equinor, Aker BP, ConocoPhillips, dll
- **Total wells:** 118 wells
- **Well naming:** NPD quadrant/block system (e.g. 15/9-23)

---

## Relevansi untuk Thin Layer / Laminated Reservoir Research

1. **Heterolithic class** — litologi code "Sandstone/Shale" (65000) = thin-bedded interbedded intervals, langsung relevant
2. **Brent Group** — heterolithic/laminated facies well-documented di literatur
3. **118 wells** — cukup untuk multi-well ML analysis
4. **DTC + DTS** tersedia di beberapa wells — sonic anisotropy analysis
5. **Formation tops** bisa di-derive dari kolom FORMATION di CSV (transisi depth = boundary)
