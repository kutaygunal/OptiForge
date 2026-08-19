# AI Start Expert

A **generative-AI-powered starting-point lens design generator** for **CODE V 2026**.
Give a small set of specifications — focal length, field of view, aperture — and it
produces a viable **first-pass lens design** as a standard **CODE V lens-sequence
(`.seq`)** file, ready for immediate refinement with CODE V's optimization tools.

It targets the earliest and often slowest stage of lens design: *creating the initial
lens form*. Instead of manually sketching a first lens, you get optically sensible
starting points in seconds.

- **Inputs:** FOV, focal length, and aperture (EPD / NA / F/#), plus optional
  distortion and image-clearance constraints.
- **Outputs:** a lens drawing, first-order + third-order design data, and a CODE V
  `.seq` file you load with `IN <file>`.
- **Forms:** camera / double Gauss, Cooke triplet, telephoto, retrofocus (wide angle),
  Petzval (projector), collimator / projector, and microscope objectives.
- **On-premise:** runs entirely locally with **no internet** — suitable for secure /
  offline engineering environments.

> Built intelligence for lens *shape and sensitivity*: every generated lens is a scaled
> classical form whose third-order and chromatic aberrations are balanced by a local
> merit-function optimizer, so the result is optically sensible, not arbitrary.

---

## Quickstart

### CLI

```bash
# Camera lens: EFL 50 mm, FOV 40°, F/2.8
python -m aistart.cli --efl 50 --fov 40 --fno 2.8 --type double_gauss --out double_gauss.seq --layout

# Telephoto: long focal in a short tube
python -m aistart.cli --efl 120 --fov 20 --fno 4 --type telephoto --layout

# Wide angle (retrofocus)
python -m aistart.cli --efl 20 --fov 90 --fno 4 --type retrofocus --layout

# Microscope objective (finite conjugate, object-side NA)
python -m aistart.cli --efl 1 --naf 0.25 --type microscope --obj-dist 1.0 --obj-height 0.1 --layout
```

Help: `python -m aistart.cli --help`

### Web UI

```bash
python -m aistart.webapp
# open http://127.0.0.1:5000
```

Everything is served locally. The page auto-generates a design on load and lets you
download the `.seq`.

### In CODE V

```text
IN <filename>.seq
```
then run your normal optimization (AUT / SAB / etc.) on the starting point.

---

## How it works

1. **Specs** (`aistart/specs.py`) — normalize the aperture (EPD ↔ F/# ↔ NA) and field.
2. **Prototype selection** (`aistart/autotype.py`) — choose a classical form from the
   operating point (field + speed), or honor an explicit choice.
3. **Classical library** (`aistart/catalog.py`) — real Schott starting forms stored at
   a reference focal length.
4. **Optimization** (`aistart/optimize.py`) — hold the target EFL exactly (homothetic
   normalization), then shape the lens to reduce third-order (Seidel) and chromatic
   aberration while respecting minimum thickness and image clearance. scipy SLSQP.
5. **Export** (`aistart/codevexport.py`) — write a standard CODE V `.seq` lens sequence.
6. **Validation** (`aistart/seqparse.py`) — round-trip the exported file and re-trace it
   to confirm EFL / F# / geometry.

### The optics engine (`aistart/optics.py`)
Pure-numpy paraxial ray tracing + third-order (Seidel) aberration coefficients:
S1 spherical, S2 coma, S3 astigmatism, S4 field curvature, S5 distortion, plus
axial/lateral chromatic. No ray-tracing library is required — it is self-contained.

### CODE V output format
The exported `.seq` is plain-text LDM commands:
```
TIT 'Double Gauss'
DIM MM
WL 656.3 587.6 486.1
REF 2
EPD 17.857143
YAN 0.0 20.000000
SO 0.0 1.0E+13
S 850.93 4.25 NSSK2_SCHOTT
...
STO
S -16.77 70.93
SI 0.0 0.0
GO
```
Finite conjugates (microscope) use `NAO` and `YOB` instead.

---

## Project layout

```
aistart/
  specs.py       input specification model
  catalog.py     classical starting-point library
  autotype.py    auto lens-form selection
  optics.py      paraxial + third-order engine
  optimize.py    merit-function optimizer (scipy)
  generator.py   end-to-end generation orchestrator
  codevexport.py CODE V .seq writer
  seqparse.py    .seq parser / round-trip validator
  report.py      text report + lens drawing (matplotlib)
  cli.py         command-line interface
  webapp.py      local Flask UI
  templates/     web UI HTML
examples/        sample .seq + layout images per lens form
tests/           pytest suite
```

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers: aperture conversions, EFL/F# accuracy against a real CODE V double-Gauss,
homothetic scale invariance, per-lens-type generation, and `.seq` round-trip validity.

---

### Notes
- All generated radii are in **radius mode** (CODE V default), so `S <radius> <thickness> <glass>`.
- Glass references are CODE V catalog strings (`NBK7_SCHOTT`, etc.).
- Third-order distortion is reported as an estimate; use CODE V for a full design.
