# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment.pdf` for the assessment instructions.

## Cycle 1: data exploration and visualization

Open [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb).
The notebook explains the columns, audits data quality, and explores prices,
distance, equipment, weight, dates, geography, and the two numeric signals.
It also compares development and final-input data and explains the reduced
December inputs. It does not train a model or fill prediction files.

The notebook converts negative `weight` values to positive values in the working
pandas DataFrames using `abs()`, preserves missing weights, and adds a
`weight_was_negative` flag. Original signed copies remain available for audits
and the sign-error diagnostics. The supplied CSVs are never overwritten.

A Python 3.13 virtual environment is available in `.venv`. From this project
directory in PowerShell, launch JupyterLab with:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_data_exploration.ipynb
```

Select the **Python (spotter)** kernel and run all cells from top to bottom.
Saved cell outputs include the tables and plots. The notebook locates the CSVs
in the project root and leaves them unchanged.

To recreate the environment on another Windows machine with Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name spotter --display-name "Python (spotter)"
```

To activate it in a PowerShell session, run `.\.venv\Scripts\Activate.ps1`.
Activation is optional when using the explicit interpreter commands above.
In VS Code, selecting `.venv\Scripts\python.exe` as the notebook kernel also works.

## Cycle 2: CatBoost training and measurement

Open [`notebooks/02_catboost_pipeline.ipynb`](notebooks/02_catboost_pipeline.ipynb)
and run its cells in order with the **Python (spotter)** kernel. The notebook
uses the reusable pipeline in [`freight_model.py`](freight_model.py).

The raw labeled rows are split **before fitting preprocessing or models**:

| Partition | Dates in 2025 | Rows | Purpose |
| --- | --- | ---: | --- |
| Train | January-June | 28,806 | Fit candidate models and preprocessing |
| Development | July-August | 9,671 | Early stopping and model selection |
| Test | September-October | 9,523 | Measure the selected model after refitting on January-August |

The split follows the future prediction task instead of mixing dates at random.
September-October is held out from model selection; it was already inspected in
Cycle 1, so it is not a completely untouched analysis set. `validation.csv` has
hidden targets and is not the development split or a measurable test set.

The first shared-input model uses pickup/delivery cities, equipment, distance,
absolute weight, missing/negative-weight flags, four separate coordinates, and
days since the training start, weekday, and day of month. CatBoost handles
numerical NaNs directly. Missing categories get an explicit label. Coordinate
columns/cells absent at inference are filled using a city lookup learned only
from the fitting rows; an unknown city with no supplied coordinates stays NaN.
Provided coordinates are preserved, including negative longitudes.

`load_id` and all price/output columns are excluded from features. `quote_signal`
is excluded because its relationship with price changes across months.
`market_index` is excluded from this first model because the December scenario
does not provide it. No final-input data is used to learn preprocessing.

Three predeclared CatBoost configurations compare depth 6/MAE, depth 8/MAE, and
depth 6/RMSE. Each predicts total dollars directly, uses learning rate 0.05, seed
42, up to 1,200 trees, and development MAE for early stopping with patience 100.
These are starting settings, not claims of optimal values. The selected tree
count is frozen before refitting on January-August; the test set is never passed
as an early-stopping set. Predictions have a fixed $0.01 floor in the complete
pipeline and all reported metrics.

The baseline uses the fitting data's median price per mile by equipment,
multiplied by distance. Measurements include MAE/RMSE in dollars, R-squared,
signed bias (prediction minus actual), and errors by month, equipment, distance,
missing weight, corrected negative weight, and seen/unseen lanes.

The completed first run selected **depth 6, MAE loss, and 94 trees**. Its
development MAE was **$155.66** versus **$222.73** for the baseline. After the
January-August refit, the September-October results were:

| Metric | Equipment baseline | CatBoost |
| --- | ---: | ---: |
| MAE | $229.10 | $128.08 |
| RMSE | $670.66 | $643.24 |
| R-squared | 0.8069 | 0.8223 |
| Mean signed error | +$40.93 | -$67.04 |
| Median absolute error | $115.72 | $42.52 |

CatBoost reduced test MAE by **44.1%**. Its larger RMSE and negative bias show
that large errors and underprediction remain: 128 of 9,523 test rows had an
absolute error above $1,000. No rows were removed from these measurements and
the positive prediction floor affected zero test predictions. These are local
chronological results, not a measured November-December submission score.

The notebook saves diagnostics and artifacts under `outputs/catboost/`:

- `catboost_pipeline.joblib`: the complete preprocessing and model pipeline,
  fitted on **January-August**, with the measured test-period behavior.
- `metadata.json`: selected settings, versions, feature list, limitations, and
  original source-file SHA-256 hashes.
- `split_assignments.csv`, `split_summary.csv`: reproducible raw-row partitions.
- `dev_metrics.csv`, `test_metrics.csv`, `slice_metrics.csv`: measured results.
- `dev_predictions.csv`, `test_predictions.csv`: local evaluation predictions,
  not final submission files.
- `feature_importance.csv`, `learning_curves.json`, and notebook charts.

Run the same training and save its model/tables from PowerShell without Jupyter:

```powershell
.\.venv\Scripts\python.exe freight_model.py
```

To use the saved pipeline from the project root:

```python
import joblib
import pandas as pd

pipeline = joblib.load("outputs/catboost/catboost_pipeline.joblib")
raw_loads = pd.read_csv("validation.csv")
predictions = pipeline.predict(raw_loads)
```

Only load model files you trust. Keep `freight_model.py` importable and install
the pinned dependencies when loading this artifact. The model includes its
feature preparation; pass raw named columns, not manually encoded arrays.

Run the focused pipeline checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Local dev/test rows contain no cities unseen in their fitting periods, so they
cannot establish unseen-city accuracy. Trees also cannot extrapolate a trend
past their observed ordinal dates, and there are no labeled November-December
observations to establish recurring end-of-year behavior. The notebook checks
that the reduced December schema works; that check does not establish accuracy.
Further tuning after inspecting test errors would require a new evaluation
plan. A future submission step can refit the locked configuration on all labeled
rows; this cycle preserves the model measured on September-October.

## Cycle 3: preserved CatBoost, XGBoost, and FFN comparison

The previous work is backed up in
[`outputs/snapshots/catboost_baseline_2026-10-05.zip`](outputs/snapshots/catboost_baseline_2026-10-05.zip),
including the original data, code, notebooks, and results plus a SHA-256 manifest.
The CatBoost model and its existing output files remain unchanged.

Open [`notebooks/03_xgboost_ffn_comparison.ipynb`](notebooks/03_xgboost_ffn_comparison.ipynb).
It contains the executed comparison, preprocessing audit, learning curves, and
error breakdowns. Its default `RUN_TRAINING = False` reads the already trained
artifacts; set it to `True` to rerun only the two new experiments. The command-line
entry points are:

```powershell
.\.venv\Scripts\python.exe xgboost_experiment.py
.\.venv\Scripts\python.exe ffn_experiment.py
```

Both experiments verify the original split manifest and preserve the same core
14 features. [`comparison_common.py`](comparison_common.py) corrects weight signs
in memory, then fits median numerical imputation and dense one-hot encoding on
training rows only. Missing/sign flags are retained. FFN also standardizes
numerical features and its training target, converting predictions back into
dollars. No source CSV or target outlier is changed. A missing value's median
replacement is an estimate, not the recovered measurement.

Known cities with absent coordinates use the training-only city lookup. An
unknown city with no supplied coordinates ultimately gets training-median
coordinates as a numerical fallback, not real recovered geography. Unknown
categories use an all-zero one-hot block.

XGBoost compared three configurations, selecting depth 4, absolute-error loss,
and **81 rounds** by July-August MAE. The FFN is a fully connected ReLU network
implemented with scikit-learn `MLPRegressor` and Adam. Its three configurations
selected **128 and 64 hidden units, L2 penalty 0.001, and 2 epochs**. FFN training
continued beyond epoch 2 while checking for improvements, then restored that
best development checkpoint. Its early stopping uses the chronological dev
set explicitly, not sklearn's random internal validation split. The target
scaler and all feature preparation are refitted with the selected model on
January-August before September-October measurement.

| Model | Dev MAE | Test MAE | Test RMSE | Test MAPE | Test within 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved CatBoost | $155.66 | $128.08 | $643.24 | 6.62% | 94.90% |
| XGBoost | $135.37 | $130.53 | $639.58 | 6.48% | 93.80% |
| FFN | $284.58 | $180.80 | $644.25 | 9.67% | 77.76% |

All test measurements cover the same 9,523 rows. “Within 10%” means absolute
prediction error divided by actual price is at most 0.10; it is not generic
classification accuracy. The notebook also reports within 5%/20%, R-squared,
bias, and median absolute error. XGBoost leads on development MAE. CatBoost has
the lowest reused-test MAE, while XGBoost has slightly lower RMSE and MAPE.
This FFN does not improve on either tree model. These results describe three
bounded candidates per family and one seed, not all possible configurations.

September-October was already inspected during the previous cycle. It is a
**reused comparison period**, not a fresh untouched test. The new candidates
were chosen before their test evaluation; there was no tuning after test results.
Future experiments need a validation plan established before further tuning.
The original CatBoost retains native handling of nulls/categories, so this is a
comparison of complete pipelines with model-appropriate preprocessing.

Saved results:

- `outputs/xgboost/pipeline.joblib` and `outputs/ffn/pipeline.joblib`: complete
  raw-input inference pipelines, both fitted through August 2025.
- Each family directory contains its experiment plan, locked selection,
  development/test predictions and metrics, subgroup results, learning curves,
  and metadata.
- `outputs/comparison/comparison_metrics.csv`, `comparison_slices.csv`, and
  `comparison_summary.json`: aligned three-model and baseline comparisons.
- `outputs/comparison/`: comparison and learning-curve plots.

The comparison notebook verifies saved-pipeline inference, reduced December
schema compatibility, source CSV hashes, and preservation of the CatBoost files.
The test suite includes train-only preprocessing, unknown categories, target
scaling, chronological FFN checkpoint restoration, and model serialization.
Cycle 3 itself does not write final submission prediction files.

## Final validation predictions

Open [`notebooks/04_final_validation_predictions.ipynb`](notebooks/04_final_validation_predictions.ipynb)
and run all cells with the **Python (spotter)** kernel to refit the final model
and regenerate the filled template. The executed notebook includes each step:
input checks, frozen model settings, the full-data refit, preprocessing audit,
ID-aligned predictions, CSV export, model reload, and preservation checks.
It runs the training and prediction code directly; it does not just display a
previous CSV. The earlier evaluation notebooks and results remain unchanged.

[`validation-predictions.csv`](validation-predictions.csv) contains all 12,000
predictions, matched by `load_id` and kept in the original template's row order.
The columns are exactly `load_id,predicted_rate`, with prices saved to two decimal
places. The original template and input CSVs are unchanged.

[`predict_validation.py`](predict_validation.py) refits the frozen CatBoost
configuration (depth 6, MAE loss, 94 trees) on all 48,000 January-October labeled
loads, then predicts `validation.csv`. Its complete pipeline and provenance are
saved separately in `outputs/final_catboost/`; the previous evaluation model and
results remain intact. Earlier test metrics refer to that evaluation model, not
to an independently measured score for these final predictions.

```powershell
.\.venv\Scripts\python.exe predict_validation.py
```

Both the notebook and script validate all IDs, row count, column order, finite positive prices,
saved-model reload, template ordering, and unchanged source/evaluation hashes.

## Completed submission files and report

Open [`notebooks/05_submission_and_report.ipynb`](notebooks/05_submission_and_report.ipynb).
Its five executed code cells regenerate the official prediction files from the
saved full-data model, run the supplied scorer, show the December chart, build
the PDF report and verify preservation of the earlier work.

| Deliverable | Location |
| --- | --- |
| Official validation predictions | [`validation_predictions.csv`](validation_predictions.csv) |
| Completed December scenario | [`december_predictions.csv`](december_predictions.csv) |
| Required scorer-generated chart | [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) |
| Five-page PDF report | [`output/pdf/freight_rate_report.pdf`](output/pdf/freight_rate_report.pdf) |
| Scorer output | [`scorer_results/scorer_output.txt`](scorer_results/scorer_output.txt) |
| Prediction audit and hashes | [`outputs/submission/submission_checks.json`](outputs/submission/submission_checks.json) |
| 2 minute 39 second walkthrough video | [`output/video/spotter_walkthrough.mp4`](output/video/spotter_walkthrough.mp4) |
| Walkthrough script | [`walkthrough_script.txt`](walkthrough_script.txt) |

The official underscore-named validation file contains exactly the same bytes
as the earlier `validation-predictions.csv`. Both have 12,000 rows in template
order with only `load_id,predicted_rate`. `december_predictions.csv` preserves
the original seven-column order and all input values; it fills 31 prices without
overwriting `december-chart-inputs.csv`.

The official scorer accepted both files. December predictions range from
**$818.09 to $826.55**. Only the date changes in this fixed-load scenario; the
curve is not evidence of learned holiday seasonality. The scorer validates
format and generates the chart; Spotter calculates the hidden-target metrics.

### Reproduce the submission

With the environment above installed, run from the repository root:

```powershell
# Optional report dependencies; prediction code does not require them.
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt

# Refit the already selected configuration on all 48,000 labeled loads.
.\.venv\Scripts\python.exe predict_validation.py

# Predict both final input schemas and run the original scorer.
.\.venv\Scripts\python.exe complete_submission.py

# Rebuild the PDF from saved measurements and the scorer-generated chart.
.\.venv\Scripts\python.exe build_report.py
```

If the saved final pipeline already exists, the refit command is optional.
The scorer can also be run directly:

```powershell
.\.venv\Scripts\python.exe score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

`complete_submission.py` checks source/model provenance, template IDs and order,
finite positive predictions, CSV reload precision, December constants, coordinate
lookup availability and preservation of source files and `readme-spotter.md`
when present. The notebook also fingerprints all previous model/comparison files.
The PDF was rendered and visually checked; the 18 existing pipeline tests pass.

### Loom handoff

The local video covers exploration findings, data quality, model choice, the
chronological split, actual code excerpts and the December chart. It uses a
clearly labeled standard computer-generated English narrator, not the applicant's
voice. It is 159.5 seconds long, with 1080p video and an audio track.

**The required Loom link is still pending.** Browser security blocked the
recording controls. Upload the prepared MP4 in your Loom account if available,
or record the walkthrough using the supplied script, then submit its accessible
share link. No Loom link has been fabricated.

Optional local video regeneration on Windows requires `ffmpeg` on PATH and an
installed English System.Speech voice:

```powershell
.\.venv\Scripts\python.exe build_walkthrough.py
```

### Submit

- Accessible repository: https://github.com/Mutasem-Tamimi/spotter
- `validation_predictions.csv`
- `output/pdf/freight_rate_report.pdf` (includes the required December chart)
- The 2-3 minute Loom share link after the handoff above

The repository is intentionally private; the owner will arrange assessor access.
The supplied assessment instructions in `readme-spotter.md` are unchanged.
