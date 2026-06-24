# ai-bias-reportcard — AI Fairness Report Card Generator

> A web tool that takes a CSV of AI model predictions and patient demographics,
> computes fairness metrics across protected groups, and generates a shareable
> one-page visual Report Card that answers: "Did your AI treat everyone equally?"

---

## Background & Context

### The Problem

When an AI system is used to make decisions — loan approvals, disease risk scores,
hiring recommendations, insurance claims — it may perform significantly worse for
certain demographic groups than others. This is called **AI bias** or **model
unfairness**.

The consequences in healthcare specifically are severe:

- A diagnostic AI that misses cancer in Black patients at higher rates than white
  patients creates real harm.
- A risk-scoring model that under-predicts readmission risk for elderly patients
  leads to premature discharge.
- Regulators (ONC, Joint Commission, EU AI Act) now require evidence that AI
  systems have been evaluated for bias before and during deployment.

Most organizations deploying AI have **no tooling to check this**. They run a model,
check overall accuracy, and ship it. Demographic subgroup performance is never
measured. That gap is the exact problem this tool solves.

### What This Tool Does

A user uploads a CSV that contains:

- A column of ground truth labels (what actually happened)
- A column of model predictions (what the AI said would happen)
- One or more demographic columns (age group, gender, race, etc.)

The tool computes a set of standard fairness metrics **per demographic group**, then
renders a clean, shareable **Report Card** — a one-page HTML document with color-coded
scores (green / yellow / red) that a non-technical executive can read and understand
in 60 seconds.

### Demo Dataset

Use the **UCI Heart Disease Dataset** (Cleveland subset) — publicly available, healthcare
flavored, contains demographic columns (age, sex), a binary outcome (presence of heart
disease: 0 or 1), and is small enough to process instantly.

Download: https://archive.ics.uci.edu/dataset/45/heart+disease

For the demo, simulate a "model prediction" column by training a simple logistic
regression on the dataset and saving its predictions alongside the ground truth.
This gives a realistic demo where you can show the AI performs differently across
age groups and sex.

---

## Tech Stack

- **Backend**: Python 3.11+ with FastAPI
- **Fairness computation**: scikit-learn + pandas + numpy (no specialized fairness
  library — compute metrics manually so you understand and can explain every number)
- **Frontend**: Single-page HTML/CSS/JS — no framework. Served directly by FastAPI
  as a static file. Keep it in one `index.html` file.
- **Report Card output**: A rendered HTML page (printable / shareable via URL or
  downloadable as PDF via the browser's print dialog)
- **Storage**: No database. Everything is computed in-memory per request.
  Files are not persisted server-side.
- **Packaging**: A single `docker-compose.yml` for one-command startup.

---

## Project Structure

```
ai-bias-reportcard/
├── AGENTS.md                  # This file
├── README.md                  # Setup and usage instructions
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                # FastAPI app entrypoint
│   ├── metrics.py             # All fairness metric computations
│   ├── report.py              # Report Card HTML renderer
│   └── demo/
│       ├── generate_demo.py   # Script to produce demo CSV from UCI dataset
│       └── heart_disease_demo.csv  # Pre-generated demo file
└── frontend/
    └── index.html             # Upload UI — single file, no build step
```

---

## Step-by-Step Build Instructions

### Step 1 — Set Up the Python Backend

1. Create `backend/requirements.txt`:

   ```
   fastapi==0.111.0
   uvicorn[standard]==0.29.0
   pandas==2.2.2
   numpy==1.26.4
   scikit-learn==1.4.2
   python-multipart==0.0.9
   jinja2==3.1.4
   ```

2. Create `backend/main.py` with a FastAPI app that has these routes:
   - `GET /` — serves `frontend/index.html`
   - `POST /analyze` — accepts a multipart CSV file upload plus a JSON body
     specifying which columns are: `label_col`, `prediction_col`, `demographic_cols`
     (list of strings). Returns a JSON object with all computed metrics.
   - `POST /report` — same input as `/analyze` but returns a full rendered HTML
     string of the Report Card (for display in an iframe or download).

3. The `/analyze` endpoint must:
   - Parse the uploaded CSV into a pandas DataFrame
   - Validate that specified columns exist; return a 422 with a clear error message
     if any column is missing
   - Call `compute_fairness_metrics(df, label_col, prediction_col, demographic_cols)`
     from `metrics.py`
   - Return the result as JSON

### Step 2 — Build `metrics.py`

This is the core of the project. Implement the following functions:

#### `compute_fairness_metrics(df, label_col, prediction_col, demographic_cols) -> dict`

For **each column in `demographic_cols`**, call
`compute_group_metrics(df, label_col, prediction_col, demographic_col)` and collect
the results. Return a dict keyed by demographic column name.

#### `compute_group_metrics(df, label_col, prediction_col, demographic_col) -> dict`

For each unique value in `demographic_col`, compute:

1. **Accuracy** — `correct_predictions / total_predictions` for this group
2. **Precision** — `TP / (TP + FP)` for this group
3. **Recall (Sensitivity)** — `TP / (TP + FN)` for this group
4. **False Positive Rate** — `FP / (FP + TN)` for this group
5. **False Negative Rate** — `FN / (FN + TP)` for this group
6. **Group size** — number of rows in this group
7. **Positive rate** — fraction of rows in this group where `label == 1`

Then compute these **cross-group** disparity metrics:

8. **Demographic Parity Difference** — `max(prediction_rate) - min(prediction_rate)`
   across all groups, where `prediction_rate = predicted_positive / total` per group.
   Threshold: flag as WARN if > 0.1, flag as FAIL if > 0.2.

9. **Equalized Odds Difference** — `max(recall) - min(recall)` across all groups.
   Threshold: flag as WARN if > 0.1, flag as FAIL if > 0.2.

10. **Disparate Impact Ratio** — `min(prediction_rate) / max(prediction_rate)`.
    A value below 0.8 is the legal "four-fifths rule" threshold for discrimination.
    Flag as FAIL if < 0.8, WARN if between 0.8 and 0.9.

Return structure per demographic column:

```python
{
  "column": "sex",
  "groups": {
    "male":   { "accuracy": 0.84, "recall": 0.81, "fpr": 0.12, "fnr": 0.19, "size": 206, "positive_rate": 0.55 },
    "female": { "accuracy": 0.79, "recall": 0.71, "fpr": 0.18, "fnr": 0.29, "size": 97,  "positive_rate": 0.26 }
  },
  "disparity": {
    "demographic_parity_difference": 0.18,
    "equalized_odds_difference": 0.10,
    "disparate_impact_ratio": 0.74,
    "overall_status": "FAIL"   # worst of the three individual statuses
  }
}
```

`overall_status` is:

- `"PASS"` — all three metrics are within thresholds
- `"WARN"` — at least one metric triggered a WARN
- `"FAIL"` — at least one metric triggered a FAIL

#### Important implementation notes:

- Handle division-by-zero gracefully (a group may have no positive labels). Return
  `null` for metrics that cannot be computed rather than crashing.
- Treat the label and prediction columns as binary (0/1). If non-binary values are
  detected, return a 422 error.
- Cast demographic columns to `str` before grouping so numeric demographic values
  (e.g. age as integer) are handled correctly.

### Step 3 — Build `report.py`

Implement `render_report(metrics_dict, filename) -> str` that takes the output of
`compute_fairness_metrics` and returns a self-contained HTML string.

The Report Card HTML must include:

#### Header section

- Title: "AI Fairness Report Card"
- Subtitle: filename of the uploaded CSV + timestamp of generation
- Overall verdict banner:
  - Green background + "PASS" if all demographic columns passed
  - Yellow background + "REVIEW REQUIRED" if any column is WARN
  - Red background + "BIAS DETECTED" if any column is FAIL

#### Per demographic column section

For each demographic column, render:

1. A **summary badge** — the column name + its `overall_status` in the appropriate
   color (green/yellow/red)

2. A **group performance table** with columns:
   `Group | Size | Accuracy | Recall | False Positive Rate | False Negative Rate`
   - Highlight the lowest-performing group in each metric column with a light red
     background

3. A **disparity metrics panel** showing:
   - Demographic Parity Difference: value + status badge
   - Equalized Odds Difference: value + status badge
   - Disparate Impact Ratio: value + status badge
   - A one-sentence plain-English explanation for each metric (hardcoded strings are
     fine, e.g. "Measures whether the AI predicts positive outcomes at similar rates
     across groups. Values above 0.1 suggest potential unfairness.")

#### Footer section

- "Generated by AI Bias Report Card" + link to the GitHub repo
- A note: "This report is for informational purposes. Consult a qualified AI ethics
  expert before making compliance determinations."

#### Styling rules:

- Entirely inline CSS — no external stylesheets or CDN links. The HTML must be
  fully self-contained so it can be saved and shared as a single file.
- Clean, minimal design. Use a white card layout on a light gray background.
- Status colors: PASS = `#22c55e`, WARN = `#f59e0b`, FAIL = `#ef4444`
- Use a system font stack: `font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
- The page must look good when printed (use `@media print` to hide any buttons)

### Step 4 — Build the Frontend (`frontend/index.html`)

A single HTML file with inline CSS and inline JavaScript. No build step, no npm.

#### UI flow:

1. **Upload screen** — A centered card with:
   - A file drop zone (drag & drop or click to browse) for CSV files
   - After a file is selected, parse the CSV header client-side using plain JS
     (`FileReader` + split on newline + split on comma for the first line)
   - Display a column mapping UI: three form fields:
     - "Ground Truth Column" — a `<select>` pre-populated with all CSV column names
     - "Model Prediction Column" — a `<select>` pre-populated with all CSV column names
     - "Demographic Columns" — a multi-select or a list of checkboxes, one per column
   - A "Generate Report Card" button

2. **Loading state** — Show a spinner and "Analyzing fairness..." text while the
   POST request is in flight.

3. **Results screen** — Render the returned HTML from `POST /report` inside a
   full-width `<iframe>` that fills the viewport. Add a "Download Report" button
   that triggers a download of the iframe content as `report.html`, and a
   "Analyze another file" button that resets the UI.

4. **Error state** — If the API returns an error, show a red banner with the
   error message returned by the server.

#### JavaScript notes:

- Use `fetch` with `FormData` for the file upload.
- For CSV parsing client-side, only parse the header row — you don't need to parse
  the full file in the browser.
- Keep all JS in a single `<script>` tag at the bottom of `<body>`.

### Step 5 — Generate the Demo CSV

Create `backend/demo/generate_demo.py`:

1. Download the UCI Heart Disease (Cleveland) dataset. The file is available at:
   https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
   Column names (in order): `age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target`

2. Load into pandas. The target column has values 0–4; binarize it to 0 (no disease)
   and 1 (disease present, i.e. original value > 0).

3. Replace `sex` values: 1 → "male", 0 → "female"

4. Create an `age_group` column:
   - "< 45" for age < 45
   - "45–54" for age 45–54
   - "55–64" for age 55–64
   - "65+" for age >= 65

5. Train a `LogisticRegression` (default params) using an 80/20 train/test split
   (random_state=42). Features: all columns except `target`, `sex`, `age`, `age_group`.
   Do NOT use demographic columns as features — this ensures the model has no explicit
   access to demographics, making any observed bias purely a reflection of data
   patterns.

6. Generate predictions on the full dataset (not just test set) for a richer demo.

7. Save a CSV with columns: `age, age_group, sex, target, prediction`

8. Save as `backend/demo/heart_disease_demo.csv`

Run this script once and commit the output CSV so the demo works without a network
request.

### Step 6 — Dockerfile and docker-compose

`backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:

```yaml
version: "3.9"
services:
  app:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./frontend:/app/frontend
```

FastAPI must serve `frontend/index.html` for `GET /` using
`StaticFiles` or a simple file response. Mount the `frontend/` directory at startup.

### Step 7 — README.md

Write a README that covers:

1. One-paragraph description of what the tool does and why it matters
2. Prerequisites: Docker + Docker Compose (or Python 3.11+)
3. Quick start:
   ```bash
   git clone ...
   cd ai-bias-reportcard
   docker compose up
   # Open http://localhost:8000
   # Upload backend/demo/heart_disease_demo.csv to see a live demo
   ```
4. Column mapping instructions (what ground truth / prediction / demographic mean)
5. Explanation of each metric in plain English (3–4 sentences each)
6. Screenshot placeholder: `![Report Card Screenshot](docs/screenshot.png)`

---

## Correctness Checklist

Before considering the project complete, verify all of the following:

- [ ] Uploading `heart_disease_demo.csv` with `target` as label, `prediction` as
      prediction, and `sex` + `age_group` as demographic columns produces a report
      without errors
- [ ] The report shows at least one WARN or FAIL for the demo data (it should —
      logistic regression on heart disease data reliably shows sex-based disparities)
- [ ] The report HTML is fully self-contained (no external URLs) — test by opening
      the downloaded file offline
- [ ] Uploading a CSV with a missing column returns a clear 422 error displayed in
      the UI, not a 500
- [ ] Uploading a CSV with a non-binary label column returns a clear error
- [ ] The "Download Report" button produces a valid `.html` file
- [ ] `docker compose up` starts the app with no errors and the UI loads at
      `http://localhost:8000`
- [ ] All metric computations handle groups with zero positive labels without
      crashing (return `null` for undefined metrics)

---

## What Good Looks Like

Running the tool with the demo dataset produces immediate, actionable insights:

1. Open the app at `http://localhost:8000`
2. Upload `backend/demo/heart_disease_demo.csv`
3. Map the columns: `target` as label, `prediction` as prediction, `sex` + `age_group` as demographics
4. Hit "Generate Report Card" (instant analysis)
5. The report displays: "BIAS DETECTED" banner in red — signaling subgroup disparities
6. Examine the `sex` section: female patients have recall ~0.62 vs male ~0.81 — a 19-percentage-point gap
7. Disparate Impact Ratio: 0.74 — below the legal 0.8 "four-fifths rule" threshold, flagged FAIL
8. Examine `age_group`: patients under 45 show 38.5% false negative rate vs 10% for 65+ — the model dangerously misses diagnoses in younger patients
9. Download and share the report as a standalone HTML file — it works offline and in email

The tool fulfills its core promise: **make demographic disparities visible in 60 seconds, without requiring data science expertise**.
