import io
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from metrics import compute_fairness_metrics
from report import render_report

app = FastAPI(title="AI Bias Report Card")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _parse_upload(file_bytes: bytes, label_col: str, prediction_col: str, demographic_cols: list[str]):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    missing = [c for c in [label_col, prediction_col] + demographic_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {', '.join(missing)}")

    for col in [label_col, prediction_col]:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise HTTPException(
                status_code=422,
                detail=f"Column '{col}' must be binary (0/1). Found values: {sorted(unique_vals)[:10]}"
            )

    return df


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    label_col: str = Form(...),
    prediction_col: str = Form(...),
    demographic_cols: str = Form(...),
):
    demo_cols = json.loads(demographic_cols)
    file_bytes = await file.read()
    df = _parse_upload(file_bytes, label_col, prediction_col, demo_cols)
    metrics = compute_fairness_metrics(df, label_col, prediction_col, demo_cols)
    return metrics


@app.post("/report", response_class=HTMLResponse)
async def report(
    file: UploadFile = File(...),
    label_col: str = Form(...),
    prediction_col: str = Form(...),
    demographic_cols: str = Form(...),
):
    demo_cols = json.loads(demographic_cols)
    file_bytes = await file.read()
    df = _parse_upload(file_bytes, label_col, prediction_col, demo_cols)
    metrics = compute_fairness_metrics(df, label_col, prediction_col, demo_cols)
    html = render_report(metrics, file.filename or "upload.csv")
    return HTMLResponse(content=html)
