# VishieldAPI

VishieldAPI is a small repository that combines:
- a **FastAPI** service exposing a simple **text classification** endpoint (e.g., “safe” vs “phishing”),
- a **training notebook** used to explore datasets and fine-tune a Transformer model,
- a **sample SMS dataset** and a ready-to-use HTTP file to quickly test the API.

---

## Repository structure

- `api.py` — FastAPI application that loads a pretrained text classification model and exposes prediction endpoints.
- `training.ipynb` — Jupyter notebook for dataset exploration, preprocessing, training / evaluation, and model publishing.
- `dataset_sms.csv` — SMS dataset (labels + text + metadata columns).
---

## Requirements

- Python 3.x
- Common ML/NLP libraries (Transformers, PyTorch, etc.)
- FastAPI + an ASGI server (typically `uvicorn`)

---

## Running the API locally

1. Start the ASGI server

- `http://127.0.0.1:8000/` for a basic health check
- `http://127.0.0.1:8000/docs` for Swagger UI

Once the API is running, you can send a request to the prediction route (see `/predit/?text=YOUR_TEXT_TO_CLASSIFY` for ready-made examples).  
Typical workflow:
- send a text payload/query parameter,
- the API returns a predicted **label** and its **class id**.

---

## Training / fine-tuning

The notebook `training.ipynb` covers the end-to-end experimentation pipeline:
- loading datasets (SMS + email sources),
- text cleaning / normalization,
- visualization and dataset analysis,
- fine-tuning a Transformer model and evaluating results.

