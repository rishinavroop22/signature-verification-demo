# Offline Signature Verification – Demo

Web demo of a Siamese CNN + Transformer that labels a questioned signature as **Genuine**, **Forged** or
**AI-generated**, and says whether it comes from the same person as the known genuine signatures.
Trained on CEDAR with synthetic signatures from a cGAN, a WGAN-GP and a DDPM. A model is included for each
writer split (70:30, 60:40 and 80:20). The Results tab shows the full evaluation tables and graphs.

## Run locally

```
pip install -r requirements.txt
streamlit run demo_app/app.py
```

## Deploy

* **Streamlit Community Cloud:** New app → this repository → main file `demo_app/app.py`.
* **Render / Railway / Hugging Face Spaces (Docker):** the included `Dockerfile` runs on CPU and listens on `$PORT`.

Results were measured on writers not seen in training. It is a student project, trained on CEDAR scans only,
and is not proof of forgery.
