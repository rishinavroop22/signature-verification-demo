"""Signature verification demo.   streamlit run demo_app/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Member4_Detector")]
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from common.imageproc import IMG_H, IMG_W, canonical_ink, to_uint8
from common.paths import RATIOS, RESULTS, RUNS
from model import CLASSES, SiameseCNNTransformer

st.set_page_config(page_title="Signature Verification", layout="wide")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load(ratio):
    p = RUNS / "detector" / ratio / "E5_all" / "best.pt"
    if not p.exists():
        return None
    net = SiameseCNNTransformer(pretrained=False).to(DEV).eval()
    net.load_state_dict(torch.load(p, map_location=DEV, weights_only=True)["net"])
    return net


def read(f):
    a = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None:
        return None
    if a.shape == (IMG_H, IMG_W) and np.median(a) == 255:       # already cleaned (project dataset or generated image)
        return (255 - a).astype(np.float32) / 255
    if max(a.shape) > 900:
        k = 900 / max(a.shape)
        a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return canonical_ink(a)


st.title("Offline signature verification")
tab1, tab2 = st.tabs(["Check a signature", "Results"])

with tab1:
    ratio = st.selectbox("Model (dataset split)", RATIOS, format_func=lambda r: r.replace("_", ":"))
    net = load(ratio)
    if net is None:
        st.warning("This model has not been trained yet.")
    c1, c2 = st.columns(2)
    refs = c1.file_uploader("Known genuine signatures (1 to 5 images)", ["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
    q = c2.file_uploader("Questioned signature", ["png", "jpg", "jpeg", "tif"])
    if net is not None and refs and q:
        ri = [x for x in (read(f) for f in refs[:5]) if x is not None and (x > 0.5).mean() > 0.003]
        qi = read(q)
        if not ri or qi is None or (qi > 0.5).mean() < 0.003:
            st.error("No signature was found in one of the images. Use dark ink on a light, plain background.")
        else:
            c1.image([to_uint8(x) for x in ri], caption=[f"Known {i + 1} (cleaned)" for i in range(len(ri))], width=230)
            c2.image(to_uint8(qi), caption="Questioned (cleaned)", width=340)
            r = torch.tensor(np.stack(ri))[:, None].to(DEV)
            t = torch.tensor(qi)[None, None].to(DEV).expand(len(ri), -1, -1, -1)
            with torch.no_grad():
                lo, _, sm = net(r, t)
                p = lo.softmax(1).mean(0).cpu().numpy()
                same = float(sm.sigmoid().mean())
            k = int(p.argmax())
            {0: st.success, 1: st.error, 2: st.warning}[k](f"Result: {CLASSES[k]}  ({p[k]:.1%} confidence)")
            for n, v in zip(CLASSES, p):
                st.progress(float(v), text=f"{n}: {v:.1%}")
            thr = 0.5
            f = RESULTS / "same_person_metrics.csv"
            if f.exists():
                d = pd.read_csv(f)
                d = d[(d.ratio == ratio) & (d.experiment == "E5_all")]
                thr = float(d.threshold.iloc[0]) if len(d) else thr
            st.subheader("Same person" if same >= thr else "Different person")
            st.progress(same, text=f"Same-person score: {same:.1%} (decision threshold {thr:.1%})")
            st.caption("Trained on CEDAR only. Phone photos and other scanners can give less reliable results. "
                       "This is a student project, not proof of forgery.")

with tab2:
    for title, f in (("Detection network", "detector_metrics.csv"), ("Generative models", "generator_metrics.csv"),
                     ("Same person / different person", "same_person_metrics.csv"), ("Robustness", "robustness.csv"), ("Extra test on SID", "sid_metrics.csv")):
        if (RESULTS / f).exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(RESULTS / f), width="stretch", hide_index=True)
    figs = sorted((RESULTS / "figures").glob("*.png")) if (RESULTS / "figures").exists() else []
    for f in figs:
        st.image(f.read_bytes(), caption=f.stem.replace("_", " "))
