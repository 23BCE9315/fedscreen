# FedScreen — Streamlit deployment

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run fedscreen_app.py
```

The app expects `fedscreen_fedprox_global_model.pth` in the same directory as `fedscreen_app.py`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload these four files:
   - `fedscreen_app.py`
   - `fedscreen_model.py`
   - `fedscreen_fedprox_global_model.pth`
   - `requirements.txt`
2. Open Streamlit Community Cloud and create a new app from that repository.
3. Set the main file to `fedscreen_app.py`.
4. Deploy.

The app no longer depends on Google Colab paths and does not download ImageNet weights at startup because the supplied trained checkpoint already contains the backbone weights.
