# BeverageHealth AI

Mini project: **AI-Based Beverage Ingredient Analysis and Health Impact Prediction Using Machine Learning**

## Run online without VS Code (recommended)
1. Create/sign in to GitHub.
2. Create a repository named `BeverageHealthAI`.
3. Upload these files from this folder: `app.py`, `beverage_dataset.csv`, `requirements.txt`, and `README.md`.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app** → **Yup, I have an app**.
6. Select your repository, branch (`main`) and main file path (`app.py`), then click **Deploy**.
7. When deployment finishes, open the generated `*.streamlit.app` URL.

## Run in Google Colab
This project is a Streamlit web app, so Colab alone does not directly show the same web UI without extra tunnelling setup. Streamlit Community Cloud is the simplest route for a shareable browser link.

## Run locally (optional)
Install Python, then run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit interface and ML training/evaluation.
- `beverage_dataset.csv`: synthetic example dataset for demonstration.
- `requirements.txt`: required Python packages.

## Important limitation
The included dataset is synthetic and the labels are rule-derived. Reported accuracy only measures agreement with those example labels. It does **not** measure real health-outcome prediction. Do not present this as medical advice or a clinically validated model. For a stronger academic version, replace the sample dataset with a verifiable real beverage nutrition dataset and cite its source.
