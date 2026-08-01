
# Heart Disease Risk Prediction — End-to-End ML Deployment

A machine learning web service that predicts whether a patient is at risk of heart disease based on clinical parameters. Built with **scikit-learn**, served via a **Flask REST API**, with a simple built-in web UI, and deployable for free on **Render**.

**Live app:** [https://heart-disease-deployment-s9f8.onrender.com](https://heart-disease-deployment-s9f8.onrender.com)

---

## Project Structure

```
HeartDiseaseDeployment/
├── app.py                 # Flask REST API + web UI (Task 3)
├── train_model.py         # Data preprocessing + model training (Task 1 & 2)
├── generate_dataset.py    # Creates heart.csv (see note below)
├── model.pkl              # Trained model, saved automatically by train_model.py
├── requirements.txt       # Python dependencies
├── heart.csv              # Dataset used for training
├── templates/
│   └── index.html         # Simple web UI
└── README.md

```

### ⚠️ About the Dataset — Read This First

The assignment points to a Kaggle dataset. This project ships with `generate_dataset.py`, which creates a `heart.csv` with the **exact same 13 clinical columns** as the Kaggle dataset (`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`), so everything runs immediately without needing a Kaggle account.

**For your actual submission, you should replace `heart.csv` with the real file** from:
[https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

To do that:

1. Download `heart.csv` from the Kaggle link above (Kaggle requires a free login to download).
2. Replace the `heart.csv` in this project folder with the downloaded one (keep the same file name).
3. Re-run `python train_model.py` to retrain the model on the real data.

Everything else (the API, the UI, the deployment) works unchanged either way, since the column names match exactly.

---

## Local Setup and Running

**Requirements:** Python 3.9+ installed on your machine.

```bash
# 1. Open a terminal inside the HeartDiseaseDeployment folder
cd HeartDiseaseDeployment

# 2. (Recommended) create a virtual environment
python -m venv venv
source ".\venv\Scripts\activate"   # on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Only needed if heart.csv is missing) generate the dataset
python generate_dataset.py

# 5. Train the model — this creates model.pkl
python train_model.py

# 6. Start the web app
python app.py

```

Now open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser. You'll see a form — fill in the patient's clinical details and click **Predict Risk**.

### Testing the API directly (optional)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

```

Expected response:

```json
{"prediction": "Heart Disease Detected", "risk_probability": 0.83}

```

---

## GitHub Deployment Setup

```bash
cd HeartDiseaseDeployment
git init
git add .
git commit -m "Heart disease prediction: model + Flask API + UI"

# Create a new PUBLIC repository on github.com named HeartDiseaseDeployment,
# then connect and push:
git remote add origin https://github.com/<your-username>/HeartDiseaseDeployment.git
git branch -M main
git push -u origin main

```

Make sure the repository is **public** and stays public until your submission is evaluated.

---

## Deploying to Render

Render's free "Web Service" tier costs nothing — no credit card needed for this kind of small app.

1. Go to **[https://render.com](https://render.com)** and sign up (you can sign up with your GitHub account directly — this makes the next step easier).
2. Click **New +** → **Web Service**.
3. Choose **Build and deploy from a Git repository**, then connect/select your `HeartDiseaseDeployment` GitHub repo.
4. Fill in the settings:
* **Name:** `heart-disease-deployment` (or anything you like — this becomes part of your URL)
* **Region:** any region close to you
* **Branch:** `main`
* **Runtime:** `Python 3`
* **Build Command:**
```bash
pip install -r requirements.txt && python train_model.py

```


*(This installs dependencies AND trains the model during deployment, so `model.pkl` is always freshly built from `heart.csv` in your repo.)*
* **Start Command:**
```bash
gunicorn app:app

```


* **Instance Type:** **Free**


5. Click **Create Web Service**. Render will build and deploy automatically — this takes a few minutes the first time.
6. Once it's live, Render gives you a public URL like:
`[https://heart-disease-deployment.onrender.com](https://heart-disease-deployment.onrender.com)`
7. Visit that URL to confirm the UI loads and predictions work.
8. Paste that URL at the top of this README (replacing the placeholder) and push the update to GitHub.

### Notes on Render's Free Tier

* Free web services **spin down after ~15 minutes of no traffic** and take 20–30 seconds to "wake up" on the next request. This is normal — just make sure to open the link yourself a few minutes before evaluation so it's warmed up, or mention this behavior if asked.
* No payment details are required for the Free instance type.

---

## Conclusion

The logistic regression model trained on the heart disease dataset achieved a test accuracy of approximately **78%**, showing that clinical parameters such as chest pain type, maximum heart rate, and ST depression carry meaningful predictive signal for heart disease risk. Performance could likely be improved further with a Random Forest or SVM model, hyperparameter tuning, or additional feature engineering. The main challenges during deployment were ensuring the Flask API and the trained model stayed in sync (the same feature order and scaling used in training must be reused at prediction time), and configuring Render's build and start commands so the model gets retrained automatically on deploy rather than depending on a stale local file. This project highlighted why MLOps practices matter: version-controlling code and data, automating model packaging, and using a repeatable deployment pipeline are what turn a one-off notebook experiment into a reliable, continuously usable service that other people (or systems) can depend on.

---

## Tech Stack

* **Model:** Logistic Regression (scikit-learn)
* **API:** Flask
* **UI:** Plain HTML/CSS/JS (served by Flask via Jinja templates)
* **Deployment:** Render (Free tier, Gunicorn WSGI server)
