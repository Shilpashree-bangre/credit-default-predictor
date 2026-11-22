---
title: Credit Card Default Predictor
emoji: 💳
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
---

# Credit Card Default Predictor

Tuned random forest (Module 8) predicting whether a credit card client will default
next month, trained on the UCI Default of Credit Card Clients dataset (Yeh, 2009).

Model: 142 trees, max depth 20, min samples per leaf 15, max features log2,
class weight balanced_subsample, seed 20, decision threshold 0.5.

Hosted on a free **Static** Hugging Face Space. Free accounts can no longer run
Gradio apps on CPU Basic, and Gradio-Lite currently fails to start because of a
huggingface-hub version conflict, so `index.html` uses Pyodide directly: it runs
Python and scikit-learn inside the visitor's browser, loads `model.py` and the
dataset, and trains the model on page load. `app.py` is the Gradio version for
running locally.

Every push to `main` runs the tests on GitHub Actions and, if they pass,
syncs the repository to the Space.