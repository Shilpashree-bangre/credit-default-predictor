---
title: Credit Card Default Predictor
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
---

# Credit Card Default Predictor

Tuned random forest (Module 8) predicting whether a credit card client will default
next month, trained on the UCI Default of Credit Card Clients dataset (Yeh, 2009).

Model: 142 trees, max depth 20, min samples per leaf 15, max features log2,
class weight balanced_subsample, seed 20, decision threshold 0.5.

Deployed to a free CPU Basic Hugging Face Space through GitHub Actions: every push
to `main` runs the tests and, if they pass, syncs the repository to the Space.
