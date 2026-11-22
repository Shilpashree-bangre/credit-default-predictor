# Deployment guide: Hugging Face Space + GitHub Actions

## 1. Run it locally first
    python -m venv .venv && .venv\Scripts\activate      (Windows)
    pip install -r requirements.txt pytest
    python model.py        # prints test metrics; expect F1 about 0.53
    pytest -v              # all 4 tests should pass
    python app.py          # opens the app at http://127.0.0.1:7860

## 2. Create the Space (free tier)
huggingface.co -> New -> Space. Name it (e.g. credit-default-predictor),
SDK: Gradio, Hardware: **CPU basic (free)**, Public. Leave it empty.

## 3. Create a Hugging Face token
Settings -> Access Tokens -> New token, type **Write**. Copy it.

## 4. Create the GitHub repo
New repository (e.g. credit-default-predictor), then push this folder:
    git init -b main
    git add .
    git commit -m "Initial deployment of tuned random forest"
    git remote add origin https://github.com/<you>/credit-default-predictor.git
    git push -u origin main

## 5. Add the token as a GitHub secret
Repo -> Settings -> Secrets and variables -> Actions -> New repository secret.
Name: HF_TOKEN, value: the token from step 3.

## 6. Edit the workflow
In .github/workflows/ci-cd.yml set HF_USERNAME and HF_SPACE, commit and push.
Watch the Actions tab: test and check-file-size run in parallel, deploy runs
only if both pass. Then open the Space; the build log shows the pip install
and startup, and the app appears when it says "Running".

## 7. Prove the pipeline works (good evidence for the report)
- Make a small change (e.g. the app title), push, and watch it redeploy.
- Open a pull request: tests run but deploy is skipped.
- Optional: temporarily raise the F1 gate in tests/test_model.py to 0.60,
  push, and show the deploy job being blocked. Then revert.

## Screenshots to take
Space running with a prediction; Actions run with all three jobs green;
a blocked run (if you did the gate test); Space settings showing CPU basic;
the HF_TOKEN secret listed (value hidden).

## Notes for your report (fill in as you go)
- Space URL:
- GitHub repo URL:
- Time from push to Space running:
- Errors you hit and how you fixed them:
- Anything surprising:
