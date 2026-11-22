"""Gradio front end for the tuned credit-default random forest."""
import gradio as gr

from model import FEATURES, THRESHOLD, build, predict_one

# Train once when the Space starts (a few seconds on the free CPU tier).
MODEL, METRICS = build()

PAY_STATUS = [
    ("Paid in full / no use (-2)", -2), ("Paid duly (-1)", -1), ("Revolving credit (0)", 0),
    ("1 month late", 1), ("2 months late", 2), ("3 months late", 3), ("4 months late", 4),
    ("5 months late", 5), ("6 months late", 6), ("7 months late", 7), ("8+ months late", 8),
]
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]  # PAY_0 is the most recent month


def predict(*args):
    values = dict(zip(FEATURES, args))
    p = predict_one(MODEL, values)
    decision = "Likely to DEFAULT" if p >= THRESHOLD else "Unlikely to default"
    explanation = (
        f"**{decision}** — predicted probability of default: **{p:.1%}** "
        f"(decision threshold {THRESHOLD:.0%}).\n\n"
        "Probabilities are inflated by class weighting, so read them as a risk "
        "ranking rather than a calibrated chance."
    )
    return {"Default": p, "No default": 1 - p}, explanation


metrics_md = (
    f"Test set ({METRICS['n_test']:,} clients): accuracy {METRICS['accuracy']:.1%}, "
    f"precision {METRICS['precision']:.3f}, recall {METRICS['recall']:.3f}, "
    f"F1 {METRICS['f1']:.3f}, ROC-AUC {METRICS['roc_auc']:.3f}"
)

with gr.Blocks(title="Credit Card Default Predictor") as demo:
    gr.Markdown(
        "# Credit Card Default Predictor\n"
        "Tuned random forest trained on the UCI *Default of Credit Card Clients* "
        "dataset (Taiwan, 2005). Amounts are in New Taiwan dollars. "
        "For coursework demonstration only — not for real lending decisions.\n\n"
        + metrics_md
    )
    with gr.Row():
        limit_bal = gr.Number(label="Credit limit (LIMIT_BAL)", value=50000)
        sex = gr.Dropdown([("Male", 1), ("Female", 2)], value=2, label="Sex")
        education = gr.Dropdown(
            [("Graduate school", 1), ("University", 2), ("High school", 3), ("Other", 4)],
            value=2, label="Education")
        marriage = gr.Dropdown([("Married", 1), ("Single", 2), ("Other", 3)], value=2,
                               label="Marital status")
        age = gr.Number(label="Age", value=35, precision=0)
    gr.Markdown("### Repayment status (most recent month first)")
    with gr.Row():
        pay = [gr.Dropdown(PAY_STATUS, value=0, label=m) for m in MONTHS]
    gr.Markdown("### Bill statement amounts")
    with gr.Row():
        bills = [gr.Number(label=m, value=20000) for m in MONTHS]
    gr.Markdown("### Amounts paid")
    with gr.Row():
        pays = [gr.Number(label=m, value=2000) for m in MONTHS]

    button = gr.Button("Predict", variant="primary")
    label_out = gr.Label(label="Prediction")
    text_out = gr.Markdown()
    inputs = [limit_bal, sex, education, marriage, age, *pay, *bills, *pays]
    button.click(predict, inputs=inputs, outputs=[label_out, text_out])

if __name__ == "__main__":
    demo.launch()
