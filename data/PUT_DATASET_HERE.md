Save the dataset here as `default_of_credit_card_clients.csv` (about 2.8 MB).

If you only have the original UCI `.xls` file, convert it once on your laptop:

    import pandas as pd
    df = pd.read_excel("default of credit card clients.xls", header=1)  # needs: pip install xlrd
    df.to_csv("data/default_of_credit_card_clients.csv", index=False)

Kaggle's `UCI_Credit_Card.csv` also works as-is (just rename it).
Delete this note once the CSV is in place if you like.
