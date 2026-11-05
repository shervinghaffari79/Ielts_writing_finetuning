# data/

Put the raw spreadsheet here (or upload it in Colab):

    data/IELTS_Task2_565_cleaned_minimal.xlsx

`python -m ielts_ft.data_prep` writes `train.jsonl`, `validation.jsonl`, `test.jsonl`
and `data_report.json` into `data/processed/`. These files are git-ignored because
they contain student essays.
