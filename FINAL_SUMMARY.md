# HinglishSense: final summary

**Built:** Hinglish sentiment classifier (positive/negative/neutral). It has a data loader, Hinglish preprocessing (363-entry spelling map, emoji tokens, elongation fix), TF-IDF word + char n-grams, NB/LR/SVM with validation tuning and an ablation study, test evaluation with plots and error analysis, a Streamlit app, 33 passing tests, an executed demo notebook, a Colab transformer notebook, a README and report/PPT/viva content.
**Dataset:** REAL data, not demo: SentiMix Hinglish (SemEval-2020 Task 9, HF mirror `RTT1/SentiMix`). 19,675 tweets, re-split 70/15/15 stratified, so scores are not comparable to the SemEval leaderboard.

| Model (test set, 2,952 tweets) | Accuracy | Macro-F1 |
|---|---|---|
| **Linear SVM (C=0.1, calibrated), saved as best_baseline** | **0.637** | **0.640** |
| Logistic Regression (C=1) | 0.626 | 0.629 |
| Multinomial Naive Bayes (alpha=1) | 0.619 | 0.622 |

**Ablation:** char n-grams give the biggest gain (word-only 0.615 → char-only 0.636 test). The normalization map adds +0.1 to +0.3. NEG_ marking did not help on validation, so it is off.
**Run the app:** `.venv\Scripts\activate` then `streamlit run app/app.py` (verified with AppTest and a headless start).
**Rebuild everything:** `python -m src.data_loader && python -m src.train_baselines && python -m src.evaluate && pytest`
**Skipped:** XLM-RoBERTa fine-tuning, because this machine has no GPU (torch not installed). Use `notebooks/colab_transformer_finetune.ipynb`.
**Known weaknesses:** neutral class (≈89% of errors), negation ("khana accha nahi tha" → neutral), sarcasm, label noise, and domain shift (political/cricket tweets, not reviews).

**Next steps for the team:**
1. Check every `[VERIFY REFERENCE]` in `results/report_content.md` against the real papers.
2. Optional: run the Colab notebook, unzip the model into `models/transformer/`, rerun `python -m src.evaluate`, and add the row.
3. Read `BUILD_LOG.md` and the code comments, and rehearse the viva Q&A (section 12 of report_content.md).
4. Take app screenshots for the report and PPT, and fill in names and guide on slide 1.
