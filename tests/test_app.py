"""
Headless test of the Streamlit app (no browser needed).  Run:  pytest
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app" / "app.py"
MODEL_READY = (PROJECT_ROOT / "models" / "best_baseline.joblib").exists()


@pytest.mark.skipif(not MODEL_READY, reason="run python -m src.train_baselines first")
def test_app_analyzes_a_sentence():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    app.text_area[0].input("movie bahut mast thi").run()
    app.button[0].click().run()  # the "Analyze" button
    assert not app.exception
    assert app.session_state["result"]["label"] in ["negative", "neutral", "positive"]


@pytest.mark.skipif(not MODEL_READY, reason="run python -m src.train_baselines first")
def test_example_button_fills_the_text_box():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button[1].click().run()  # first example button
    assert not app.exception
    assert app.session_state["input_text"] != ""
    assert app.session_state["result"] is not None
