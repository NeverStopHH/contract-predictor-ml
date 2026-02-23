# Contract Predictor – End-to-End ML Pipeline

Production-oriented machine learning pipeline for predicting retail lease contract outcomes.

The project demonstrates modular ML engineering practices including:

- Shared preprocessing layer
- Feature engineering abstraction
- Model-agnostic training scripts
- Evaluation with threshold analysis
- Benchmark comparison across model classes

---

## Architecture

data → preprocessing → model training → evaluation → benchmarking → reports

- `src/preprocessing.py` – shared data preparation
- `src/train_dt_rf.py` – tree-based models
- `src/train_nn.py` – neural network (TensorFlow)
- `src/evaluate.py` – metrics + visual diagnostics
- `src/benchmark.py` – unified model comparison

---

## Models Compared

- Decision Tree
- Random Forest
- Deep Neural Network (Embedding + Dense layers)

---

## Results

| Model | Accuracy | ROC-AUC | PR-AUC |
|--------|----------|----------|----------|
| Decision Tree | 0.64 | 0.69 | 0.65 |
| Random Forest | 0.63 | 0.73 | 0.70 |
| Neural Network | 0.73 | 0.79 | 0.76 |

Neural networks showed superior ranking performance (ROC-AUC, PR-AUC),
while tree-based models provided faster training times.

---

## Reproducibility

```bash
conda activate contractpredictor
pip install -r requirements.txt

python -m src.train_dt_rf
python -m src.train_nn
python -m src.benchmark
