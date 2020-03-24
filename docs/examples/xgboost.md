## Training

```python
{!./examples/training/xgboost.py!}
```

To run it :

```bash
python3 -m examples.training.xgboost
```

## Serving

```python
{!./examples/serving/xgboost.py!}
```

To run it :

```bash
uvicorn examples.serving.xgboost:app --host 0.0.0.0
```
