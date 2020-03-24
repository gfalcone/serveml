## Training

```python
{!./examples/training/keras.py!}
```

To run it :

```bash
python3 -m examples.training.keras
```

## Serving

```python
{!./examples/serving/keras.py!}
```

To run it :

```bash
uvicorn examples.serving.keras:app --host 0.0.0.0
```
