## Training

```python
{!./examples/training/prophet.py!}
```

To run it :

```bash
python3 -m examples.training.prophet
```

## Serving

```python
{!./examples/serving/prophet.py!}
```

To run it :

```bash
uvicorn examples.serving.prophet:app --host 0.0.0.0
```
