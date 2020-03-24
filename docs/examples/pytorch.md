## Training

```python
{!./examples/training/pytorch.py!}
```

To run it :

```bash
python3 -m examples.training.pytorch
```

## Serving

```python
{!./examples/serving/pytorch.py!}
```

To run it :

```bash
uvicorn examples.serving.pytorch:app --host 0.0.0.0
```
