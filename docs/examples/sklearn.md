## Training

```python
{!./examples/training/sklearn.py!}
```

To run it : 

```bash
python3 -m examples.training.sklearn
```

## Serving

```python
{!./examples/serving/sklearn.py!}
```

To run it : 

```bash
uvicorn examples.serving.sklearn:app --host 0.0.0.0
```
