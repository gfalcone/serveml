## Training

```python
{!./examples/training/tensorflow.py!}
```

To run it :

```bash
python3 -m examples.training.tensorflow
```

## Serving

```python
{!./examples/serving/tensorflow.py!}
```

To run it :

```bash
uvicorn examples.serving.tensorflow:app --host 0.0.0.0
```
