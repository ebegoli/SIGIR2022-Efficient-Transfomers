#!/usr/bin/env python

import torch
import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook
from time import perf_counter
from datasets import load_metric
from pathlib import Path

class PerformanceBenchmark:
  def __init__(self, pipeline, dataset, optim_type='BERT baseline'):
    self.pipeline = pipeline
    self.dataset = dataset
    self.optim_type = optim_type
    self.accuracy_metric = load_metric('accuracy')
    self.tmp_dir = Path('../project_dir')
    
  def compute_accuracy(self, class_labels):
    preds, labels = [], []
    for example in tqdm_notebook(self.dataset, total=len(self.dataset), desc='Computing Accuracy'):
      pred = self.pipeline(example['text'])[0]['label']
      label = example['intent']
      preds.append(class_labels.str2int(pred))
      labels.append(label)
    accuracy = self.accuracy_metric.compute(predictions=preds, references=labels)
    print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
    return accuracy

  def compute_size(self):
    state_dict = self.pipeline.model.state_dict()
    tmp_path = self.tmp_dir/'model.pt'
    torch.save(state_dict, tmp_path)
    # Calculate size in megabytes
    size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    # Delete temporary file
    tmp_path.unlink()
    print(f"Model size (MB) - {size_mb:.2f}")
    return {'size_mb': size_mb}

  def time_pipeline(self, query="What is the pin number for my account?"):
    latencies = []
    # Warmup
    for _ in range(10):
        _ = self.pipeline(query)
    # Timed run
    for _ in range(100):
        start_time = perf_counter()
        _ = self.pipeline(query)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
    return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

  def run_benchmark(self, class_labels):
    metrics = {}
    metrics[self.optim_type] = self.compute_accuracy(class_labels)
    # metrics[self.optim_type] = self.compute_size()
    metrics[self.optim_type].update(self.time_pipeline())
    metrics[self.optim_type].update(self.compute_size())
    return metrics