# Low-tensor-rank RNNs

This package allows fitting a [low-tensor-rank recurrent neural network (ltrRNN)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/27030ad2ec1d8f2c3847a64e382c30ca-Abstract-Conference.html) to neural data recorded over learning.

---

<p align="center">
  <img width="700" src="https://raw.githubusercontent.com/arthur-pe/LtrRNN/main/img/ltrRNN.png">
</p>


## Installation 

```commandline
pip install slicetca
```

## Full documentation

The documentation can be found [here](https://github.com/arthur-pe/ltrRNN/blob/master/documentation.md).

## Examples

### Quick example 

```python
import slicetca
import torch
from matplotlib import pyplot as plt

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# your_data is a numpy array of shape (trials, neurons, time).
data = torch.tensor(your_data, dtype=torch.float, device=device)

# The real-time output of fitting can be found in the ./runs directory
ltrRNN.fit(data, condition) # condition is of shape (trial)
```

### Notebook

See the [example notebook](https://github.com/arthur-pe/ltrRNN/blob/master/ltrRNN_example_notebook.ipynb) for an application of ltrRNN to simulated data.

<a target="_blank" href="https://github.com/arthur-pe/ltrRNN/blob/master/ltrRNN_example_notebook.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Reference

A. Pellegrino<sub>@</sub>, N. A. Cayco-Gaijc<sup>†</sup>, A. Chadwick<sup>†</sup>. (2024). Low Tensor Rank Learning of Neural Dynamics. [https://proceedings.neurips.cc/paper_files/paper/2023/hash/27030ad2ec1d8f2c3847a64e382c30ca-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/27030ad2ec1d8f2c3847a64e382c30ca-Abstract-Conference.html).