# CholeskySynth Time-Series Simulator

CholeskySynth is a multivariate time-series simulator that samples from a matrix-normal distribution using covariance factors constructed from a flexible bank of Gaussian process kernels. It generalizes the univariate KernelSynth procedure by pairing temporal and cross-variate covariances and drawing samples via Cholesky decomposition. This repository hosts a Jupyter notebook implementation that can synthesize large batches of realistic multivariate trajectories for downstream learning objectives.

- Generates long, multi-variate series with controllable temporal and cross-channel structure.
- Builds covariance factors by randomly composing Gram matrices derived from common kernels.
- Produces numerically stable samples by adding diagonal jitter before Cholesky factorization.
- Streams batches through a `tf.data.Dataset` writer for large-scale dataset creation.

## Repository Layout

- `cholesky_synth.ipynb` — end-to-end notebook containing all kernels, sampling routines, visualization helpers, and dataset exporters.
- `requirements.txt` — CPU-oriented dependencies for running the notebook.

## Prerequisites

- Python 3.10+ (tested with recent CPython/JAX releases).
- Conda, `venv`, or another environment manager is recommended to isolate dependencies.
- Latest NVIDIA drivers and CUDA 12.x if you intend to execute the GPU setup path.

## Installation

Create and activate an environment, then install the requirements. Two setup paths are provided in the notebook:

### CPU-only workflow

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install jax==0.6.2 tensorflow-cpu==2.17.0
```

### GPU workflow (CUDA 12)

```bash
pip install --upgrade pip
pip install "jax[cuda12]==0.6.2"
pip install "tensorflow[and-cuda]"
pip install -r requirements.txt
```

The notebook contains optional `%pip uninstall` steps that help clear conflicting GPU builds; run them only if you need to switch backends on a shared machine.

## Getting Started

1. Launch JupyterLab or VS Code and open `cholesky_synth.ipynb`.
2. Run the environment setup cell that matches your hardware (CPU or GPU).
3. Execute the remaining cells to import libraries, define kernels, generate samples, and optionally persist datasets.

A minimal Python snippet from the notebook illustrates how to draw a batch of matrix-normal samples:

```python
import jax
from jax import random

key = random.key(0)
batch_size = 10        # number of series
length = 2500          # time steps per series
num_variates = 10      # channels per series
num_time_kernels = 4   # temporal kernel components
num_variate_kernels = 3
eps = 1e-2             # diagonal jitter for stability

ts_batch, key = generate_matrix_normal_dataset(
    key,
    num_samples=batch_size,
    num_rows=length,
    num_cols=num_variates,
    num_time_kernels=num_time_kernels,
    num_variate_kernels=num_variate_kernels,
    eps=eps,
)
```

Use `plot_time_series_grid(ts_batch)` to visualize generated trajectories.

## Algorithm Overview

CholeskySynth minimizes the simulator objective described in Equation (1) of the accompanying manuscript by sampling multivariate time series from a matrix-normal distribution:

1. **Kernel bank** — Temporal (`U`) and cross-variate (`V`) covariances are constructed from Gram matrices computed with constant, white-noise, polynomial, RBF, rational-quadratic, and periodic kernels at multiple hyper-parameter settings.
2. **Random convolution** — For each draw, the notebook samples a subset of Gram matrices for both `U` and `V` and recursively combines them with random additive or multiplicative operators to emulate kernel composition.
3. **Numerical stabilization** — A configurable jitter `eps` is added to the diagonals of `U` and `V` to guarantee positive definiteness prior to Cholesky factorization.
4. **Matrix-normal sampling** — Standard normal noise is transformed with the Cholesky factors `A` and `B`, yielding samples with covariance `U ⊗ V`. This provides an efficient stand-in for a general matrix-normal sampler, which is not available off-the-shelf in JAX.

The implementation scales cubically with both the number of time steps and variates due to the Cholesky decompositions, with quadratic memory usage. Adjust `length`, `num_variates`, and kernel counts carefully when targeting very large batches.

## Data Generation Pipeline

- `create_data_batch` wraps `generate_matrix_normal_dataset` and guards against NaNs by regenerating batches if necessary.
- `data_generator` streams batches with randomized kernel counts into a `tf.data.Dataset`, enabling asynchronous prefetching and on-disk persistence.
- Batches are saved with timestamped directories under `../data/tempo_v1_largest_*` by invoking `dataset.save(...)`. Ensure the parent directory exists when running outside the notebook root.

To export a million samples one series at a time, adjust `batch_size`, `length`, and the kernel ranges near the bottom of the notebook before executing the save cell.

## Customization Tips

- **Kernel diversity**: Extend `compute_all_gram_matrices` with additional kernels or alternate hyper-parameter grids to broaden the covariance family.
- **Stability vs. fidelity**: Increase `eps` for more aggressive jitter if you encounter decomposition failures; decrease it to retain sharper correlations once the setup is stable.
- **Performance**: Leverage GPUs (or TPUs via JAX) when generating very large datasets. Consider reducing `num_time_kernels`/`num_variate_kernels` to shorten kernel composition chains.

## Acknowledgements

CholeskySynth is a multivariate extension of KernelSynth [7], relying on properties of the matrix-normal distribution [9]. The implementation here packages the sampling routine, kernel compositions, and dataset writer in a single notebook for reproducibility and reuse.

---

For questions or contributions, please open an issue or submit a pull request once the repository is expanded beyond the notebook prototype.
