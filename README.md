
# CQT - Jax


An invertible and differentiable implementation of the Constant-Q Transform (CQT) using Non-stationary Gabor Transform (NSGT), in Jax.

```bash
pip install cqt-jax
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/cqt-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/cqt-pytorch/)


## Usage

```python
import jax.numpy as jnp
from cqt_jax import CQT

# Initialize the CQT transform
transform = CQT(
    num_octaves=8,
    num_bins_per_octave=64,
    sample_rate=48000,
    block_length=2 ** 18
)

# Generate a random audio waveform tensor x
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1, 2, 2**18))  # [1, 2, 262144] = [batch_size, channels, timesteps]

# Encode the waveform
z = transform.encode(x)  # [1, 2, 512, 2839] = [batch_size, channels, frequencies, time]

# Decode the transformed signal
y = transform.decode(z)  # [1, 2, 262144]

print(f"Original shape: {x.shape}")
print(f"Encoded shape: {z.shape}")
print(f"Decoded shape: {y.shape}")
```

### Example CQT Magnitude Spectrogram (z)
<img src="./IMAGE.png"></img>

## TODO
* [x] Power of 2 length (with `power_of_2_length` constructor arg).
* [x] Understand why/if inverse window is necessary (it is necessary for perfect inversion).
* [x] Allow variable audio lengths by chunking (now input can be a multiple of `block_length`)

## Appreciation
Special thanks to [Eloi Moliner](https://github.com/eloimoliner) for taking the time to help me understand how CQT works. Check out his own implementation with interesting features at [eloimoliner/CQT_pytorch](https://github.com/eloimoliner/CQT_pytorch).

## Citations

```bibtex
@article{1210.0084,
Author = {Nicki Holighaus and Monika Dörfler and Gino Angelo Velasco and Thomas Grill},
Title = {A framework for invertible, real-time constant-Q transforms},
Year = {2012},
Eprint = {arXiv:1210.0084},
Doi = {10.1109/TASL.2012.2234114},
}
```
