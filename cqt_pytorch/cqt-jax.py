import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Optional

def next_power_of_2(x):
    return 2 ** jnp.ceil(jnp.log2(x)).astype(int)

def get_center_frequencies(num_octaves, num_bins_per_octave, sample_rate):
    frequency_nyquist = sample_rate / 2
    frequency_min = frequency_nyquist / (2**num_octaves)
    num_bins = num_octaves * num_bins_per_octave
    frequencies = frequency_min * (2 ** (jnp.arange(num_bins) / num_bins_per_octave))
    return jnp.concatenate([
        frequencies,
        jnp.array([frequency_nyquist]),
        sample_rate - jnp.flip(frequencies)
    ])

def get_bandwidths(num_octaves, num_bins_per_octave, sample_rate, frequencies):
    num_bins = num_octaves * num_bins_per_octave
    q = 1.0 / (2 ** (1.0 / num_bins_per_octave) - 2 ** (-1.0 / num_bins_per_octave))
    bandwidths = frequencies[1:num_bins+1] / q
    bandwidths_symmetric = jnp.flip(frequencies[1:num_bins+1]) / q
    return jnp.concatenate([
        bandwidths,
        jnp.array([sample_rate - 2 * frequencies[num_bins]]),
        bandwidths_symmetric
    ])

def get_windows(lengths, max_length):
    num_bins = lengths.shape[0] // 2
    pad_left = lambda length: jnp.floor(max_length / 2 - length / 2).astype(int)
    pad_right = lambda length: max_length - length - pad_left(length)
    
    create_window = lambda length: jnp.pad(
        jnp.hanning(int(length)),
        (pad_left(length), pad_right(length))
    )
    
    return vmap(create_window)(lengths[:num_bins])

class CQT:
    def __init__(self, num_octaves, num_bins_per_octave, sample_rate, block_length=None):
        self.block_length = block_length or sample_rate
        frequencies = get_center_frequencies(num_octaves, num_bins_per_octave, sample_rate)
        bandwidths = get_bandwidths(num_octaves, num_bins_per_octave, sample_rate, frequencies)
        
        window_lengths = jnp.round(bandwidths * self.block_length / sample_rate)
        self.max_window_length = next_power_of_2(int(window_lengths.max()))
        
        positions = jnp.round(frequencies * self.block_length / sample_rate)
        self.windows_range_idx = (jnp.arange(self.max_window_length)[None, :] + 
                                  (positions[:, None] - self.max_window_length // 2)) % self.block_length
        
        self.windows = get_windows(window_lengths, self.max_window_length)
        self.windows_inverse = self._get_windows_inverse()
    
    def _get_windows_inverse(self):
        windows_overlap = jnp.zeros(self.block_length).at[self.windows_range_idx].add(self.windows**2)
        return self.windows / (windows_overlap[self.windows_range_idx] + 1e-8)
    
    @partial(jit, static_argnums=(0,))
    def encode(self, waveform):
        n_blocks = waveform.shape[-1] // self.block_length
        waveform = waveform.reshape(-1, self.block_length)
        
        frequencies = jnp.fft.fft(waveform)
        crops = frequencies[:, self.windows_range_idx]
        crops_windowed = crops * self.windows
        transform = jnp.fft.ifft(crops_windowed)
        
        return transform.reshape(-1, n_blocks, self.max_window_length)
    
    @partial(jit, static_argnums=(0,))
    def decode(self, transform):
        b, n, _ = transform.shape
        transform = transform.reshape(b * n, -1)
        
        crops_windowed = jnp.fft.fft(transform)
        crops = crops_windowed * self.windows_inverse
        
        frequencies = jnp.zeros((b * n, self.block_length), dtype=complex)
        frequencies = frequencies.at[self.windows_range_idx].add(crops)
        
        waveform = jnp.fft.irfft(frequencies, self.block_length)
        return waveform.reshape(b, -1)
