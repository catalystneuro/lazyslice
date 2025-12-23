"""Lazy transposing and slicing operations for h5py datasets and zarr arrays.

This package provides the `DatasetView` class for composing lazy slice and
transpose operations on h5py datasets and zarr arrays without reading data
until explicitly requested.

Example
-------
>>> from lazyslice import DatasetView
>>> import h5py
>>>
>>> with h5py.File('data.h5', 'r') as f:
...     view = DatasetView(f['dataset'])
...     sliced = view.lazy_slice[10:20, :, 5]
...     transposed = sliced.lazy_transpose([1, 0])
...     data = transposed[:]
"""

from lazyslice.lazy_loading import DatasetView, lazy_transpose
from lazyslice.version import version as __version__

__all__ = ["DatasetView", "lazy_transpose", "__version__"]
