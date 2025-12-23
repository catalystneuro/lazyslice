# lazyslice

[![PyPI version](https://img.shields.io/pypi/v/lazyslice.svg)](https://pypi.org/project/lazyslice/)
[![codecov](https://codecov.io/gh/catalystneuro/lazyslice/branch/main/graph/badge.svg)](https://codecov.io/gh/catalystneuro/lazyslice)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/pypi/pyversions/lazyslice.svg)](https://pypi.org/project/lazyslice/)

**Lazy transposing and slicing of h5py datasets and zarr arrays**

`lazyslice` allows you to compose multiple slice and transpose operations on h5py datasets and zarr arrays without reading data until explicitly requested. This is useful for working with large datasets where you want to defer I/O operations.

## Installation

```bash
pip install lazyslice
```

For zarr support:

```bash
pip install lazyslice[zarr]
```

## Quick Start

```python
from lazyslice import DatasetView
import h5py

# Open an HDF5 file and create a lazy view
with h5py.File('data.h5', 'r') as f:
    view = DatasetView(f['dataset'])
    
    # Chain lazy operations - no data is read yet
    result = (view
        .lazy_slice[1:40:2, :, 0:50:5]
        .lazy_transpose([2, 0, 1])
        .lazy_slice[8, 5:10])
    
    # Data is only read when you access it
    data = result[:]  # or result.read()
```

## Usage

### h5py Datasets

```python
from lazyslice import DatasetView
import h5py

with h5py.File('data.h5', 'r') as f:
    dsetview = DatasetView(f['dataset'])
    
    # Lazy slicing
    view1 = dsetview.lazy_slice[1:40:2, :, 0:50:5]
    
    # Lazy transpose
    view2 = view1.lazy_transpose([2, 0, 1])
    
    # Chain operations
    view3 = view2.lazy_slice[8, 5:10]
    
    # Read data
    data = view3[:]          # Using indexing
    data = view3.read()      # Using read() method
```

### zarr Arrays

```python
from lazyslice import DatasetView
import zarr

zarray = zarr.open('data.zarr')
zarrview = DatasetView(zarray)

# Same API as h5py
view = zarrview.lazy_slice[1:10:2, :, 5:10].lazy_transpose([0, 2, 1])
data = view.read()
```

### Lazy Iteration

```python
# Iterate over slices without loading all data at once
for slice_view in view.lazy_iter(axis=1):
    chunk = slice_view[:]
    process(chunk)
```

### Transpose Shortcut

```python
from lazyslice import lazy_transpose

# Quick transpose without creating a view first
transposed_view = lazy_transpose(dataset, axes=[2, 0, 1])
data = transposed_view[:]
```

## API Reference

### `DatasetView`

The main class for creating lazy views over datasets.

**Properties:**
- `shape` - Shape of the view after all lazy operations
- `dataset` - The underlying dataset
- `lazy_slice` - Enable lazy slicing mode for the next indexing operation
- `T` - Transpose (reverse axis order)

**Methods:**
- `read()` - Read and return the data as a numpy array
- `lazy_transpose(axis_order=None)` - Lazily transpose the array
- `lazy_iter(axis=0)` - Lazily iterate over slices along an axis
- `read_direct(dest, source_sel=None, dest_sel=None)` - Read directly into an existing array

### `lazy_transpose(dset, axes=None)`

Convenience function to create a transposed view of a dataset.

## Migration from `lazy_ops`

If you're upgrading from `lazy_ops`, the main changes are:

1. Package renamed: `lazy_ops` → `lazyslice`
2. Import path changed: `from lazy_ops import DatasetView` → `from lazyslice import DatasetView`
3. New `read()` method (recommended over `dsetread()`, which is still available for backward compatibility)

## Development

```bash
# Clone the repository
git clone https://github.com/catalystneuro/lazyslice.git
cd lazyslice

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=lazyslice
```

## License

BSD-3-Clause
