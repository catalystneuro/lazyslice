"""Lazy transposing and slicing operations for h5py datasets and zarr arrays.

This module provides a `DatasetView` class that allows for lazy transposing and
slicing operations on h5py datasets and zarr arrays. Operations are composed
and only executed when data is actually read.

Example
-------
>>> from lazyslice import DatasetView
>>> import h5py
>>> 
>>> # h5py usage
>>> with h5py.File('data.h5', 'r') as f:
...     dsetview = DatasetView(f['dataset'])
...     view = dsetview.lazy_slice[1:40:2, :, 0:50:5].lazy_transpose([2, 0, 1])
...     data = view[:]  # Data is read here
>>>
>>> # zarr usage
>>> import zarr
>>> zarrview = DatasetView(zarr.open('data.zarr'))
>>> view = zarrview.lazy_slice[1:10:2, :, 5:10].lazy_transpose([0, 2, 1])
>>> data = view.read()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import zarr

# Check for zarr availability
try:
    import zarr as zarr_module

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False
    zarr_module = None


def _is_zarr_array(obj: Any) -> bool:
    """Check if object is a zarr array without importing zarr."""
    if HAVE_ZARR:
        return isinstance(obj, zarr_module.core.Array)
    return "zarr" in str(type(obj)).lower()


class DatasetView:
    """A lazy view over an h5py dataset or zarr array.
    
    DatasetView allows composing multiple slice and transpose operations
    without reading data. Data is only read when explicitly requested
    via indexing (`view[:]`) or the `read()` method.
    
    Note: Negative strides are not currently supported and will raise an error.
    
    Parameters
    ----------
    dataset : h5py.Dataset or zarr.core.Array
        The underlying dataset to create a view over.
    
    Examples
    --------
    >>> import h5py
    >>> from lazyslice import DatasetView
    >>> 
    >>> with h5py.File('data.h5', 'r') as f:
    ...     view = DatasetView(f['dataset'])
    ...     sliced = view.lazy_slice[10:20, :, 5]
    ...     transposed = sliced.lazy_transpose([1, 0])
    ...     data = transposed[:]
    """

    def __init__(
        self,
        dataset: Any,
        slice_index: Optional[Tuple[tuple, tuple]] = None,
        axis_order: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize the DatasetView.
        
        Parameters
        ----------
        dataset : h5py.Dataset or zarr.core.Array
            The underlying dataset.
        slice_index : tuple, optional
            The aggregate slice and int indices.
        axis_order : tuple, optional
            The aggregate axis order after transpositions.
        """
        # Validate dataset type
        if not isinstance(dataset, h5py.Dataset) and not _is_zarr_array(dataset):
            if "zarr" in str(type(dataset)).lower():
                raise TypeError(
                    "To use DatasetView with a zarr array, install zarr:\n"
                    "  pip install zarr"
                )
            raise TypeError(
                "DatasetView requires either an h5py.Dataset or zarr.core.Array. "
                f"Got: {type(dataset)}"
            )

        slice_index = slice_index or (np.index_exp[:], ())
        self._axis_order = axis_order or tuple(range(len(dataset.shape)))
        self._lazy_slice_call = False
        self._dataset = dataset
        self._shape, self._key, self._int_index, self._axis_order = self._slice_shape(
            slice_index
        )

    @property
    def lazy_slice(self) -> "DatasetView":
        """Enable lazy slicing mode.
        
        Returns
        -------
        DatasetView
            Self, with lazy slicing enabled for the next indexing operation.
            
        Examples
        --------
        >>> view = DatasetView(dataset)
        >>> sliced = view.lazy_slice[10:20, :]  # Returns new DatasetView
        """
        self._lazy_slice_call = True
        return self

    @property
    def dataset(self) -> Any:
        """The underlying dataset."""
        return self._dataset

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the view after all lazy operations."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying dataset."""
        return self._dataset.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self._shape[0]

    @property
    def key(self) -> tuple:
        """The accumulated slice key to be passed to the dataset."""
        return self._key

    @property
    def axis_order(self) -> Tuple[int, ...]:
        """The axis permutation order after lazy transpositions."""
        return self._axis_order

    def _normalize_key(self, key: Union[slice, int, np.integer, NDArray]) -> tuple:
        """Normalize a single slice/int/array to a tuple."""
        if isinstance(key, (slice, int, np.integer, np.ndarray)):
            return (key,)
        return (*key,)

    def _slice_shape(
        self, slice_: Tuple[tuple, tuple]
    ) -> Tuple[Tuple[int, ...], tuple, tuple, Tuple[int, ...]]:
        """Calculate the shape resulting from a slice operation.
        
        Returns
        -------
        tuple
            (shape, slice_key, int_index, axis_order)
        """
        int_ind = slice_[1]
        slice_ = self._normalize_key(slice_[0])

        # Convert slices to regular slices with integer bounds
        slice_regindices = []
        for i in range(len(slice_)):
            if isinstance(slice_[i], slice):
                slice_regindices.append(
                    slice(*slice_[i].indices(self.dataset.shape[self.axis_order[i]]))
                )
            else:
                slice_regindices.append(slice_[i])

        slice_shape_list = []
        int_index_list = []
        axis_order_list = []

        for i, idx in enumerate(slice_):
            if isinstance(idx, slice):
                start, stop, step = (
                    slice_regindices[i].start,
                    slice_regindices[i].stop,
                    slice_regindices[i].step,
                )
                
                if step < 1:
                    raise ValueError("Slice step parameter must be positive")
                
                # Clamp stop to start if reversed (empty slice)
                stop = max(start, stop) if stop >= start else start
                slice_regindices[i] = slice(start, stop, step)
                slice_shape_list.append((stop - start + step - 1) // step)
                axis_order_list.append(self.axis_order[i])
            elif isinstance(idx, (int, np.integer)):
                int_index_list.append((i, idx, self.axis_order[i]))
            else:
                # idx is an iterator of integers
                slice_shape_list.append(len(idx))
                axis_order_list.append(self.axis_order[i])

        slice_regindices = tuple(
            el for el in slice_regindices if not isinstance(el, (int, np.integer))
        )
        axis_order_list.extend(
            self.axis_order[len(axis_order_list) + len(int_index_list) :]
        )
        int_index = tuple(int_index_list) + int_ind
        slice_shape = tuple(slice_shape_list) + self.dataset.shape[
            len(slice_shape_list) + len(int_index) :
        ]

        return slice_shape, slice_regindices, int_index, tuple(axis_order_list)

    def __getitem__(self, new_slice) -> Union["DatasetView", NDArray]:
        """Support Python's colon slicing syntax."""
        key_reinit = self._slice_composition(new_slice)
        if self._lazy_slice_call:
            self._lazy_slice_call = False
            return DatasetView(
                self.dataset, (key_reinit, self._int_index), self.axis_order
            )
        return DatasetView(
            self.dataset, (key_reinit, self._int_index), self.axis_order
        ).read()

    def lazy_iter(self, axis: int = 0):
        """Lazily iterate over slices along an axis.
        
        Parameters
        ----------
        axis : int, default 0
            The axis to iterate over.
            
        Yields
        ------
        DatasetView
            A view for each slice along the axis.
        """
        for i in range(self._shape[axis]):
            yield self.lazy_slice[(*np.index_exp[:] * axis, i)]

    def __call__(self, new_slice) -> Union["DatasetView", NDArray]:
        """Allow lazy_slice function calls with slice objects as input."""
        return self.__getitem__(new_slice)

    def read(self) -> NDArray:
        """Read and return the data from the dataset.
        
        This method applies all accumulated lazy operations and returns
        the actual numpy array data.
        
        Returns
        -------
        numpy.ndarray
            The data after applying all slice and transpose operations.
        """
        lazy_axis_order = self.axis_order
        lazy_key = self.key

        for ind in self._int_index:
            lazy_axis_order = (
                lazy_axis_order[: ind[0]] + (ind[2],) + lazy_axis_order[ind[0] :]
            )
            lazy_key = lazy_key[: ind[0]] + (ind[1],) + lazy_key[ind[0] :]

        reversed_axis_order = sorted(
            range(len(lazy_axis_order)), key=lambda i: lazy_axis_order[i]
        )
        reversed_slice_key = tuple(
            lazy_key[i] for i in reversed_axis_order if i < len(lazy_key)
        )

        # Reduce axis_order values to account for dimensions dropped by int indexing
        reversed_axis_order_read = sorted(
            range(len(self.axis_order)), key=lambda i: self.axis_order[i]
        )
        axis_order_read = sorted(
            range(len(self.axis_order)), key=lambda i: reversed_axis_order_read[i]
        )

        return self.dataset[reversed_slice_key].transpose(axis_order_read)

    # Backward compatibility alias
    def dsetread(self) -> NDArray:
        """Read and return the data. Deprecated: use `read()` instead."""
        return self.read()

    def _slice_composition(self, new_slice) -> tuple:
        """Compose a new slice with the current accumulated slice."""
        new_slice = self._normalize_key(new_slice)
        new_slice = self._expand_ellipsis(new_slice)
        slice_result_list = []

        for i in range(len(new_slice)):
            if isinstance(new_slice[i], slice):
                slice_result_list.append(self._compose_slice_with_slice(i, new_slice[i]))
            elif isinstance(new_slice[i], (int, np.integer)):
                slice_result_list.append(self._compose_slice_with_int(i, new_slice[i]))
            else:
                slice_result_list.append(self._compose_slice_with_array(i, new_slice[i]))

        slice_result = tuple(slice_result_list)
        slice_result += self.key[len(new_slice) :]
        return slice_result

    def _compose_slice_with_slice(self, dim: int, new_slice: slice):
        """Compose a slice with the current key at dimension `dim`."""
        if dim < len(self.key):
            newkey_start, newkey_stop, newkey_step = new_slice.indices(self._shape[dim])
            if newkey_step < 1:
                raise ValueError("Slice step parameter must be positive")
            if newkey_stop < newkey_start:
                newkey_start = newkey_stop

            if isinstance(self.key[dim], slice):
                return slice(
                    min(
                        self.key[dim].start + self.key[dim].step * newkey_start,
                        self.key[dim].stop,
                    ),
                    min(
                        self.key[dim].start + self.key[dim].step * newkey_stop,
                        self.key[dim].stop,
                    ),
                    newkey_step * self.key[dim].step,
                )
            else:
                # self.key[dim] is an iterator of integers
                return self.key[dim][new_slice]
        else:
            return slice(*new_slice.indices(self.dataset.shape[self.axis_order[dim]]))

    def _compose_slice_with_int(self, dim: int, new_int: int):
        """Compose an integer index with the current key at dimension `dim`."""
        if dim < len(self.key):
            if new_int >= self._shape[dim] or new_int <= ~self._shape[dim]:
                raise IndexError(
                    f"Index {new_int} out of range, dim {dim} of size {self._shape[dim]}"
                )
            if isinstance(self.key[dim], slice):
                return self.key[dim].start + self.key[dim].step * (
                    new_int % self._shape[dim]
                )
            else:
                return self.key[dim][new_int]
        else:
            return new_int

    def _compose_slice_with_array(self, dim: int, new_array):
        """Compose an array index with the current key at dimension `dim`."""
        try:
            if not all(isinstance(el, (int, np.integer)) for el in new_array):
                if new_array.dtype.kind != "b":
                    raise ValueError("Indices must be either integers or booleans")
                # Boolean indexing
                if len(new_array) != self.shape[dim]:
                    raise IndexError(
                        f"Length of boolean index {len(new_array)} must be equal to "
                        f"size {self.shape[dim]} in dim {dim}"
                    )
                new_array = new_array.nonzero()[0]

            if dim < len(self.key):
                if any(
                    el >= self._shape[dim] or el <= ~self._shape[dim] for el in new_array
                ):
                    raise IndexError(
                        f"Index {new_array} out of range, dim {dim} of size {self._shape[dim]}"
                    )
                if isinstance(self.key[dim], slice):
                    return tuple(
                        self.key[dim].start + self.key[dim].step * (ind % self._shape[dim])
                        for ind in new_array
                    )
                else:
                    return tuple(self.key[dim][ind] for ind in new_array)
            else:
                return new_array
        except (TypeError, AttributeError) as e:
            raise IndexError(
                "Indices must be integers, iterators of integers, slice objects, "
                "or numpy boolean arrays"
            ) from e

    @property
    def T(self) -> "DatasetView":
        """Transpose the view (reverse axis order)."""
        return self.lazy_transpose()

    def lazy_transpose(
        self, axis_order: Optional[Sequence[int]] = None
    ) -> "DatasetView":
        """Lazily transpose the array.
        
        Parameters
        ----------
        axis_order : sequence of int, optional
            The desired axis permutation. If None, reverses all axes.
            
        Returns
        -------
        DatasetView
            A new view with the transposed axis order.
        """
        if axis_order is None:
            axis_order = tuple(reversed(range(len(self.axis_order))))

        axis_order_reinit = tuple(
            self.axis_order[i] if i < len(self.axis_order) else i for i in axis_order
        )
        key_reinit = tuple(
            self.key[i] if i < len(self.key) else np.s_[:] for i in axis_order
        )
        key_reinit += tuple(
            self.key[i] for i in self.axis_order if i not in axis_order_reinit
        )
        axis_order_reinit += tuple(
            i for i in self.axis_order if i not in axis_order_reinit
        )

        return DatasetView(
            self.dataset, (key_reinit, self._int_index), axis_order_reinit
        )

    def __array__(self, dtype=None, copy=None) -> NDArray:
        """Convert to numpy array."""
        result = np.atleast_1d(self.read())
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result

    def _expand_ellipsis(self, new_slice: tuple) -> tuple:
        """Expand Ellipsis in slice to explicit slice objects."""
        ellipsis_count = sum(
            s is Ellipsis for s in new_slice if not isinstance(s, np.ndarray)
        )
        if ellipsis_count == 1:
            ellipsis_index = new_slice.index(Ellipsis)
            if ellipsis_index == len(new_slice) - 1:
                new_slice = new_slice[:-1]
            else:
                num_ellipsis_dims = len(self.shape) - (len(new_slice) - 1)
                new_slice = (
                    new_slice[:ellipsis_index]
                    + np.index_exp[:] * num_ellipsis_dims
                    + new_slice[ellipsis_index + 1 :]
                )
        elif ellipsis_count > 1:
            raise IndexError("Only a single Ellipsis is allowed")
        return new_slice

    def read_direct(
        self,
        dest: NDArray,
        source_sel: Optional[tuple] = None,
        dest_sel: Optional[tuple] = None,
    ) -> None:
        """Read data directly into an existing array (h5py only).
        
        Parameters
        ----------
        dest : numpy.ndarray
            Destination array (must be C-contiguous).
        source_sel : tuple, optional
            Selection on the source data.
        dest_sel : tuple, optional
            Selection on the destination array.
            
        Raises
        ------
        NotImplementedError
            If the underlying dataset doesn't support read_direct (e.g., zarr).
        """
        if not hasattr(self.dataset, "read_direct"):
            raise NotImplementedError(
                f"read_direct is not supported for {type(self.dataset).__name__}. "
                "Use read() instead."
            )

        if source_sel is None:
            new_key = self.key
            new_int_index = self._int_index
            new_axis_order = self.axis_order
        else:
            key_reinit = self._slice_composition(source_sel)
            _, new_key, new_int_index, new_axis_order = self._slice_shape(
                (key_reinit, ())
            )

        axis_order_slices = new_axis_order
        for ind in new_int_index:
            new_axis_order = (
                new_axis_order[: ind[0]] + (ind[2],) + new_axis_order[ind[0] :]
            )
            new_key = new_key[: ind[0]] + (ind[1],) + new_key[ind[0] :]

        reversed_axis_order = sorted(
            range(len(new_axis_order)), key=lambda i: new_axis_order[i]
        )
        reversed_slice_key = tuple(
            new_key[i] for i in reversed_axis_order if i < len(new_key)
        )

        reversed_axis_order_read = sorted(
            range(len(axis_order_slices)), key=lambda i: axis_order_slices[i]
        )
        axis_order_read = sorted(
            range(len(axis_order_slices)), key=lambda i: reversed_axis_order_read[i]
        )

        reversed_dest_shape = tuple(
            dest.shape[i] for i in reversed_axis_order_read if i < len(dest.shape)
        )
        reversed_dest = np.empty(shape=reversed_dest_shape, dtype=dest.dtype)

        if dest_sel is None:
            reversed_dest_sel = None
        else:
            reversed_dest_sel = tuple(
                dest_sel[i] for i in reversed_axis_order if i < len(dest_sel)
            )

        self.dataset.read_direct(
            reversed_dest, source_sel=reversed_slice_key, dest_sel=reversed_dest_sel
        )
        np.copyto(dest, reversed_dest.transpose(axis_order_read))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DatasetView(shape={self.shape}, dtype={self.dtype})"


def lazy_transpose(
    dset: Any, axes: Optional[Sequence[int]] = None
) -> DatasetView:
    """Lazily transpose a dataset.
    
    Parameters
    ----------
    dset : h5py.Dataset or zarr.core.Array
        The dataset to transpose.
    axes : sequence of int, optional
        The desired axis permutation. If None, reverses all axes.
        
    Returns
    -------
    DatasetView
        A lazy view with the transposed axis order.
    """
    if axes is None:
        axes = tuple(reversed(range(len(dset.shape))))
    return DatasetView(dset).lazy_transpose(axis_order=axes)
