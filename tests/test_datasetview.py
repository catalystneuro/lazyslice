"""Tests for DatasetView class.

This module contains systematic tests for the lazyslice DatasetView functionality
with both h5py and zarr backends. Tests use deterministic data and explicit
test cases rather than random values.
"""

from __future__ import annotations

import tempfile
from typing import Generator

import h5py
import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from lazyslice import DatasetView, lazy_transpose


# ============================================================================
# Fixtures with deterministic data
# ============================================================================


@pytest.fixture
def h5py_file() -> Generator[h5py.File, None, None]:
    """Create a temporary HDF5 file with deterministic test datasets."""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
        with h5py.File(f.name, "w") as h5f:
            # 3D dataset with shape (10, 8, 6) - values are indices for easy verification
            data_3d = np.arange(10 * 8 * 6).reshape(10, 8, 6).astype(float)
            h5f.create_dataset("data_3d", data=data_3d)
            
            # 2D dataset with shape (5, 4)
            data_2d = np.arange(20).reshape(5, 4).astype(float)
            h5f.create_dataset("data_2d", data=data_2d)
            
            # 1D dataset with shape (12,)
            data_1d = np.arange(12).astype(float)
            h5f.create_dataset("data_1d", data=data_1d)
            
            # 4D dataset with shape (4, 3, 5, 2)
            data_4d = np.arange(4 * 3 * 5 * 2).reshape(4, 3, 5, 2).astype(float)
            h5f.create_dataset("data_4d", data=data_4d)
            
            yield h5f


@pytest.fixture
def zarr_group() -> Generator[zarr.Group, None, None]:
    """Create a temporary zarr group with deterministic test arrays."""
    with tempfile.TemporaryDirectory(suffix=".zgroup") as temp_dir:
        group = zarr.group(store=temp_dir, overwrite=True)
        
        # Same datasets as h5py for consistent testing
        data_3d = np.arange(10 * 8 * 6).reshape(10, 8, 6).astype(float)
        data_2d = np.arange(20).reshape(5, 4).astype(float)
        data_1d = np.arange(12).astype(float)
        
        # Zarr 3.x uses create_array, Zarr 2.x uses create_dataset with data=
        if hasattr(group, "create_array"):
            # Zarr 3.x API
            group.create_array("data_3d", shape=data_3d.shape, dtype=data_3d.dtype)
            group["data_3d"][:] = data_3d
            group.create_array("data_2d", shape=data_2d.shape, dtype=data_2d.dtype)
            group["data_2d"][:] = data_2d
            group.create_array("data_1d", shape=data_1d.shape, dtype=data_1d.dtype)
            group["data_1d"][:] = data_1d
        else:
            # Fall back to Zarr 2.x API
            group.create_dataset("data_3d", data=data_3d)
            group.create_dataset("data_2d", data=data_2d)
            group.create_dataset("data_1d", data=data_1d)
        
        yield group


@pytest.fixture
def h5_3d(h5py_file: h5py.File) -> h5py.Dataset:
    """Get the 3D h5py dataset."""
    return h5py_file["data_3d"]


@pytest.fixture
def h5_2d(h5py_file: h5py.File) -> h5py.Dataset:
    """Get the 2D h5py dataset."""
    return h5py_file["data_2d"]


@pytest.fixture
def h5_1d(h5py_file: h5py.File) -> h5py.Dataset:
    """Get the 1D h5py dataset."""
    return h5py_file["data_1d"]


@pytest.fixture
def h5_4d(h5py_file: h5py.File) -> h5py.Dataset:
    """Get the 4D h5py dataset."""
    return h5py_file["data_4d"]


@pytest.fixture
def zarr_3d(zarr_group: zarr.Group) -> zarr.core.Array:
    """Get the 3D zarr array."""
    return zarr_group["data_3d"]


@pytest.fixture
def zarr_2d(zarr_group: zarr.Group) -> zarr.core.Array:
    """Get the 2D zarr array."""
    return zarr_group["data_2d"]


# ============================================================================
# Test classes
# ============================================================================


class TestDatasetViewCreation:
    """Tests for DatasetView creation and basic properties."""

    def test_create_from_h5py(self, h5_3d: h5py.Dataset):
        """Test creating DatasetView from h5py dataset."""
        view = DatasetView(h5_3d)
        assert view.shape == (10, 8, 6)
        assert view.dtype == h5_3d.dtype
        assert view.ndim == 3
        assert view.dataset is h5_3d

    def test_create_from_zarr(self, zarr_3d: zarr.core.Array):
        """Test creating DatasetView from zarr array."""
        view = DatasetView(zarr_3d)
        assert view.shape == (10, 8, 6)
        assert view.dtype == zarr_3d.dtype
        assert view.ndim == 3
        assert view.dataset is zarr_3d

    def test_invalid_input_numpy_array(self):
        """Test that numpy array raises TypeError."""
        with pytest.raises(TypeError, match="DatasetView requires"):
            DatasetView(np.array([1, 2, 3]))

    def test_invalid_input_list(self):
        """Test that list raises TypeError."""
        with pytest.raises(TypeError, match="DatasetView requires"):
            DatasetView([1, 2, 3])

    def test_invalid_input_dict(self):
        """Test that dict raises TypeError."""
        with pytest.raises(TypeError, match="DatasetView requires"):
            DatasetView({"a": 1})

    def test_len_3d(self, h5_3d: h5py.Dataset):
        """Test __len__ returns first dimension size for 3D array."""
        view = DatasetView(h5_3d)
        assert len(view) == 10

    def test_len_2d(self, h5_2d: h5py.Dataset):
        """Test __len__ returns first dimension size for 2D array."""
        view = DatasetView(h5_2d)
        assert len(view) == 5

    def test_len_1d(self, h5_1d: h5py.Dataset):
        """Test __len__ returns first dimension size for 1D array."""
        view = DatasetView(h5_1d)
        assert len(view) == 12

    def test_repr(self, h5_3d: h5py.Dataset):
        """Test string representation."""
        view = DatasetView(h5_3d)
        repr_str = repr(view)
        assert "DatasetView" in repr_str
        assert "(10, 8, 6)" in repr_str


class TestBasicOperations:
    """Tests for basic DatasetView operations."""

    def test_array_conversion(self, h5_3d: h5py.Dataset):
        """Test __array__ method returns correct data."""
        view = DatasetView(h5_3d)
        result = np.array(view)
        expected = h5_3d[()]
        assert_array_equal(result, expected)

    def test_read_method(self, h5_3d: h5py.Dataset):
        """Test read() method returns full data."""
        view = DatasetView(h5_3d)
        result = view.read()
        expected = h5_3d[()]
        assert_array_equal(result, expected)

    def test_dsetread_backward_compat(self, h5_3d: h5py.Dataset):
        """Test dsetread() backward compatibility alias."""
        view = DatasetView(h5_3d)
        assert_array_equal(view.read(), view.dsetread())

    def test_nonlazy_full_slice(self, h5_3d: h5py.Dataset):
        """Test direct indexing with [:] returns data."""
        view = DatasetView(h5_3d)
        assert_array_equal(h5_3d[:], view[:])

    def test_nonlazy_empty_tuple(self, h5_3d: h5py.Dataset):
        """Test direct indexing with () returns data."""
        view = DatasetView(h5_3d)
        assert_array_equal(h5_3d[()], view[()])


class TestLazySlicingBasic:
    """Tests for basic lazy slicing operations with explicit slice values."""

    def test_lazy_slice_full(self, h5_3d: h5py.Dataset):
        """Test lazy_slice[:] returns full data."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[:]
        assert isinstance(result, DatasetView)
        assert_array_equal(h5_3d[:], result.read())

    def test_lazy_slice_first_dim_only(self, h5_3d: h5py.Dataset):
        """Test slicing only the first dimension."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[2:8]
        assert result.shape == (6, 8, 6)
        assert_array_equal(h5_3d[2:8], result.read())

    def test_lazy_slice_all_dims(self, h5_3d: h5py.Dataset):
        """Test slicing all dimensions."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[1:7, 2:6, 0:4]
        assert result.shape == (6, 4, 4)
        assert_array_equal(h5_3d[1:7, 2:6, 0:4], result.read())

    def test_lazy_slice_single_element_range(self, h5_3d: h5py.Dataset):
        """Test slicing with single element range."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[5:6, 3:4, 2:3]
        assert result.shape == (1, 1, 1)
        assert_array_equal(h5_3d[5:6, 3:4, 2:3], result.read())


class TestLazySlicingWithStep:
    """Tests for lazy slicing with step values."""

    @pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
    def test_step_values_dim0(self, h5_3d: h5py.Dataset, step: int):
        """Test various step values on first dimension."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[::step]
        expected = h5_3d[::step]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    @pytest.mark.parametrize("step", [1, 2, 3])
    def test_step_values_all_dims(self, h5_3d: h5py.Dataset, step: int):
        """Test same step value on all dimensions."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[::step, ::step, ::step]
        expected = h5_3d[::step, ::step, ::step]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_different_steps_per_dim(self, h5_3d: h5py.Dataset):
        """Test different step values per dimension."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[::2, ::3, ::1]
        expected = h5_3d[::2, ::3, ::1]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_step_with_start_stop(self, h5_3d: h5py.Dataset):
        """Test step combined with start and stop."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[1:9:2, 0:7:3, 2:6:2]
        expected = h5_3d[1:9:2, 0:7:3, 2:6:2]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_large_step_exceeding_range(self, h5_3d: h5py.Dataset):
        """Test step larger than the slice range."""
        view = DatasetView(h5_3d)
        # Step 10 on dimension of size 10 should give 1 element
        result = view.lazy_slice[::10]
        expected = h5_3d[::10]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestLazySlicingIntegerIndex:
    """Tests for lazy slicing with integer indexing."""

    def test_single_int_index_first_dim(self, h5_3d: h5py.Dataset):
        """Test integer index on first dimension."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[5]
        expected = h5_3d[5]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_single_int_index_middle_dim(self, h5_3d: h5py.Dataset):
        """Test integer index on middle dimension."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[:, 3]
        expected = h5_3d[:, 3]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_single_int_index_last_dim(self, h5_3d: h5py.Dataset):
        """Test integer index on last dimension."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[:, :, 2]
        expected = h5_3d[:, :, 2]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_multiple_int_indices(self, h5_3d: h5py.Dataset):
        """Test multiple integer indices."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[2, 4]
        expected = h5_3d[2, 4]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_all_int_indices(self, h5_3d: h5py.Dataset):
        """Test integer index on all dimensions (scalar result)."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[3, 5, 2]
        expected = h5_3d[3, 5, 2]
        assert_array_equal(expected, result.read())

    def test_mixed_slice_and_int(self, h5_3d: h5py.Dataset):
        """Test mixing slices and integer indices."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[2:8, 3, 1:5]
        expected = h5_3d[2:8, 3, 1:5]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestLazySlicingNegativeIndices:
    """Tests for lazy slicing with negative indices."""

    def test_negative_start(self, h5_3d: h5py.Dataset):
        """Test negative start index."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[-5:]
        expected = h5_3d[-5:]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_negative_stop(self, h5_3d: h5py.Dataset):
        """Test negative stop index."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[:-3]
        expected = h5_3d[:-3]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_negative_start_and_stop(self, h5_3d: h5py.Dataset):
        """Test both negative start and stop."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[-7:-2]
        expected = h5_3d[-7:-2]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_negative_int_index(self, h5_3d: h5py.Dataset):
        """Test negative integer index."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[-1]
        expected = h5_3d[-1]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_negative_indices_all_dims(self, h5_3d: h5py.Dataset):
        """Test negative indices on all dimensions."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[-8:-2, -6:-1, -4:-1]
        expected = h5_3d[-8:-2, -6:-1, -4:-1]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestChainedSlicing:
    """Tests for chained lazy slice operations."""

    def test_chain_two_slices(self, h5_3d: h5py.Dataset):
        """Test chaining two slice operations."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[2:8].lazy_slice[1:4]
        expected = h5_3d[2:8][1:4]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_chain_three_slices(self, h5_3d: h5py.Dataset):
        """Test chaining three slice operations."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[1:9].lazy_slice[1:6].lazy_slice[1:3]
        expected = h5_3d[1:9][1:6][1:3]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_chain_with_step(self, h5_3d: h5py.Dataset):
        """Test chained slicing with steps."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[::2].lazy_slice[::2]
        expected = h5_3d[::2][::2]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_chain_different_dims(self, h5_3d: h5py.Dataset):
        """Test chained slicing on different dimensions."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[2:8, :, :].lazy_slice[:, 1:5, :].lazy_slice[:, :, 0:3]
        expected = h5_3d[2:8, 1:5, 0:3]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestLazyTranspose:
    """Tests for lazy transpose operations."""

    def test_transpose_default_3d(self, h5_3d: h5py.Dataset):
        """Test default transpose (reverse axes) for 3D array."""
        view = DatasetView(h5_3d)
        result = view.lazy_transpose()
        expected = h5_3d[()].T
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_transpose_default_2d(self, h5_2d: h5py.Dataset):
        """Test default transpose for 2D array."""
        view = DatasetView(h5_2d)
        result = view.lazy_transpose()
        expected = h5_2d[()].T
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_transpose_property(self, h5_3d: h5py.Dataset):
        """Test T property for transpose."""
        view = DatasetView(h5_3d)
        result = view.T
        expected = h5_3d[()].T
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    @pytest.mark.parametrize(
        "axes",
        [
            (0, 1, 2),  # identity
            (0, 2, 1),  # swap last two
            (1, 0, 2),  # swap first two
            (1, 2, 0),  # cycle
            (2, 0, 1),  # cycle other direction
            (2, 1, 0),  # full reverse
        ],
    )
    def test_transpose_all_permutations_3d(self, h5_3d: h5py.Dataset, axes: tuple):
        """Test all axis permutations for 3D array."""
        view = DatasetView(h5_3d)
        result = view.lazy_transpose(axes)
        expected = h5_3d[()].transpose(axes)
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_transpose_4d(self, h5_4d: h5py.Dataset):
        """Test transpose with 4D array."""
        view = DatasetView(h5_4d)
        axes = (3, 1, 0, 2)
        result = view.lazy_transpose(axes)
        expected = h5_4d[()].transpose(axes)
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestLazyTransposeFunction:
    """Tests for module-level lazy_transpose function."""

    def test_function_default(self, h5_3d: h5py.Dataset):
        """Test lazy_transpose function with default axes."""
        result = lazy_transpose(h5_3d)
        expected = h5_3d[()].T
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_function_explicit_axes(self, h5_3d: h5py.Dataset):
        """Test lazy_transpose function with explicit axes."""
        axes = (1, 2, 0)
        result = lazy_transpose(h5_3d, axes)
        expected = h5_3d[()].transpose(axes)
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestCombinedOperations:
    """Tests for combined lazy slice and transpose operations."""

    def test_slice_then_transpose(self, h5_3d: h5py.Dataset):
        """Test slicing followed by transpose."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[2:8, 1:6, 0:4].lazy_transpose([2, 0, 1])
        expected = h5_3d[2:8, 1:6, 0:4].transpose([2, 0, 1])
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_transpose_then_slice(self, h5_3d: h5py.Dataset):
        """Test transpose followed by slicing."""
        view = DatasetView(h5_3d)
        result = view.lazy_transpose([2, 0, 1]).lazy_slice[1:4, 2:7, 0:5]
        # Original shape (10, 8, 6) -> transposed shape (6, 10, 8)
        expected = h5_3d[()].transpose([2, 0, 1])[1:4, 2:7, 0:5]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_multiple_operations(self, h5_3d: h5py.Dataset):
        """Test multiple chained operations."""
        view = DatasetView(h5_3d)
        result = (
            view.lazy_slice[1:9, :, :]
            .lazy_transpose([1, 0, 2])
            .lazy_slice[2:6, :, 1:5]
        )
        expected = h5_3d[1:9, :, :].transpose([1, 0, 2])[2:6, :, 1:5]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestLazyIteration:
    """Tests for lazy iteration."""

    def test_lazy_iter_axis0(self, h5_3d: h5py.Dataset):
        """Test lazy iteration over axis 0."""
        view = DatasetView(h5_3d)
        for i, slice_view in enumerate(view.lazy_iter(axis=0)):
            expected = h5_3d[i]
            assert_array_equal(expected, slice_view.read())

    def test_lazy_iter_axis1(self, h5_3d: h5py.Dataset):
        """Test lazy iteration over axis 1."""
        view = DatasetView(h5_3d)
        for i, slice_view in enumerate(view.lazy_iter(axis=1)):
            expected = h5_3d[:, i, :]
            assert_array_equal(expected, slice_view.read())

    def test_lazy_iter_axis2(self, h5_3d: h5py.Dataset):
        """Test lazy iteration over axis 2."""
        view = DatasetView(h5_3d)
        for i, slice_view in enumerate(view.lazy_iter(axis=2)):
            expected = h5_3d[:, :, i]
            assert_array_equal(expected, slice_view.read())


class TestEllipsis:
    """Tests for ellipsis handling."""

    def test_ellipsis_at_start(self, h5_3d: h5py.Dataset):
        """Test ellipsis at start of index."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[..., 2]
        expected = h5_3d[..., 2]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_ellipsis_at_end(self, h5_3d: h5py.Dataset):
        """Test ellipsis at end of index."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[3, ...]
        expected = h5_3d[3, ...]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_ellipsis_in_middle(self, h5_4d: h5py.Dataset):
        """Test ellipsis in middle of index."""
        view = DatasetView(h5_4d)
        result = view.lazy_slice[1, ..., 0]
        expected = h5_4d[1, ..., 0]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestEmptySlices:
    """Tests for empty slice results."""

    def test_empty_slice_stop_before_start(self, h5_3d: h5py.Dataset):
        """Test slice where stop <= start gives empty result."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[5:5]
        assert result.shape[0] == 0

    def test_empty_slice_zero_range(self, h5_3d: h5py.Dataset):
        """Test slice with zero range."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[0:0]
        assert result.shape[0] == 0

    def test_empty_slice_negative_range(self, h5_3d: h5py.Dataset):
        """Test slice where normalized range is empty."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[8:3]  # stop < start after normalization
        assert result.shape[0] == 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_negative_step_raises(self, h5_3d: h5py.Dataset):
        """Test that negative step raises ValueError."""
        view = DatasetView(h5_3d)
        with pytest.raises(ValueError, match="step parameter must be positive"):
            view.lazy_slice[::-1]

    def test_zero_step_raises(self, h5_3d: h5py.Dataset):
        """Test that zero step raises ValueError."""
        view = DatasetView(h5_3d)
        with pytest.raises(ValueError, match="(step parameter must be positive|slice step cannot be zero)"):
            view.lazy_slice[::0]

    def test_index_too_large_raises(self, h5_3d: h5py.Dataset):
        """Test that out-of-range positive index raises IndexError."""
        view = DatasetView(h5_3d)
        with pytest.raises(IndexError, match="out of range"):
            view.lazy_slice[100]

    def test_index_too_negative_raises(self, h5_3d: h5py.Dataset):
        """Test that out-of-range negative index raises IndexError."""
        view = DatasetView(h5_3d)
        with pytest.raises(IndexError, match="out of range"):
            view.lazy_slice[-100]

    def test_multiple_ellipsis_raises(self, h5_3d: h5py.Dataset):
        """Test that multiple ellipsis raises IndexError."""
        view = DatasetView(h5_3d)
        with pytest.raises(IndexError, match="single Ellipsis"):
            view.lazy_slice[..., 1, ...]


class TestBooleanIndexing:
    """Tests for boolean array indexing."""

    def test_boolean_index_first_dim(self, h5_3d: h5py.Dataset):
        """Test boolean indexing on first dimension."""
        view = DatasetView(h5_3d)
        bool_idx = np.array([True, False, True, False, True, False, True, False, True, False])
        result = view.lazy_slice[bool_idx, :, :]
        expected = h5_3d[bool_idx, :, :]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_boolean_index_all_true(self, h5_2d: h5py.Dataset):
        """Test boolean index with all True."""
        view = DatasetView(h5_2d)
        bool_idx = np.array([True, True, True, True, True])
        result = view.lazy_slice[bool_idx, :]
        expected = h5_2d[bool_idx, :]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_boolean_index_all_false(self, h5_2d: h5py.Dataset):
        """Test boolean index with all False."""
        view = DatasetView(h5_2d)
        bool_idx = np.array([False, False, False, False, False])
        result = view.lazy_slice[bool_idx, :]
        expected = h5_2d[bool_idx, :]
        assert result.shape == expected.shape


class TestArrayIndexing:
    """Tests for integer array indexing."""

    def test_array_index_sequential(self, h5_3d: h5py.Dataset):
        """Test array indexing with sequential indices."""
        view = DatasetView(h5_3d)
        indices = [0, 1, 2, 3]
        # h5py requires a second index for array indexing
        result = view.lazy_slice[indices, :]
        expected = h5_3d[indices, :]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_array_index_non_contiguous(self, h5_3d: h5py.Dataset):
        """Test array indexing with non-contiguous (but sorted) indices."""
        view = DatasetView(h5_3d)
        # h5py requires sorted indices in increasing order
        indices = [0, 2, 5, 7, 9]
        result = view.lazy_slice[indices, :]
        expected = h5_3d[indices, :]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_array_index_every_other(self, h5_3d: h5py.Dataset):
        """Test array indexing selecting every other element."""
        view = DatasetView(h5_3d)
        # Even indices only
        indices = [0, 2, 4, 6, 8]
        result = view.lazy_slice[indices, :]
        expected = h5_3d[indices, :]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_array_index_single_element(self, h5_3d: h5py.Dataset):
        """Test array indexing with single element."""
        view = DatasetView(h5_3d)
        indices = [4]
        result = view.lazy_slice[indices, :]
        expected = h5_3d[indices, :]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestZarrBackend:
    """Tests specific to zarr backend."""

    def test_zarr_read(self, zarr_3d: zarr.core.Array):
        """Test basic read with zarr."""
        view = DatasetView(zarr_3d)
        result = view.read()
        expected = zarr_3d[()]
        assert_array_equal(expected, result)

    def test_zarr_lazy_slice(self, zarr_3d: zarr.core.Array):
        """Test lazy slicing with zarr."""
        view = DatasetView(zarr_3d)
        result = view.lazy_slice[2:8, 1:6, 0:4]
        expected = zarr_3d[2:8, 1:6, 0:4]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_zarr_lazy_slice_with_step(self, zarr_3d: zarr.core.Array):
        """Test lazy slicing with step on zarr."""
        view = DatasetView(zarr_3d)
        result = view.lazy_slice[::2, ::3, ::2]
        expected = zarr_3d[::2, ::3, ::2]
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_zarr_lazy_transpose(self, zarr_3d: zarr.core.Array):
        """Test lazy transpose with zarr."""
        view = DatasetView(zarr_3d)
        result = view.lazy_transpose([2, 0, 1])
        expected = zarr_3d[()].transpose([2, 0, 1])
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())

    def test_zarr_combined_operations(self, zarr_3d: zarr.core.Array):
        """Test combined slice and transpose with zarr."""
        view = DatasetView(zarr_3d)
        result = view.lazy_slice[1:8, :, 2:5].lazy_transpose([1, 2, 0])
        expected = zarr_3d[1:8, :, 2:5].transpose([1, 2, 0])
        assert result.shape == expected.shape
        assert_array_equal(expected, result.read())


class TestShapeCalculation:
    """Tests for correct shape calculation with various slice configurations."""

    @pytest.mark.parametrize(
        "slice_,expected_len",
        [
            (slice(0, 10, 1), 10),  # Full range, step 1
            (slice(0, 10, 2), 5),   # Full range, step 2
            (slice(0, 10, 3), 4),   # Full range, step 3 (10/3 = 3.33 -> 4)
            (slice(0, 10, 5), 2),   # Full range, step 5
            (slice(0, 10, 10), 1),  # Full range, step = size
            (slice(0, 10, 11), 1),  # Full range, step > size
            (slice(0, 9, 2), 5),    # 9 elements, step 2 (0,2,4,6,8)
            (slice(0, 9, 3), 3),    # 9 elements, step 3 (0,3,6)
            (slice(1, 9, 2), 4),    # Start at 1 (1,3,5,7)
            (slice(2, 8, 3), 2),    # 2,5
            (slice(5, 5, 1), 0),    # Empty range
            (slice(8, 3, 1), 0),    # Reversed range (empty)
        ],
    )
    def test_shape_calculation(self, h5_3d: h5py.Dataset, slice_: slice, expected_len: int):
        """Test that shape is calculated correctly for various slices."""
        view = DatasetView(h5_3d)
        result = view.lazy_slice[slice_]
        assert result.shape[0] == expected_len


class TestCallMethod:
    """Tests for __call__ method."""

    def test_call_with_slice(self, h5_3d: h5py.Dataset):
        """Test calling view with slice object."""
        view = DatasetView(h5_3d)
        lazy_view = view.lazy_slice
        result = lazy_view(slice(2, 8))
        expected = h5_3d[2:8]
        assert_array_equal(expected, result)

    def test_call_with_tuple(self, h5_3d: h5py.Dataset):
        """Test calling view with tuple of slices."""
        view = DatasetView(h5_3d)
        lazy_view = view.lazy_slice
        result = lazy_view((slice(1, 7), slice(2, 6)))
        expected = h5_3d[1:7, 2:6]
        assert_array_equal(expected, result)


class TestReadDirect:
    """Tests for read_direct method (h5py only)."""

    def test_read_direct_basic(self, h5_3d: h5py.Dataset):
        """Test basic read_direct without selections."""
        view = DatasetView(h5_3d)
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_3d[()]
        assert_array_equal(expected, dest)

    def test_read_direct_with_lazy_slice(self, h5_3d: h5py.Dataset):
        """Test read_direct on a sliced view."""
        view = DatasetView(h5_3d).lazy_slice[2:8, 1:6, 0:4]
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_3d[2:8, 1:6, 0:4]
        assert_array_equal(expected, dest)

    def test_read_direct_with_source_sel(self, h5_3d: h5py.Dataset):
        """Test read_direct with source_sel parameter."""
        view = DatasetView(h5_3d)
        # Select a subset using source_sel
        source_sel = (slice(2, 6), slice(1, 5), slice(0, 3))
        dest = np.empty((4, 4, 3), dtype=view.dtype)
        view.read_direct(dest, source_sel=source_sel)
        expected = h5_3d[2:6, 1:5, 0:3]
        assert_array_equal(expected, dest)

    def test_read_direct_with_transpose(self, h5_3d: h5py.Dataset):
        """Test read_direct on a transposed view."""
        view = DatasetView(h5_3d).lazy_transpose([2, 0, 1])
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_3d[()].transpose([2, 0, 1])
        assert_array_equal(expected, dest)

    def test_read_direct_with_slice_and_transpose(self, h5_3d: h5py.Dataset):
        """Test read_direct on a sliced and transposed view."""
        view = DatasetView(h5_3d).lazy_slice[1:7, 2:6, 0:4].lazy_transpose([2, 0, 1])
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_3d[1:7, 2:6, 0:4].transpose([2, 0, 1])
        assert_array_equal(expected, dest)

    def test_read_direct_with_step(self, h5_3d: h5py.Dataset):
        """Test read_direct on a view with step slicing."""
        view = DatasetView(h5_3d).lazy_slice[::2, ::2, ::2]
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_3d[::2, ::2, ::2]
        assert_array_equal(expected, dest)

    def test_read_direct_2d(self, h5_2d: h5py.Dataset):
        """Test read_direct with 2D dataset."""
        view = DatasetView(h5_2d)
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_2d[()]
        assert_array_equal(expected, dest)

    def test_read_direct_1d(self, h5_1d: h5py.Dataset):
        """Test read_direct with 1D dataset."""
        view = DatasetView(h5_1d)
        dest = np.empty(view.shape, dtype=view.dtype)
        view.read_direct(dest)
        expected = h5_1d[()]
        assert_array_equal(expected, dest)

    def test_read_direct_zarr_raises(self, zarr_3d):
        """Test that read_direct raises NotImplementedError for zarr."""
        view = DatasetView(zarr_3d)
        dest = np.empty(view.shape, dtype=view.dtype)
        with pytest.raises(NotImplementedError, match="read_direct is not supported"):
            view.read_direct(dest)

    def test_read_direct_with_dest_sel(self, h5_3d: h5py.Dataset):
        """Test read_direct with dest_sel parameter."""
        view = DatasetView(h5_3d).lazy_slice[0:4, 0:4, 0:3]
        # Create destination array matching view shape
        dest = np.empty((4, 4, 3), dtype=view.dtype)
        dest_sel = (slice(0, 4), slice(0, 4), slice(0, 3))
        view.read_direct(dest, dest_sel=dest_sel)
        expected = h5_3d[0:4, 0:4, 0:3]
        assert_array_equal(expected, dest)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
