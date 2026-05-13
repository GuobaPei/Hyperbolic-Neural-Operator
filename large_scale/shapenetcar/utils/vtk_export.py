import os
from typing import Optional

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_unstructured_grid(path: str):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def _as_scalar(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    if v.ndim == 2 and v.shape[1] == 1:
        v = v[:, 0]
    if v.ndim != 1:
        raise ValueError(f'Expected scalar array of shape (N,) or (N,1), got {v.shape}')
    return v.astype(np.float64, copy=False)


def _as_vector(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f'Expected vector array of shape (N,3), got {v.shape}')
    return v.astype(np.float64, copy=False)


def set_point_scalar(grid, name: str, values: np.ndarray, set_active: bool = False):
    vals = _as_scalar(values)
    n = grid.GetNumberOfPoints()
    if vals.shape[0] != n:
        raise ValueError(f'Point scalar length mismatch: grid has {n} points, got {vals.shape[0]}')
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(n)
    for i in range(n):
        arr.SetTuple1(i, float(vals[i]))
    grid.GetPointData().AddArray(arr)
    if set_active:
        grid.GetPointData().SetScalars(arr)
    return grid


def set_point_vector(grid, name: str, values: np.ndarray, set_active: bool = False):
    vals = _as_vector(values)
    n = grid.GetNumberOfPoints()
    if vals.shape[0] != n:
        raise ValueError(f'Point vector length mismatch: grid has {n} points, got {vals.shape[0]}')
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(3)
    arr.SetNumberOfTuples(n)
    for i in range(n):
        arr.SetTuple3(i, float(vals[i, 0]), float(vals[i, 1]), float(vals[i, 2]))
    grid.GetPointData().AddArray(arr)
    if set_active:
        grid.GetPointData().SetVectors(arr)
    return grid


def write_vtu(grid, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(path)
    writer.SetInputData(grid)
    writer.Write()


def map_vectors_by_point_coords(
    grid,
    coords: np.ndarray,
    vectors: np.ndarray,
    decimals: int = 6,
    default: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map vectors defined on 'coords' onto the points of 'grid' by (rounded) coordinate match.
    Returns an array of shape (grid_npoints, 3).
    """
    pts = vtk_to_numpy(grid.GetPoints().GetData())
    pts_key = [tuple(np.round(p, decimals=decimals)) for p in pts]
    coords_key = [tuple(np.round(p, decimals=decimals)) for p in np.asarray(coords)]
    vec = _as_vector(vectors)
    lut = {k: vec[i] for i, k in enumerate(coords_key)}

    if default is None:
        default = np.zeros(3, dtype=np.float64)
    default = np.asarray(default, dtype=np.float64)
    out = np.zeros((pts.shape[0], 3), dtype=np.float64)
    miss = 0
    for i, k in enumerate(pts_key):
        v = lut.get(k, None)
        if v is None:
            out[i] = default
            miss += 1
        else:
            out[i] = v
    # If miss is large, user might want to increase decimals tolerance
    return out


