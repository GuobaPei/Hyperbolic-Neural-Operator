import numpy as np


def _read_tokens(path: str):
    """VTK legacy ASCII is whitespace-delimited; stream tokens to avoid huge memory spikes."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                yield tok


def read_unstructured_grid_ascii(path: str):
    """
    Minimal parser for VTK legacy ASCII UNSTRUCTURED_GRID.

    Returns dict with optional keys:
      - points: (N, 3) float32
      - cells: list[np.ndarray[int]] length M, each is (k,) point indices
      - point_data: dict[str, np.ndarray] (scalars/vectors) stored as float32
    """
    toks = _read_tokens(path)

    # header (we don't strictly validate; just advance until DATASET)
    points = None
    cells = None
    point_data = {}

    def _next():
        return next(toks)

    # scan tokens
    while True:
        try:
            t = _next()
        except StopIteration:
            break

        if t.upper() == "DATASET":
            ds = _next()
            if ds.upper() != "UNSTRUCTURED_GRID":
                raise ValueError(f"Only UNSTRUCTURED_GRID supported, got {ds} in {path}")

        elif t.upper() == "POINTS":
            n = int(_next())
            _dtype = _next()  # float/double; ignore
            arr = np.empty((n, 3), dtype=np.float32)
            for i in range(n * 3):
                arr.flat[i] = float(_next())
            points = arr

        elif t.upper() == "CELLS":
            m = int(_next())
            _total = int(_next())  # total ints count; ignore
            cells = []
            for _ in range(m):
                k = int(_next())
                idx = np.fromiter((int(_next()) for __ in range(k)), dtype=np.int64, count=k)
                cells.append(idx)

        elif t.upper() == "POINT_DATA":
            _n = int(_next())  # should match points count

        elif t.upper() == "VECTORS":
            name = _next()
            _dtype = _next()
            if points is None:
                raise ValueError(f"VECTORS before POINTS in {path}")
            n = points.shape[0]
            arr = np.empty((n, 3), dtype=np.float32)
            for i in range(n * 3):
                arr.flat[i] = float(_next())
            point_data[name] = arr

        elif t.upper() == "SCALARS":
            name = _next()
            _dtype = _next()
            ncomp = int(_next())
            # Expect: LOOKUP_TABLE default
            _lt = _next()
            _lt_name = _next()
            if points is None:
                raise ValueError(f"SCALARS before POINTS in {path}")
            n = points.shape[0]
            arr = np.empty((n, ncomp), dtype=np.float32) if ncomp > 1 else np.empty((n,), dtype=np.float32)
            for i in range(n * ncomp):
                arr.flat[i] = float(_next())
            point_data[name] = arr

        # ignore other sections (CELL_TYPES, CELL_DATA, etc.)

    if points is None:
        raise ValueError(f"Failed to parse POINTS from {path}")
    if cells is None:
        cells = []

    return {"points": points, "cells": cells, "point_data": point_data}


