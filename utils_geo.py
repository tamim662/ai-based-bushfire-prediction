import numpy as np
import pandas as pd

def make_grid_cell_ids(lat, lon, grid_km: float = 5.0):
    """
    Make coarse grid cell IDs. Uses pandas string concat (robust).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # approx degrees per km (rough, fine for clustering)
    deg = grid_km / 111.0

    gy = np.floor((lat - lat.min()) / deg).astype(int)
    gx = np.floor((lon - lon.min()) / deg).astype(int)

    cell_id = (pd.Series(gy.astype(str)) + "_" + pd.Series(gx.astype(str))).values
    return cell_id
