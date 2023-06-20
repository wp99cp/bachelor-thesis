import numpy as np

from src.datahandler.DataHandler import DataHandler
from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryData
from src.datahandler.satallite_reader.SatelliteReader import SatelliteReader

MEAN_PERCENTILES_30s = [1390, 1032, 759, 544, 633, 900, 974, 907, 981, 470, 16, 236, 151]
MEAN_PERCENTILES_70s = [9084, 9637, 9415, 10306, 10437, 10482, 10377, 10113, 10184, 5892, 504, 4044, 3295]
MEAN_MEANs = [3747, 3561, 3307, 3407, 3596, 4049, 4203, 4089, 4268, 2270, 138, 1831, 1332]

MAX_ELEV = 3913.0
MEAN_ELEV = 1862.78


class NormalizedDataHandler(DataHandler):
    """

    The Dataloader class is responsible for loading the data from the satellite reader and converting it into a format
    that can be used by the model / pipeline.

    The NormalizedDataHandler normalizes the loaded data to the range [-1, 1].

    @arg satelliteReader: the satellite reader to use
    @arg resolution: the resolution of the data

    """

    def __init__(self, satellite_reader: SatelliteReader, resolution: int = 10):
        super().__init__(satellite_reader, resolution)

    def _normalize_bands(self, bands: np.ndarray) -> np.ndarray:
        """

        Normalizes the bands to the range [-1, 1].
        using the dynamic world method (as described in my report)

        """

        bands = bands.astype(np.float32)
        np.log10(bands, where=bands > 0, out=bands)

        # shift back by adding mean channel values
        a = -1.7346  # 0.15 == 1 / (1 + Exp[-x]) // Solve
        b = +1.7346  # 0.85 == 1 / (1 + Exp[-x]) // Solve
        c = np.log10(np.array(MEAN_PERCENTILES_30s))
        d = np.log10(np.array(MEAN_PERCENTILES_70s))
        factor = (b - a) / (d - c)

        # (bands - c) * (b - a) / (d - c) + a
        np.subtract(bands, c.reshape((-1, 1, 1)), out=bands)
        np.multiply(bands, factor.reshape((-1, 1, 1)), out=bands)
        np.add(bands, a, out=bands)

        # 1 / (1 + exp(-bands))
        np.negative(bands, out=bands)
        np.exp(bands, out=bands)
        np.add(bands, 1, out=bands)
        np.reciprocal(bands, out=bands)

        offset = np.log10(MEAN_MEANs)
        offset = (offset - c) * (b - a) / (d - c) + a
        offset = 1 / (1 + np.exp(-offset))
        np.subtract(bands, offset.reshape((-1, 1, 1)), out=bands)

        assert np.min(bands) >= -1.0, "Min value is not >= -1.0"
        assert np.max(bands) <= 1.0, "Max value is not <= 1.0"

        return bands

    def _normalize_auxiliary_data(self, data: np.ndarray, auxiliary_data: AuxiliaryData) -> np.ndarray:
        assert auxiliary_data == AuxiliaryData.DEM, "NormalizedDataHandler only supports DEM normalization"

        # normalize DEM to range [0, 1]
        data = data.astype(np.float32)
        data = (data - MEAN_ELEV) / MAX_ELEV

        return data
