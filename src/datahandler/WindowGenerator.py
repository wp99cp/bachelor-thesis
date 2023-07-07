from rasterio.windows import from_bounds

WINDOWS = {
    '32TNS': {
        "left": 499980,
        "bottom": 5090220,
        "right": 609780,
        "top": 5200020
    },
    '32VMP': {
        "left": 399960,
        "bottom": 6790200,
        "right": 509760,
        "top": 6900000
    },
    '13TDE': {
        "left": 399960,
        "bottom": 4390200,
        "right": 509760,
        "top": 4500000
    },
    '07VEH': {
        "left": 499980,
        "bottom": 6690240,
        "right": 609780,
        "top": 6800040
    }
}


class WindowGenerator:

    def __init__(self, transform):
        self.transform = transform

    def get_window(self, tile_id: str, margin_in_pixels: int = 0):
        assert tile_id in WINDOWS.keys(), f"Tile id {tile_id} not found in WINDOWS dictionary."
        window = WINDOWS[tile_id]
        assert margin_in_pixels >= 0, "Margin must be greater than 0"
        assert margin_in_pixels % 60 == 0, "Margin must be a multiple of 60"

        return from_bounds(left=window["left"] + margin_in_pixels,
                           bottom=window["bottom"] + margin_in_pixels,
                           right=window["right"] - margin_in_pixels,
                           top=window["top"] - margin_in_pixels,
                           transform=self.transform)
