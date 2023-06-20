from rasterio.windows import from_bounds

WINDOWS = {
    '32TNS': {
        "left": 499980,
        "bottom": 5090220,
        "right": 609780,
        "top": 5200020
    }
}


class WindowGenerator:

    def __init__(self, transform):
        self.transform = transform

    def get_window(self, tile_id: str):
        assert tile_id in WINDOWS.keys(), f"Tile id {tile_id} not found in WINDOWS dictionary."
        window = WINDOWS[tile_id]
        return from_bounds(left=window["left"],
                           bottom=window["bottom"],
                           right=window["right"],
                           top=window["top"],
                           transform=self.transform)
