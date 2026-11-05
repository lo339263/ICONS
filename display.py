import numpy as np
import colour


class DisplayModel:
    """
    Minimal display conversion helper supporting sRGB XYZ↔RGB transforms.
    XYZ values are assumed in the 0–100 range (Y=100 for reference white).
    """

    def __init__(self, color_space_name: str = "srgb") -> None:
        normalized = color_space_name.lower()
        if normalized != "srgb":
            raise ValueError("Only sRGB colour space is implemented in this DisplayModel.")
        self.name = normalized

    def xyz_to_rgb(self, xyz: np.ndarray) -> np.ndarray:
        xyz_arr = np.asarray(xyz, dtype=np.float64)
        single = xyz_arr.ndim == 1
        if single:
            xyz_arr = xyz_arr.reshape(1, 3)
        rgb = colour.models.XYZ_to_sRGB(xyz_arr / 100.0)
        return rgb.reshape(-1, 3) if not single else rgb.reshape(1, 3)

    def rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        rgb_arr = np.asarray(rgb, dtype=np.float64)
        single = rgb_arr.ndim == 1
        if single:
            rgb_arr = rgb_arr.reshape(1, 3)
        xyz = colour.models.sRGB_to_XYZ(rgb_arr) * 100.0
        return xyz.reshape(-1, 3) if not single else xyz.reshape(1, 3)
