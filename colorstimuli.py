import uuid
import warnings
from typing import Any, Dict, Optional, Sequence
import numpy as np
import colour
if __package__:
    from .models import ColorStimulus, ScientificData, AppearanceData, DisplayRepresentation, SourceInfo, generate_u_name
    from .util import scam_to_xyz, xyz_to_scam
    from .colorname import get_ncs_hue_name, get_depth_description, classify_cielab_colors
    from .display import DisplayModel
else:
    from models import ColorStimulus, ScientificData, AppearanceData, DisplayRepresentation, SourceInfo, generate_u_name
    from util import scam_to_xyz, xyz_to_scam
    from colorname import get_ncs_hue_name, get_depth_description, classify_cielab_colors
    from display import DisplayModel
try:
    if __package__:
        from .config import global_config
    else:
        from config import global_config
except ModuleNotFoundError as exc:
    if exc.name != (f'{__package__}.config' if __package__ else 'config'):
        raise

    class _DefaultConfig:
        xyzw = [95.047, 100.0, 108.883]
        la = 64.0
        yb = 20.0
        surround = 'avg'
    global_config = _DefaultConfig()

def _check_viewing_condition(value):
    if value is not None:
        warnings.warn('viewing_condition is unused; pass xyzw, la, yb and surround instead.', DeprecationWarning, stacklevel=3)

def _triplet(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain three finite values.')
    return result

def _conditions(xyzw, la, yb, surround, config):
    white = _triplet(xyzw if xyzw is not None else config.xyzw, 'xyzw')
    la = float(la if la is not None else config.la)
    yb = float(yb if yb is not None else config.yb)
    surround = surround if surround is not None else config.surround
    if np.any(white <= 0):
        raise ValueError('xyzw must be positive.')
    if not np.isfinite([la, yb]).all() or la < 0 or yb <= 0:
        raise ValueError('la must be nonnegative and yb positive, both finite.')
    if not isinstance(surround, str) or surround.lower() not in ('avg', 'dim', 'dark'):
        raise ValueError('surround must be avg, dim or dark.')
    return (white, la, yb, surround.lower())

def jch_to_colorstimulus(jch: list, xyzw: Optional[list]=None, la: Optional[float]=None, yb: Optional[float]=None, surround: Optional[str]=None, viewing_condition: Optional[Any]=None, config=global_config) -> ColorStimulus:
    """Convert input via the shared XYZ appearance and naming pipeline."""
    jch = _triplet(jch, 'jch')
    if np.any(jch[:2] < 0):
        raise ValueError('J and C must be nonnegative.')
    xyzw, la, yb, surround = _conditions(xyzw, la, yb, surround, config)
    xyz = scam_to_xyz(jch, xyzw, yb, la, surround)
    stimulus = xyz_to_colorstimulus(xyz, xyzw, la, yb, surround, viewing_condition, config)
    stimulus.source.type = 'JCH_INPUT'
    return stimulus

def lab_to_colorstimulus(lab: list, xyzw: Optional[list]=None, la: Optional[float]=None, yb: Optional[float]=None, surround: Optional[str]=None, viewing_condition: Optional[Any]=None, config=global_config) -> ColorStimulus:
    """Convert input via the shared XYZ appearance and naming pipeline."""
    lab = _triplet(lab, 'lab')
    xyzw, la, yb, surround = _conditions(xyzw, la, yb, surround, config)
    xyz = colour.Lab_to_XYZ(lab, illuminant=colour.XYZ_to_xyY(xyzw))
    stimulus = xyz_to_colorstimulus(xyz, xyzw, la, yb, surround, viewing_condition, config)
    stimulus.source.type = 'LAB_INPUT'
    return stimulus

def srgb_to_colorstimulus(rgb: list, xyzw: Optional[list]=None, la: Optional[float]=None, yb: Optional[float]=None, surround: Optional[str]=None, viewing_condition: Optional[Any]=None, config=global_config) -> ColorStimulus:
    """Convert input via the shared XYZ appearance and naming pipeline."""
    rgb = _triplet(rgb, 'rgb')
    xyz = DisplayModel('srgb').rgb_to_xyz(rgb)[0]
    stimulus = xyz_to_colorstimulus(xyz, xyzw, la, yb, surround, viewing_condition, config)
    stimulus.source.type = 'SRGB_INPUT'
    rep = stimulus.display_representations['sRGB']
    rep.rgb_values = np.clip(rgb, 0, 1).tolist()
    rep.is_out_of_gamut = bool(np.any(rgb < 0) or np.any(rgb > 1))
    return stimulus

def xyz_to_colorstimulus(xyz: Sequence[float] | np.ndarray, xyzw: Optional[Sequence[float]]=None, la: Optional[float]=None, yb: Optional[float]=None, surround: Optional[str]=None, viewing_condition: Optional[Any]=None, config=global_config) -> ColorStimulus:
    """Convert one XYZ tristimulus value to a ColorStimulus."""
    _check_viewing_condition(viewing_condition)
    xyz_arr = _triplet(xyz, 'xyz')
    xyzw_arr, la_val, yb_val, surround_val = _conditions(xyzw, la, yb, surround, config)
    xyY_w = colour.XYZ_to_xyY(xyzw_arr)
    lab = colour.XYZ_to_Lab(xyz_arr, illuminant=xyY_w)
    lab_list = np.asarray(lab).tolist()
    jch_full, hd = xyz_to_scam(xyz=xyz_arr, xyz_w=xyzw_arr, y_b=yb_val, l_a=la_val, surround=surround_val, mode='full')
    H_val, D_val = (hd[0], hd[1])
    appearance = AppearanceData(lab_value=lab_list, JCh=jch_full.tolist(), NCS_name=get_ncs_hue_name(H_val), depth_description=get_depth_description(D_val), classification=classify_cielab_colors(lab_list))
    scientific_core = ScientificData(xyz_value=xyz_arr.tolist(), observer_angle=10, reference_white_xyz=xyzw_arr.tolist(), adapting_luminance_La=la_val, background_luminance_Yb=yb_val, surround_condition=surround_val)
    srgb_model = DisplayModel('srgb')
    rgb_values = srgb_model.xyz_to_rgb(np.array([xyz_arr]))[0]
    tolerance = 0.001
    is_out_of_gamut = bool(np.any(rgb_values < -tolerance) or np.any(rgb_values > 1 + tolerance))
    rgb_values_clipped = np.clip(rgb_values, 0, 1)
    display_rep = DisplayRepresentation(color_space_name='sRGB', rgb_values=rgb_values_clipped.tolist(), is_out_of_gamut=is_out_of_gamut)
    display_representations = {'sRGB': display_rep}
    color_id = str(uuid.uuid4())
    u_name = generate_u_name(appearance)  # ICONS
    metadata: Dict[str, Any] = {}
    source = SourceInfo(type='XYZ_INPUT', origin_identifier=None)
    return ColorStimulus(id=color_id, u_name=u_name, source=source, scientific_core=scientific_core, metadata=metadata, appearance=appearance, display_representations=display_representations)

def xyz2colorname(xyz: Sequence[float] | np.ndarray, xyzw: Optional[Sequence[float]]=None, la: Optional[float]=None, yb: Optional[float]=None, surround: Optional[str]=None, config=global_config, include_details: bool=False) -> Dict[str, Any] | str:
    """Return the colour name or detailed appearance data for XYZ."""
    stimulus = xyz_to_colorstimulus(xyz=xyz, xyzw=xyzw, la=la, yb=yb, surround=surround, config=config)
    appearance = stimulus.appearance
    if appearance is None:
        raise ValueError('Appearance data is missing from ColorStimulus.')
    if not include_details:
        return appearance.classification
    srgb_rep = stimulus.display_representations.get('sRGB')
    return {'color_name': appearance.classification, 'ncs_name': appearance.NCS_name, 'depth_description': appearance.depth_description, 'lab_value': appearance.lab_value, 'JCh': appearance.JCh, 'u_name': stimulus.u_name, 'is_out_of_gamut': srgb_rep.is_out_of_gamut if srgb_rep else None, 'rgb_preview': srgb_rep.rgb_values if srgb_rep else None}

def create_colorstimulus_from_xyz(xyz_values: Sequence[float] | np.ndarray, xyzw: Optional[Sequence[float]]=None, la: Optional[float]=None, yb: Optional[float]=None, surround: Optional[str]=None, viewing_condition: Optional[Any]=None, config=global_config) -> ColorStimulus:
    """Compatibility wrapper for xyz_to_colorstimulus."""
    return xyz_to_colorstimulus(xyz=xyz_values, xyzw=xyzw, la=la, yb=yb, surround=surround, viewing_condition=viewing_condition, config=config)
