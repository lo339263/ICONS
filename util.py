import numpy as np
XYZ_TO_LMS_MATRIX = np.array([[0.400227052325915, 0.707583738918745, -0.080809279274243], [-0.226301727628935, 1.16533416092519, 0.045699378859811], [0.0, 0.0, 0.918416046041877]])
LMS_TO_XYZ_MATRIX = np.linalg.inv(XYZ_TO_LMS_MATRIX)
LMS_TO_IAB_MATRIX = np.array([[65.57377049, 32.78688525, 1.63934426], [430.0, -470.0, 40.0], [49.0, 49.0, -98.0]])
IAB_TO_LMS_MATRIX = np.linalg.inv(LMS_TO_IAB_MATRIX)
CAT16_XYZ_TO_RGB = np.array([[0.401288, 0.650173, -0.051461], [-0.250268, 1.204414, 0.045854], [-0.002079, 0.048952, 0.953127]])
SCAM_SURROUND_PARAMS = {'avg': {'F': 1.0, 'c': 0.52, 'Fm': 1.0}, 'dim': {'F': 0.9, 'c': 0.5, 'Fm': 0.95}, 'dark': {'F': 0.8, 'c': 0.39, 'Fm': 0.85}}
HUE_DATA = {'h_i': np.array([16.5987, 80.2763, 157.779, 219.7174, 376.5987]), 'e_i': np.array([0.7, 0.6, 1.2, 0.9, 0.7]), 'H_i': np.array([0, 100, 200, 300, 400])}
CAT16_RGB_TO_XYZ = np.array([[1.86206786, -1.01125463, 0.14918677], [0.38752654, 0.62144744, -0.00897398], [-0.0158415, -0.03412294, 1.04996444]])
CAT16_XYZ_TO_RGB_T = CAT16_XYZ_TO_RGB.T
CAT16_RGB_TO_XYZ_T = CAT16_RGB_TO_XYZ.T
# Fixed reference of this sUCS/sCAM implementation, not the input whitepoint.
XYZW_D65 = np.array([95.047, 100.0, 108.883])
# Matrix-derived neutral axis for equal LMS responses.
SCAM_MODEL_WHITE = np.linalg.solve(XYZ_TO_LMS_MATRIX, np.ones(3))
SCAM_MODEL_WHITE *= 100.0 / SCAM_MODEL_WHITE[1]

def xyz_to_sucs(xyz):
    """Convert model-reference XYZ to sUCS (Iab)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    is_single_vector = xyz.ndim == 1
    if is_single_vector:
        xyz = xyz.reshape(1, -1)
    xyz_scaled = xyz / 100.0
    lms = xyz_scaled @ XYZ_TO_LMS_MATRIX.T
    lms_tm = np.sign(lms) * np.abs(lms) ** 0.43
    iab = lms_tm @ LMS_TO_IAB_MATRIX.T
    I = iab[:, 0]
    a = iab[:, 1]
    b = iab[:, 2]
    neutral = np.hypot(a, b) < 1e-10
    a = np.where(neutral, 0.0, a)
    b = np.where(neutral, 0.0, b)
    C = np.sqrt(a ** 2 + b ** 2)
    C_1 = np.log1p(0.0447 * C) / 0.0252
    a_1 = np.divide(C_1 * a, C, out=np.zeros_like(C), where=C != 0)
    b_1 = np.divide(C_1 * b, C, out=np.zeros_like(C), where=C != 0)
    iab_1 = np.stack((I, a_1, b_1), axis=1)
    return iab_1.flatten() if is_single_vector else iab_1

def sucs_to_xyz(iab_1):
    """Convert sUCS (Iab) to model-reference XYZ."""
    iab_1 = np.asarray(iab_1, dtype=np.float64)
    is_single_vector = iab_1.ndim == 1
    if is_single_vector:
        iab_1 = iab_1.reshape(1, -1)
    I = iab_1[:, 0]
    a_1 = iab_1[:, 1]
    b_1 = iab_1[:, 2]
    C_1 = np.sqrt(a_1 ** 2 + b_1 ** 2)
    C = np.expm1(0.0252 * C_1) / 0.0447
    a = np.divide(C * a_1, C_1, out=np.zeros_like(C_1), where=C_1 != 0)
    b = np.divide(C * b_1, C_1, out=np.zeros_like(C_1), where=C_1 != 0)
    iab = np.stack((I, a, b), axis=1)
    lms_tm = iab @ IAB_TO_LMS_MATRIX.T
    power = 1.0 / 0.43
    lms = np.sign(lms_tm) * np.abs(lms_tm) ** power
    xyz_d65 = lms @ LMS_TO_XYZ_MATRIX.T
    xyz = xyz_d65 * 100.0
    return xyz.flatten() if is_single_vector else xyz

def cat16(xyz_source: np.ndarray, xyz_w_source: np.ndarray, xyz_w_ref: np.ndarray, la: float, f_level: float, degree: float | None = None) -> np.ndarray:
    """Adapt XYZ between reference whites using CAT16."""
    xyz_source = np.asarray(xyz_source, dtype=np.float64)
    xyz_w_source = np.asarray(xyz_w_source, dtype=np.float64).reshape(1, 3)
    xyz_w_ref = np.asarray(xyz_w_ref, dtype=np.float64).reshape(1, 3)
    is_single_vector = xyz_source.ndim == 1
    if is_single_vector:
        xyz_source = xyz_source.reshape(1, 3)
    d_source = f_level * (1.0 - 1.0 / 3.6 * np.exp((-la - 42.0) / 92.0))
    d_ref = f_level * (1.0 - 1.0 / 3.6 * np.exp((-la - 42.0) / 92.0))
    d_source = np.clip(d_source, 0.0, 1.0)
    d_ref = np.clip(d_ref, 0.0, 1.0)
    if degree is not None:
        if not np.isfinite(degree) or not 0 <= degree <= 1:
            raise ValueError('degree must be between 0 and 1.')
        d_source = d_ref = float(degree)
    rgb_w_source = xyz_w_source.dot(CAT16_XYZ_TO_RGB.T)
    rgb_w_ref = xyz_w_ref.dot(CAT16_XYZ_TO_RGB.T)
    y_w_source = xyz_w_source.flatten()[1]
    y_w_ref = xyz_w_ref.flatten()[1]
    d_factor_source = d_source * (y_w_source / rgb_w_source) + 1.0 - d_source
    d_factor_ref = d_ref * (y_w_ref / rgb_w_ref) + 1.0 - d_ref
    adaptation_ratios = d_factor_source / d_factor_ref
    rgb_source = xyz_source @ CAT16_XYZ_TO_RGB_T
    rgb_adapted = rgb_source * adaptation_ratios
    xyz_adapted = np.linalg.solve(CAT16_XYZ_TO_RGB, rgb_adapted.T).T
    return xyz_adapted

def xyz_to_scam(xyz, xyz_w, y_b, l_a, surround, mode='lite'):
    """Convert XYZ to sCAM appearance coordinates."""
    if mode not in ['lite', 'full']:
        raise ValueError("mode must be 'lite' or 'full'.")
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz_w = np.asarray(xyz_w, dtype=np.float64)
    is_single_vector = xyz.ndim == 1
    if is_single_vector:
        xyz = xyz.reshape(1, -1)
    if xyz_w.ndim == 1:
        xyz_w = xyz_w.reshape(1, -1)
    s_params = SCAM_SURROUND_PARAMS.get(surround.lower())
    if s_params is None:
        raise ValueError("surround must be 'avg', 'dim' or 'dark'.")
    F, c = (s_params['F'], s_params['c'])
    n = y_b / xyz_w[:, 1, np.newaxis]
    z = 1.48 + np.sqrt(n)
    xyz_wt = SCAM_MODEL_WHITE
    # Forward: input viewing white -> fixed model reference -> sUCS.
    xyz_d65 = cat16(xyz, xyz_w, xyz_wt, l_a, F, degree=1.0)
    iab = xyz_to_sucs(xyz_d65)
    I, a, b = (iab[:, 0], iab[:, 1], iab[:, 2])
    C_1 = np.sqrt(a ** 2 + b ** 2)
    L = 100 * (I / 100) ** (c * z.flatten())
    J = L
    h = np.rad2deg(np.arctan2(b, a))
    h[h < 0] += 360
    jch_output = np.stack((J, C_1, h), axis=1)
    if mode == 'lite':
        return jch_output.flatten() if is_single_vector else jch_output
    h_i, e_i, H_i = (HUE_DATA['h_i'], HUE_DATA['e_i'], HUE_DATA['H_i'])
    h_mod = np.where(h < h_i[0], h + 360, h)
    indices = np.searchsorted(h_i, h_mod, side='right') - 1
    temp = (h_mod - h_i[indices]) / e_i[indices]
    temp1 = (h_i[indices + 1] - h_mod) / e_i[indices + 1]
    H = H_i[indices] + 100 * temp / (temp + temp1)
    D = 1.3 * np.sqrt(np.power(100 - J, 2) + 1.6 * np.power(C_1, 2))
    external_scales_output = np.stack((H, D), axis=1)
    if is_single_vector:
        return (jch_output.flatten(), external_scales_output.flatten())
    return (jch_output, external_scales_output)

def scam_to_xyz(jch, xyz_w, y_b, l_a, surround):
    """Converts sCAM (J, C, h) perceptual correlates back to XYZ tristimulus values."""
    jch = np.asarray(jch, dtype=np.float64)
    xyz_w = np.asarray(xyz_w, dtype=np.float64)
    is_single_vector = jch.ndim == 1
    if is_single_vector:
        jch = jch.reshape(1, -1)
    if xyz_w.ndim == 1:
        xyz_w = xyz_w.reshape(1, -1)
    s_params = SCAM_SURROUND_PARAMS.get(surround.lower())
    if s_params is None:
        raise ValueError("surround parameter must be one of 'avg', 'dim', or 'dark'.")
    F, c = (s_params['F'], s_params['c'])
    n = y_b / xyz_w[:, 1, np.newaxis]
    z = 1.48 + np.sqrt(n)
    J = jch[:, 0]
    I = 100.0 * (J / 100.0) ** (1.0 / (c * z.flatten()))
    C = jch[:, 1]
    h = jch[:, 2]
    h_rad = np.deg2rad(h)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    iab = np.stack((I, a, b), axis=1)
    xyz_d65 = sucs_to_xyz(iab)
    xyz_wt = SCAM_MODEL_WHITE
    # Inverse: fixed model reference -> requested viewing white.
    # Full adaptation matches the forward transform and preserves neutral colours.
    xyz_final = cat16(xyz_d65, xyz_wt, xyz_w, l_a, F, degree=1.0)
    return xyz_final.flatten() if is_single_vector else xyz_final
