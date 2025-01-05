#!/usr/bin/env python3
"""
SyntheticCellSimulator: A Python class that replicates the MATLAB DIC Cell Simulator Toolbox.

References to original MATLAB files:
  1) DefaultSyntheticCellParams.m
  2) GenerateSyntheticCellSequence.m
  3) GenerateRandomCellShape.m
  4) EmbedCoordToImageSpace.m
  5) DIC_EPSF.m
  6) iRadialAvgPSD.m
  7) CheckCrossOver.m
  8) Demo.m (replicated in the if __name__ == "__main__" block)

Requires:
    numpy
    scipy
    scikit-image
    matplotlib
    opencv-python (if you use certain image transformations)

Make sure you have:
    - BiasStaticNoiseData64.mat
    - Cell_PCA_data.mat
in accessible locations.

Usage:
    python synthetic_cell_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from math import sqrt
from scipy.io import loadmat
from scipy.signal import convolve2d
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.draw import polygon


class SyntheticCellSimulator:
    """
    SyntheticCellSimulator replicates the functionality of:
      - DefaultSyntheticCellParams.m
      - GenerateSyntheticCellSequence.m
      - and associated helper .m files.
    """

    def __init__(self, params=None):
        """
        Constructor sets default parameters, mirroring DefaultSyntheticCellParams.m

        If params is not provided, we create default ones:
           param.imsize = 64
           param.NbrFrames = 100
           param.epsf_M = 5
           param.epsf_shear_angle = 225
           param.epsf_sigma = 0.5
           param.PixelspaceParam  (a sub-dict)
           param.poiss_lambda, param.poiss_amp, ...
           param.SNR, etc.
        """
        if params is None:
            self.params = self.default_params()
        else:
            self.params = params

        # Placeholders for loaded .mat data
        self.bias_data = None   # Will hold 'Bias', 'RAPSD', etc.
        self.pca_data = None    # Will hold 'PCA_data' from Cell_PCA_data.mat

    # ---------------------------------------------------------------------
    # (1) "DefaultSyntheticCellParams.m" => default_params method
    # ---------------------------------------------------------------------
    def default_params(self):
        """
        Equivalent to DefaultSyntheticCellParams.m
        """
        param = {}
        param["imsize"] = 64
        param["NbrFrames"] = 100

        # DIC kernel / EPSF
        param["epsf_M"] = 5
        param["epsf_shear_angle"] = 225
        param["epsf_sigma"] = 0.5

        # PixelSpaceParam sub-structure
        PixelspaceParam = {}
        PixelspaceParam["ImSize"] = param["imsize"]
        # for rotation, scaling, etc., we pick random or defaults
        PixelspaceParam["RotationAngle"] = np.random.uniform(0, 360)
        PixelspaceParam["ScalingRatio"] = np.random.uniform(0.7, 0.8)
        PixelspaceParam["Method"] = "fractal"      # e.g., "fractal", "perlin"
        PixelspaceParam["Persistence"] = sqrt(2)   # e.g., used for fractal/perlin noise
        PixelspaceParam["NbrFrames"] = param["NbrFrames"]
        PixelspaceParam["Motion"] = "shrink-expand"
        param["PixelspaceParam"] = PixelspaceParam

        # Noise parameters
        param["poiss_lambda"] = 1e-10
        param["poiss_amp"] = 20.6
        param["gauss_mu"] = 0
        param["gauss_sigma"] = 0.0181
        param["gauss_amp"] = 0.98
        param["SNR"] = -1

        return param

    # ---------------------------------------------------------------------
    # (2) "GenerateSyntheticCellSequence.m" => method name: GenerateSyntheticCellSequence
    # ---------------------------------------------------------------------
    def GenerateSyntheticCellSequence(self):
        """
        Replicates the logic in GenerateSyntheticCellSequence.m

        Steps:
          1) Load PCA data (for cell shape) from self.pca_data
          2) Generate a random cell shape
          3) Embed coords into image space to get I, BW
          4) Convolve with DIC kernel => I_DIC
          5) [Could handle bias, noise, SNR, etc. if we replicate the entire pipeline]
          6) Return minimal or extended set of arrays

        For demonstration, we'll return:
           I  : The cell intensity image (or multi-frame stack)
           BW : The binary mask (or multi-frame stack)
           I_DIC : The DIC-convolved image
        """
        # gather parameters
        p = self.params
        NbrFrames = p["NbrFrames"]
        imsize = p["imsize"]

        # (A) Check that the .mat files have been loaded
        if self.bias_data is None or self.pca_data is None:
            raise ValueError("Please call load_mat_files(...) before generating the synthetic cell sequence.")

        # Extract PCA data structure
        # The loaded .mat typically has pca_data['PCA_data'][0,0],
        # which in turn might contain 'V', 'x_bar', 'b', etc.
        PCA_struct = self.pca_data["PCA_data"][0, 0]
        # e.g. V = PCA_struct['V'], x_bar = PCA_struct['x_bar']

        # Step #1: Generate random cell shape
        # We follow GenerateRandomCellShape.m logic
        d = PCA_struct["b"].shape[0]  # dimension
        NbrCoeffs = d
        x_syn_data = self.GenerateRandomCellShape(PCA_struct, NbrCoeffs, 1)

        # x_syn_data -> first half is x, second half is y
        half = d // 2
        x = x_syn_data[:half, 0]
        y = x_syn_data[half:, 0]

        # Step #2: embed coords => I, BW
        # replicate EmbedCoordToImageSpace.m
        I, BW = self.EmbedCoordToImageSpace(x, y, p["PixelspaceParam"])

        # Step #3: DIC convolution => I_DIC
        # replicate DIC_EPSF.m usage
        epsf = self.DIC_EPSF(p["epsf_M"], p["epsf_shear_angle"], p["epsf_sigma"])

        # If multi-frame, I has shape [H,W,F], so convolve each frame
        if I.ndim == 2:
            # single frame
            I_DIC = convolve2d(I, epsf, mode='same')
        else:
            # multiple frames
            nF = I.shape[2]
            I_DIC = np.zeros_like(I)
            for f in range(nF):
                I_DIC[:, :, f] = convolve2d(I[:, :, f], epsf, mode='same')

        # Step #4: Return (for now, the minimal set). The original .m also handles
        # bias (B), static/dynamic noise, etc. If you want to replicate the entire
        # pipeline, we can do that, but here's the minimal 3-output version:
        return I, BW, I_DIC

    # ---------------------------------------------------------------------
    # (3) "GenerateRandomCellShape.m"
    # ---------------------------------------------------------------------
    def GenerateRandomCellShape(self, PCA_struct, NbrCoeffs, NbrSyntheticCells):
        """
        Python version of GenerateRandomCellShape.m

        PCA_struct is typically something like:
            PCA_struct['b']      # (d, N)
            PCA_struct['V']      # (d, d) or similar
            PCA_struct['x_bar']  # (d,)
        NbrCoeffs = number of coefficients to use
        NbrSyntheticCells = how many shapes to generate

        We replicate the kernel density approach or direct random sampling from 'b'.
        We also apply CheckCrossOver() to skip shapes that cross themselves.
        """
        b = PCA_struct['b']      # shape (d, N)
        V = PCA_struct['V']      # shape (d, d)
        x_bar = PCA_struct['x_bar']  # shape (d,)
        d, N = b.shape

        # We'll generate random cells
        x_syn_data = np.zeros((d, NbrSyntheticCells), dtype=float)

        c = 0
        while c < NbrSyntheticCells:
            # create random coeffs
            b_rand = np.zeros((d,), dtype=float)
            # sample from the distribution in b, especially the last NbrCoeffs entries
            # for i in range(d - NbrCoeffs, d):
            for i in range(d - NbrCoeffs, d):
                data = b[i, :]  # all examples
                # simplest approach: pick a random value from data
                b_rand[i] = np.random.choice(data)

            # Synthesize shape
            x_syn = x_bar + V @ b_rand

            # Check cross-over
            if not self.CheckCrossOver(x_syn):
                x_syn_data[:, c] = x_syn
                c += 1

        return x_syn_data

    # ---------------------------------------------------------------------
    # (4) "EmbedCoordToImageSpace.m"
    # ---------------------------------------------------------------------
    def EmbedCoordToImageSpace(self, x, y, Param):
        """
        Python version of EmbedCoordToImageSpace.m

        x, y: shape coordinates
        Param: a dict with:
          ImSize, RotationAngle, ScalingRatio, Method, Persistence, NbrFrames, Motion, etc.

        Returns (I, BW): either single-frame or multi-frame data.
        """
        ImSize = Param["ImSize"]
        RotationAngle = Param["RotationAngle"]
        ScalingRatio = Param["ScalingRatio"]
        Method = Param.get("Method", "fractal")
        Persistence = Param.get("Persistence", sqrt(2))
        NbrFrames = Param.get("NbrFrames", 1)
        Motion = Param.get("Motion", "shrink-expand")

        # 1) rotate
        theta = np.deg2rad(RotationAngle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        xy = np.vstack((x, y))
        xy_rot = R @ xy

        # 2) Spline interp + scaling
        (tck, u) = splprep([xy_rot[0, :], xy_rot[1, :]], s=0, per=True)
        u_new = np.linspace(0, 1, 200)
        x_int, y_int = splev(u_new, tck)

        # figure out the bounding
        Max_Width = np.max(np.abs([x_int, y_int]))
        x_int = x_int / Max_Width * ScalingRatio * (ImSize / 2.0)
        y_int = y_int / Max_Width * ScalingRatio * (ImSize / 2.0)

        # 3) build binary mask
        x_centered = x_int + (ImSize / 2.0)
        y_centered = y_int + (ImSize / 2.0)
        BW_2D = np.zeros((ImSize, ImSize), dtype=np.uint8)
        rr, cc = polygon(y_centered, x_centered, shape=BW_2D.shape)
        BW_2D[rr, cc] = 1

        # 4) generate texture
        if Method.lower() == "fractal":
            P_N = self.fractal_noise((ImSize, ImSize), Persistence)
        elif Method.lower() == "perlin":
            P_N = self.perlin_noise((ImSize, ImSize), Persistence)
        else:
            P_N = np.zeros((ImSize, ImSize), dtype=float)

        # normalize
        P_N -= np.min(P_N)
        if np.max(P_N) > 1e-12:
            P_N /= np.max(P_N)

        # approximate a disk filter with Gaussian
        H = gaussian_filter(BW_2D.astype(float), sigma=3)
        I_2D = H + BW_2D * P_N

        # if 1 frame, return
        if NbrFrames == 1:
            return I_2D, BW_2D

        # if multi-frame, replicate the motion logic from the MATLAB code
        return self._create_motion_sequence(I_2D, BW_2D, P_N, Param)

    def _create_motion_sequence(self, I_2D, BW_2D, P_N, Param):
        """
        Helper: replicates the multi-frame motion logic from EmbedCoordToImageSpace.m
        (barrel distortion, translation, etc.)
        """
        ImSize = Param["ImSize"]
        Motion = Param["Motion"]
        NbrFrames = Param["NbrFrames"]
        Persistence = Param["Persistence"]

        # we keep track of two 'maps': K for the mask, L for the noise
        K = BW_2D.astype(float)
        L = P_N.copy()

        I_stack = np.zeros((ImSize, ImSize, NbrFrames), dtype=float)
        BW_stack = np.zeros((ImSize, ImSize, NbrFrames), dtype=np.uint8)

        # define motion function a(f)
        def a_of_f(f):
            if Motion.lower() == 'asympotic':
                return 1e-2 * np.exp(-(f**1.3))
            elif Motion.lower() == 'shrink-expand':
                turning_point = 0.25
                return ((NbrFrames - f) / NbrFrames - turning_point) * 9e-6
            elif Motion.lower() == 'expand-shrink':
                turning_point = 0.25
                return -(((NbrFrames - f)/NbrFrames) - turning_point) * 9e-6
            else:
                return 0.0

        # define a random translation path
        trans_sigma = 2
        trans_start = trans_sigma * np.random.randn(2)
        trans_end = -np.sign(trans_start) * np.abs(trans_sigma * np.random.randn(2))

        for f in range(NbrFrames):
            af = a_of_f(f)

            # barrel-distortion equivalent
            K = self.barrel_distortion(K, af)
            L = self.barrel_distortion(L, af)
            K_mask = np.round(K).astype(np.uint8)

            # combine with smooth filter
            J = gaussian_filter(K_mask.astype(float), sigma=3) + (K_mask * L)

            # translation
            t = np.exp(-f/(NbrFrames/2))
            tx = trans_start[0] + t*(trans_end[0] - trans_start[0])
            ty = trans_start[1] + t*(trans_end[1] - trans_start[1])

            # warp with OpenCV
            M_trans = np.float32([[1, 0, tx],
                                  [0, 1, ty]])
            J_trans = cv2.warpAffine(J, M_trans, (ImSize, ImSize))
            BW_trans = cv2.warpAffine(K_mask.astype(float), M_trans, (ImSize, ImSize))
            BW_trans = np.ceil(BW_trans).astype(np.uint8)

            I_stack[:, :, f] = J_trans
            BW_stack[:, :, f] = BW_trans

        return I_stack, BW_stack

    # ---------------------------------------------------------------------
    # (5) "DIC_EPSF.m" => DIC_EPSF method
    # ---------------------------------------------------------------------
    def DIC_EPSF(self, M, shear_angle, epsf_sigma):
        """
        Python version of DIC_EPSF.m
        """
        xg, yg = np.meshgrid(np.arange(1, M+1), np.arange(1, M+1))
        xg = xg - np.mean(xg[0, :])
        yg = yg - np.mean(yg[:, 0])
        shear_angle_rad = np.deg2rad(shear_angle)
        exponent = np.exp(-((xg**2 + yg**2) / (epsf_sigma**2)))
        EPSF = (-xg * exponent * np.cos(shear_angle_rad)
                -yg * exponent * np.sin(shear_angle_rad))
        return EPSF

    # ---------------------------------------------------------------------
    # (6) "iRadialAvgPSD.m" => iRadialAvgPSD method
    # ---------------------------------------------------------------------
    def iRadialAvgPSD(self, PSD):
        """
        Python version of iRadialAvgPSD.m
        """
        spectral_size = (len(PSD) - 1) * 2
        center = [spectral_size/2, spectral_size/2]
        F = np.ones((spectral_size, spectral_size), dtype=float) * PSD[-1]

        X, Y = np.meshgrid(np.arange(spectral_size), np.arange(spectral_size))
        # fill from outside in
        for r in range(len(PSD)-1, 0, -1):
            radius = r
            XY = ((X - center[0])**2 + (Y - center[1])**2) <= radius**2
            F[XY] = PSD[r]

        # shift
        F = np.fft.ifftshift(F)
        # DC
        F[0, 0] = PSD[0]
        return F

    # ---------------------------------------------------------------------
    # (7) "CheckCrossOver.m" => CheckCrossOver method
    # ---------------------------------------------------------------------
    def CheckCrossOver(self, x_temp):
        """
        Python version of CheckCrossOver.m
        x_temp is 1D with concatenated (x1..xK, y1..yK).
        Returns True if shape crosses over itself.
        """
        d = len(x_temp)
        half = d // 2
        cross_itself = False
        # make a float copy
        x_temp = x_temp.astype(float).copy()

        for p in range(half):
            # reshuffle
            x_part = np.concatenate([x_temp[p:half], x_temp[:p]])
            y_part = np.concatenate([x_temp[half+p:d], x_temp[half:half+p]])
            # normalize by first
            x0 = x_part[0]
            y0 = y_part[0]
            x_part -= x0
            y_part -= y0

            # rotate so first->second is horizontal
            rot_angle = -np.arctan2(y_part[1], x_part[1])
            cosA, sinA = np.cos(rot_angle), np.sin(rot_angle)
            xy = np.vstack((x_part, y_part))
            R = np.array([[cosA, -sinA], [sinA, cosA]])
            xy_rot = R @ xy
            x_part = xy_rot[0, :]
            y_part = xy_rot[1, :]

            coord1 = [x_part[0], y_part[0]]
            coord2 = [x_part[1], y_part[1]]

            # check cross with all segments from k=2.. half-1
            for k in range(2, half - 1):
                coord3 = [x_part[k],     y_part[k]]
                coord4 = [x_part[k + 1], y_part[k + 1]]
                # slopes
                denom1 = coord2[0] - coord1[0]
                denom2 = coord4[0] - coord3[0]
                if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
                    continue
                m1 = (coord2[1] - coord1[1]) / denom1
                c1 = coord2[1] - m1 * coord2[0]
                m2 = (coord4[1] - coord3[1]) / denom2
                c2 = coord4[1] - m2 * coord4[0]
                if abs(m1 - m2) < 1e-12:
                    continue
                x_cross = (c2 - c1) / (m1 - m2)
                # check if crossing is within segment bounds
                if (min(coord1[0], coord2[0]) < x_cross < max(coord1[0], coord2[0])):
                    if (min(coord3[0], coord4[0]) < x_cross < max(coord3[0], coord4[0])):
                        cross_itself = True
                        break
            if cross_itself:
                break

        return cross_itself

    # ---------------------------------------------------------------------
    # Additional helper functions to replicate functionality from .m code
    # (e.g., perlin_noise, fractal_noise, barrel_distortion)
    # ---------------------------------------------------------------------
    def perlin_noise(self, shape, persistence=0.0):
        """
        Approx Python version of perlin_noise from the MATLAB code.
        """
        n, m = shape
        result = np.zeros((n, m), dtype=float)
        i = 0
        w = np.sqrt(n*m)
        while w > 3:
            i += 1
            # small random field upscaled
            # (use cv2 for interpolation or direct rescale)
            d = np.random.randn(n//2+1, m//2+1).astype(np.float32)
            d_big = cv2.resize(d, (m, n), interpolation=cv2.INTER_CUBIC)
            if abs(persistence) < 1e-12:
                result += i * d_big
            else:
                result += (persistence**i) * d_big
            w = w - np.ceil(w/2 - 1)
        return result

    def fractal_noise(self, shape, p=1.0):
        """
        1/f^p fractal (pink) noise, akin to fractal_noise in MATLAB code.
        """
        N, M = shape
        hr, hc = (N-1)//2, (M-1)//2
        y, x = np.mgrid[-hr:hr+1, -hc:hc+1]
        D = np.sqrt(x**2 + y**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            H = 1.0 / (D**p)
        H[np.isnan(H)] = 1.0
        H = H / np.linalg.norm(H)

        rand_phase = np.fft.fft2(np.random.randn(N, M))
        F = rand_phase * np.fft.fftshift(H)
        im = np.fft.ifft2(F).real
        return im

    def barrel_distortion(self, I, a):
        """
        Python version of barrel_distortion from the MATLAB code in EmbedCoordToImageSpace.
        If a>0 => image shrinks, if a<0 => expands.
        """
        nrows, ncols = I.shape
        xi, yi = np.meshgrid(np.arange(ncols), np.arange(nrows))
        imid = ncols // 2
        jmid = nrows // 2

        xt = xi - imid
        yt = yi - jmid
        r = np.sqrt(xt**2 + yt**2)
        theta = np.arctan2(yt, xt)
        s = r + a*(r**3)
        ut = s*np.cos(theta) + imid
        vt = s*np.sin(theta) + jmid

        coords = np.vstack((vt.ravel(), ut.ravel()))
        I_barrel = map_coordinates(I, coords, order=1, cval=0).reshape(I.shape)
        return I_barrel

    # ---------------------------------------------------------------------
    # (8) "Demo.m" => We'll replicate as an if __name__ == "__main__" block
    # ---------------------------------------------------------------------
    def load_mat_files(self, bias_file, pca_file):
        """
        Load .mat files (equivalent usage to what's in GenerateSyntheticCellSequence.m)
        Checking for 'Bias', 'RAPSD', 'Cell_PCA_data' or 'PCA_data' as needed.
        """
        try:
            self.bias_data = loadmat(bias_file)
            # Typically: self.bias_data['Bias'], self.bias_data['RAPSD'], ...
        except Exception as e:
            raise ValueError(f"Error loading bias file: {bias_file} - {str(e)}")

        try:
            self.pca_data = loadmat(pca_file)
            # Typically: self.pca_data['PCA_data'][0,0]
        except Exception as e:
            raise ValueError(f"Error loading PCA file: {pca_file} - {str(e)}")

        # Validate keys
        for key in ["Bias", "RAPSD"]:
            if key not in self.bias_data:
                raise ValueError(f"Bias .mat file missing '{key}'")

        if "PCA_data" not in self.pca_data:
            raise ValueError("PCA .mat file missing 'PCA_data'")

        print("Successfully loaded .mat files.")

# ---------------------------------------------------------------------
# (8) Demo.m => replicate in an if __name__ == "__main__" block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create instance
    simulator = SyntheticCellSimulator()

    # Just like Demo.m, we can set custom parameters
    simulator.params["SNR"] = -10
    simulator.params["NbrFrames"] = 5
    simulator.params["PixelspaceParam"]["RotationAngle"] = 45
    simulator.params["PixelspaceParam"]["ScalingRatio"] = 0.75
    simulator.params["PixelspaceParam"]["Method"] = "fractal"

    # Load .mat files (update paths as appropriate)
    bias_file = r"C:\Users\sa-forest\Documents\GitHub\DIC-Cell-Simulator\MATLAB\BiasStaticNoiseData64.mat"
    pca_file = r"C:\Users\sa-forest\Documents\GitHub\DIC-Cell-Simulator\MATLAB\Cell_PCA_data.mat"
    simulator.load_mat_files(bias_file, pca_file)

    # Generate the synthetic cell sequence
    I, BW, I_DIC = simulator.GenerateSyntheticCellSequence()

    # Now replicate the plotting style of Demo.m
    # We'll do a multi-frame style loop if 3D, or single if 2D.
    if I.ndim == 2:
        # single frame
        plt.figure(figsize=(10, 8))
        # subplot(2,2,1): I
        plt.subplot(2, 2, 1)
        plt.imshow(I, cmap='gray')
        plt.title("I")
        plt.axis('off')

        # subplot(2,2,2): I_DIC
        plt.subplot(2, 2, 2)
        plt.imshow(I_DIC, cmap='gray')
        plt.title("I_DIC")
        plt.axis('off')

        # subplot(2,2,3): BW
        plt.subplot(2, 2, 3)
        plt.imshow(BW, cmap='gray')
        plt.title("BW")
        plt.axis('off')

        # subplot(2,2,4): let's overlay BW on top of I_DIC
        plt.subplot(2, 2, 4)
        plt.imshow(I_DIC, cmap='gray')
        bw_mask = np.ma.masked_where(BW == 0, BW)
        plt.imshow(bw_mask, cmap='jet', alpha=0.4)
        plt.title("I_DIC + BW overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        # multi-frame
        n_frames = I.shape[2]
        for f in range(n_frames):
            plt.figure(figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(I[:, :, f], cmap='gray')
            plt.title(f"I (Frame {f+1})")
            plt.axis('off')
            plt.contour(BW[:, :, f], [0.5], colors='b')

            plt.subplot(2, 2, 2)
            plt.imshow(I_DIC[:, :, f], cmap='gray')
            plt.title("I_DIC")
            plt.axis('off')
            plt.contour(BW[:, :, f], [0.5], colors='b')

            plt.subplot(2, 2, 3)
            plt.imshow(BW[:, :, f], cmap='gray')
            plt.title("BW")
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(I_DIC[:, :, f], cmap='gray')
            bw_mask = np.ma.masked_where(BW[:, :, f] == 0, BW[:, :, f])
            plt.imshow(bw_mask, cmap='jet', alpha=0.3)
            plt.title("I_DIC + BW overlay")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    print("Done generating synthetic data. Script finished.")
