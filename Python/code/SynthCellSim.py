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
    opencv-python
    pyqtgraph
    PyQt5

Make sure you have:
    - BiasStaticNoiseData64.mat
    - Cell_PCA_data.mat
in accessible file paths.

Usage:
    python SyntheticCellSimulator.py
"""

import sys
import numpy as np
import cv2
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt

from math import sqrt
from scipy.io import loadmat
from scipy.signal import convolve2d
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import gaussian_kde
from skimage.draw import polygon


class SyntheticCellSimulator:
    """
    SyntheticCellSimulator replicates:
      - DefaultSyntheticCellParams.m
      - GenerateSyntheticCellSequence.m
      - and associated helper .m files.

    Once constructed, you can:
      1) load_mat_files(...) for .mat data
      2) call GenerateSyntheticCellSequence() for results
      3) or call generate_and_visualize_pyqtgraph() to auto-plot frames
    """

    def __init__(self, params=None):
        """
        Constructor sets default parameters, mirroring DefaultSyntheticCellParams.m
        """
        if params is None:
            self.params = self.default_params()
        else:
            self.params = params

        # Placeholders for loaded .mat data
        self.bias_data = None   # Will hold 'Bias', 'RAPSD', etc.
        self.pca_data = None    # Will hold 'PCA_data' from Cell_PCA_data.mat

    # ---------------------------------------------------------------------
    # (1) "DefaultSyntheticCellParams.m" => default_params
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
        PixelspaceParam["RotationAngle"] = np.random.uniform(0, 360)
        PixelspaceParam["ScalingRatio"] = np.random.uniform(0.7, 0.8)
        PixelspaceParam["Method"] = "fractal"   # e.g. "fractal", "perlin", "sumofgauss"
        PixelspaceParam["Persistence"] = sqrt(2)
        PixelspaceParam["NbrFrames"] = param["NbrFrames"]
        PixelspaceParam["Motion"] = "shrink-expand"
        param["PixelspaceParam"] = PixelspaceParam

        # Noise parameters
        param["poiss_lambda"] = 1e-10
        param["poiss_amp"] = 20.6
        param["gauss_mu"] = 0
        param["gauss_sigma"] = 0.0181
        param["gauss_amp"] = 0.98
        param["SNR"] = -1  # in dB

        return param

    # ---------------------------------------------------------------------
    # (2) "GenerateSyntheticCellSequence.m" => GenerateSyntheticCellSequence
    # ---------------------------------------------------------------------
    def GenerateSyntheticCellSequence(self):
        """
        Replicates the logic from GenerateSyntheticCellSequence.m (MATLAB),
        including noise handling, static/dynamic noise generation, and SNR scaling.

        Returns
        -------
        I_N : ndarray
            [imsize x imsize x NbrFrames], final noisy DIC image sequence
        BW : ndarray
            [imsize x imsize x NbrFrames], binary mask(s)
        I : ndarray
            [imsize x imsize x NbrFrames], noiseless cell surface
        I_DIC : ndarray
            [imsize x imsize x NbrFrames], noiseless DIC image
        B : ndarray
            [imsize x imsize], random bias (static)
        """
        p = self.params
        NbrFrames = p["NbrFrames"]
        imsize    = p["imsize"]

        # Must have loaded .mat data
        if self.bias_data is None or self.pca_data is None:
            raise ValueError("Please call load_mat_files(...) before generating the synthetic cell sequence.")

        # 1) Generate Random Cell
        PCA_struct = self.pca_data["PCA_data"][0, 0]
        d = PCA_struct["b"].shape[0]
        x_syn_data = self.GenerateRandomCellShape(PCA_struct, d, 1)
        half = d // 2
        x, y = x_syn_data[:half, 0], x_syn_data[half:, 0]

        # 2) Embed => I, BW
        I, BW = self.EmbedCoordToImageSpace(x, y, p["PixelspaceParam"])

        # 3) DIC kernel => I_DIC
        epsf = self.DIC_EPSF(p["epsf_M"], p["epsf_shear_angle"], p["epsf_sigma"])
        I_DIC = np.empty_like(I)
        for f in range(NbrFrames):
            I_DIC[..., f] = convolve2d(I[..., f], epsf, mode='same')

        # 4) Random Bias => B
        rand_idx = np.random.rand() * self.bias_data["Bias"].shape[2]
        rand_idx = int(np.ceil(rand_idx)) - 1
        B = self.bias_data["Bias"][:, :, rand_idx]

        # 5) Static noise from RAPSD
        rand_idx_r = np.random.rand() * self.bias_data["RAPSD"].shape[1]
        rand_idx_r = int(np.ceil(rand_idx_r)) - 1
        rapsd_col = self.bias_data["RAPSD"][:, rand_idx_r]

        G = self.iRadialAvgPSD(rapsd_col)
        phase = np.exp(1j * 2 * np.pi * np.random.rand(*G.shape))
        spectrum = np.sqrt(2) * G * phase

        # ifftshift => ifft2 => real
        spectrum_unshifted = np.fft.ifftshift(spectrum)
        N_static = np.real(np.fft.ifft2(spectrum_unshifted))

        # dynamic noise arrays
        N_dyn = np.zeros((imsize, imsize, NbrFrames), dtype=float)
        N_syn = np.zeros((imsize, imsize, NbrFrames), dtype=float)

        signal_energy = np.zeros(NbrFrames, dtype=float)
        noise_energy  = np.zeros(NbrFrames, dtype=float)

        for f in range(NbrFrames):
            # Poisson + Gaussian dynamic noise
            poiss_noise = np.random.poisson(p["poiss_lambda"], size=(imsize, imsize))
            poiss_noise = p["poiss_amp"] * poiss_noise

            gauss_noise = np.random.normal(p["gauss_mu"], p["gauss_sigma"], size=(imsize, imsize))
            gauss_noise = p["gauss_amp"] * gauss_noise

            dyn_frame = poiss_noise + gauss_noise
            N_dyn[..., f] = dyn_frame

            # combined: static + dynamic
            N_syn[..., f] = N_static + dyn_frame

            # measure energies
            signal_energy[f] = np.linalg.norm(I_DIC[..., f], 'fro')**2
            noise_energy[f]  = np.linalg.norm(N_syn[..., f], 'fro')**2

        sum_signal = np.sum(signal_energy)
        sum_noise  = np.sum(noise_energy)
        scale_factor = 10.0**(p["SNR"] / 10.0) * (sum_noise / sum_signal)

        # scale signals
        I     *= scale_factor
        I_DIC *= scale_factor

        # final result
        I_N = I_DIC + N_syn
        return I_N, BW, I, I_DIC, B

    # ---------------------------------------------------------------------
    # A new method that generates & visualizes frames with PyQtGraph
    # ---------------------------------------------------------------------
    def generate_and_visualize_pyqtgraph(self):
        """
        Generates the synthetic sequence and displays the frames in real-time
        using PyQt5 + pyqtgraph, cycling through frames automatically.
        """
        # 1) Generate the data
        I_N, _, I, I_DIC, B = self.GenerateSyntheticCellSequence()

        num_frames = self.params["NbrFrames"]

        # 2) Create a QApplication if none exists
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # 3) Create the main window with a 2x2 grid of Plots
        win = pg.GraphicsLayoutWidget(show=True)
        win.setWindowTitle("DIC Cell Simulator - PyQtGraph Visualization")
        win.setWindowTitle("DIC Cell Simulator - PyQtGraph Visualization")
        win.resize(800, 800)
        win.setBackground('w')
        p1 = win.addPlot(row=0, col=0, title="Noiseless Surface (I)")
        p2 = win.addPlot(row=0, col=1, title="Noiseless DIC (I_DIC)")
        p3 = win.addPlot(row=1, col=0, title="Noisy (I_N)")
        p4 = win.addPlot(row=1, col=1, title="Noisy + Bias (I_DIC + B)")

        imgItem1 = pg.ImageItem()
        imgItem2 = pg.ImageItem()
        imgItem3 = pg.ImageItem()
        imgItem4 = pg.ImageItem()

        p1.addItem(imgItem1)
        p2.addItem(imgItem2)
        p3.addItem(imgItem3)
        p4.addItem(imgItem4)

        p1.setAspectLocked(True)
        p2.setAspectLocked(True)
        p3.setAspectLocked(True)
        p4.setAspectLocked(True)

        current_frame = 0

        def update_frames():
            nonlocal current_frame

            frame_I     = I[..., current_frame]
            frame_I_DIC = I_DIC[..., current_frame]
            frame_I_N   = I_N[..., current_frame]
            frame_bias  = I_N[..., current_frame] + B

            imgItem1.setImage(frame_I.T,     levels=(frame_I.min(),     frame_I.max()))
            imgItem2.setImage(frame_I_DIC.T, levels=(frame_I_DIC.min(), frame_I_DIC.max()))
            imgItem3.setImage(frame_I_N.T,   levels=(frame_I_N.min(),   frame_I_N.max()))
            imgItem4.setImage(frame_bias.T,  levels=(frame_bias.min(),  frame_bias.max()))

            current_frame += 1
            if current_frame >= num_frames:
                current_frame = 0  # loop from start

        timer = QtCore.QTimer()
        timer.timeout.connect(update_frames)
        timer.start(33)  # ms => ~30 FPS

        print("Launching PyQt event loop. Close window to end.")
        sys.exit(app.exec_())

    # ---------------------------------------------------------------------
    # (3) "GenerateRandomCellShape.m"
    # ---------------------------------------------------------------------
    def GenerateRandomCellShape(self, PCA_struct, NbrCoeffs, NbrSyntheticCells):
        """
        GenerateRandomCellShape using KDE sampling, replicating the MATLAB code:
          x_syn = x_bar + V * b_rand
        for the last NbrCoeffs PCA dimensions.
        """
        b = PCA_struct["b"]   # shape: (d, N)
        V = PCA_struct["V"]   # shape: (d, d)
        x_bar = PCA_struct["x_bar"].flatten()  # ensure shape (d,)
        d, N = b.shape

        print(f"PCA_struct dimensions: b={b.shape}, V={V.shape}, x_bar={x_bar.shape}")

        x_syn_data = np.zeros((d, NbrSyntheticCells))

        c = 0
        while c < NbrSyntheticCells:
            b_rand = np.zeros(d)
            for i in range(d - NbrCoeffs, d):
                kde = gaussian_kde(b[i, :])  # fit KDE to i-th row of b
                b_rand[i] = kde.resample(1).item()  # sample 1 random value

            x_syn = x_bar + V @ b_rand
            print(f"x_syn shape: {x_syn.shape}")

            # Only keep if no cross-over
            if not self.CheckCrossOver(x_syn):
                x_syn_data[:, c] = x_syn
                c += 1

        return x_syn_data

    # ---------------------------------------------------------------------
    # (4) "EmbedCoordToImageSpace.m"
    # ---------------------------------------------------------------------
    def EmbedCoordToImageSpace(self, x, y, Param):
        """
        Replicates the MATLAB code. We generate a 2D cell shape in 'I_2D'
        and then stack frames or apply motion logic if NbrFrames>1.
        """
        ImSize = Param["ImSize"]
        RotationAngle = Param["RotationAngle"]
        ScalingRatio  = Param["ScalingRatio"]
        Method        = Param["Method"].lower()
        NbrFrames     = Param["NbrFrames"]

        # 1) Rotate
        theta = np.deg2rad(RotationAngle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        xy_rot = R @ np.vstack((x, y))

        # 2) Interpolate (spline) to make closed shape
        tck, u = splprep([xy_rot[0, :], xy_rot[1, :]], s=0, per=True)
        u_new = np.linspace(0, 1, 200)
        x_int, y_int = splev(u_new, tck)

        # 3) Scale and center
        max_width = np.max(np.abs([x_int, y_int]))
        x_int = (x_int / max_width) * ScalingRatio * (ImSize / 2)
        y_int = (y_int / max_width) * ScalingRatio * (ImSize / 2)

        x_centered = x_int + (ImSize / 2)
        y_centered = y_int + (ImSize / 2)

        # 4) Create a binary mask (BW_2D)
        BW_2D = np.zeros((ImSize, ImSize), dtype=np.uint8)
        rr, cc = polygon(y_centered, x_centered, shape=BW_2D.shape)
        BW_2D[rr, cc] = 1

        # 5) Generate a noise field (fractal/perlin)
        if Method == "perlin":
            P_N = self.perlin_noise((ImSize, ImSize), Param.get("Persistence", sqrt(2)))
        elif Method == "fractal":
            P_N = self.fractal_noise((ImSize, ImSize), Param.get("Persistence", 1.0))
        else:
            P_N = np.zeros((ImSize, ImSize), dtype=float)

        # Normalize P_N to [0..1]
        P_N -= np.min(P_N)
        max_p = np.max(P_N)
        if max_p > 1e-12:
            P_N /= max_p

        # Basic cell image
        H = gaussian_filter(BW_2D.astype(float), sigma=3)
        I_2D = H + BW_2D * P_N

        if NbrFrames == 1:
            # single-frame => return [H,W,1] arrays
            return I_2D[..., None], BW_2D[..., None]

        # multi-frame => apply motion logic
        return self._create_motion_sequence(I_2D, BW_2D, P_N, Param)

    def _create_motion_sequence(self, I_2D, BW_2D, P_N, Param):
        """
        Creates a multi-frame sequence by applying barrel distortion, translations, etc.
        """
        ImSize    = Param["ImSize"]
        Motion    = Param["Motion"].lower()
        NbrFrames = Param["NbrFrames"]

        K = BW_2D.astype(float)
        L = P_N.copy()

        I_stack  = np.zeros((ImSize, ImSize, NbrFrames), dtype=float)
        BW_stack = np.zeros((ImSize, ImSize, NbrFrames), dtype=np.uint8)

        def a_of_f(f):
            """
            Motion parameter that can shrink/expand the shape each frame.
            """
            if Motion == 'asympotic':
                return 1e-2 * np.exp(-(f**1.3))
            elif Motion == 'shrink-expand':
                turning_point = 0.25
                return ((NbrFrames - f) / NbrFrames - turning_point) * 9e-6
            elif Motion == 'expand-shrink':
                turning_point = 0.25
                return -(((NbrFrames - f)/NbrFrames) - turning_point) * 9e-6
            else:
                return 0.0

        # define a random translation path
        trans_sigma = 2
        trans_start = trans_sigma * np.random.randn(2)
        trans_end   = -np.sign(trans_start) * np.abs(trans_sigma * np.random.randn(2))

        for f in range(NbrFrames):
            af = a_of_f(f)

            # apply barrel distortion
            K = self.barrel_distortion(K, af)
            L = self.barrel_distortion(L, af)
            K_mask = np.round(K).astype(np.uint8)

            # re-combine
            J = gaussian_filter(K_mask.astype(float), sigma=3) + (K_mask * L)

            # translation
            t = np.exp(-f/(NbrFrames/2))
            tx = trans_start[0] + t*(trans_end[0] - trans_start[0])
            ty = trans_start[1] + t*(trans_end[1] - trans_start[1])

            M_trans = np.float32([[1, 0, tx],
                                  [0, 1, ty]])
            J_trans  = cv2.warpAffine(J, M_trans, (ImSize, ImSize))
            BW_trans = cv2.warpAffine(K_mask.astype(float), M_trans, (ImSize, ImSize))
            BW_trans = np.ceil(BW_trans).astype(np.uint8)

            I_stack[..., f]  = J_trans
            BW_stack[..., f] = BW_trans

        return I_stack, BW_stack

    # ---------------------------------------------------------------------
    # (5) "DIC_EPSF.m"
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
    # (6) "iRadialAvgPSD.m"
    # ---------------------------------------------------------------------
    def iRadialAvgPSD(self, PSD):
        """
        Python version of iRadialAvgPSD.m
        PSD is a 1D array of radial amplitudes.
        """
        spectral_size = (len(PSD) - 1) * 2
        center = [spectral_size/2, spectral_size/2]
        F = np.ones((spectral_size, spectral_size), dtype=float) * PSD[-1]

        X, Y = np.meshgrid(np.arange(spectral_size), np.arange(spectral_size))
        for r in range(len(PSD)-1, 0, -1):
            radius = r
            XY = ((X - center[0])**2 + (Y - center[1])**2) <= radius**2
            F[XY] = PSD[r]

        F = np.fft.ifftshift(F)
        # DC component
        F[0, 0] = PSD[0]
        return F

    # ---------------------------------------------------------------------
    # (7) "CheckCrossOver.m"
    # ---------------------------------------------------------------------
    def CheckCrossOver(self, x_temp):
        """
        Python version of CheckCrossOver.m
        x_temp is 1D with the first half = x coords, second half = y coords.
        """
        d = len(x_temp)
        half = d // 2
        cross_itself = False

        for p in range(half):
            x_part = np.concatenate((x_temp[p:half], x_temp[:p]))
            y_part = np.concatenate((x_temp[half + p:d], x_temp[half:half + p]))

            x0, y0 = x_part[0], y_part[0]
            x_part -= x0
            y_part -= y0

            dx, dy = x_part[1] - x_part[0], y_part[1] - y_part[0]
            rot_angle = -np.arctan2(dy, dx)
            R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                          [np.sin(rot_angle),  np.cos(rot_angle)]])
            xy_rot = R @ np.vstack((x_part, y_part))
            x_part, y_part = xy_rot[0, :], xy_rot[1, :]

            coord1 = [x_part[0], y_part[0]]
            coord2 = [x_part[1], y_part[1]]

            for k in range(2, len(x_part) - 1):
                coord3 = [x_part[k], y_part[k]]
                coord4 = [x_part[k + 1], y_part[k + 1]]

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
                if (
                    min(coord1[0], coord2[0]) < x_cross < max(coord1[0], coord2[0])
                    and min(coord3[0], coord4[0]) < x_cross < max(coord3[0], coord4[0])
                ):
                    cross_itself = True
                    break

            if cross_itself:
                break

        return cross_itself

    # ---------------------------------------------------------------------
    # Additional helper methods: perlin_noise, fractal_noise, barrel_distortion
    # ---------------------------------------------------------------------
    def perlin_noise(self, shape, persistence=0.0):
        """
        Approx python version of perlin_noise from the MATLAB code,
        using repeated octaves and upsampling.
        """
        n, m = shape
        result = np.zeros((n, m), dtype=float)
        i = 0
        w = np.sqrt(n*m)
        while w > 3:
            i += 1
            # Create a smaller random field
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
        1/f^p fractal (pink) noise. 
        """
        N, M = shape
        y, x = np.linspace(-0.5, 0.5, N), np.linspace(-0.5, 0.5, M)
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = 1.0 / (D**p)
        H[np.isnan(H)] = 0.0
        if np.linalg.norm(H) > 0:
            H /= np.linalg.norm(H)

        rand_phase = np.fft.fft2(np.random.randn(N, M))
        F = rand_phase * np.fft.fftshift(H)
        im = np.fft.ifft2(F).real
        return im

    def barrel_distortion(self, I, a):
        """
        If a>0 => 'shrink', a<0 => 'expand'. Replicates a mild radial transform.
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
    # (8) "Demo.m" => if __name__ == "__main__": block
    # ---------------------------------------------------------------------
    def load_mat_files(self, bias_file, pca_file):
        """
        Load .mat files. Must contain 'Bias', 'RAPSD', and 'PCA_data'.
        """
        try:
            self.bias_data = loadmat(bias_file)
        except Exception as e:
            raise ValueError(f"Error loading bias file: {bias_file} - {str(e)}")

        try:
            self.pca_data = loadmat(pca_file)
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
# Usage example: replicate Demo.m in an if __name__ == "__main__": block
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Create simulator instance and set parameters
    simulator = SyntheticCellSimulator()
    # e.g., 100 frames (like the MATLAB script)
    simulator.params["NbrFrames"] = 100
    simulator.params["PixelspaceParam"]["NbrFrames"] = 100
    simulator.params["SNR"] = -10
    simulator.params["PixelspaceParam"]["RotationAngle"] = 45
    simulator.params["PixelspaceParam"]["ScalingRatio"] = 0.75
    simulator.params["PixelspaceParam"]["Method"] = "fractal"


    # Load the required .mat files (adjust paths accordingly)
    bias_file = r"C:\Users\sa-forest\Documents\GitHub\DIC-Cell-Simulator\MATLAB\BiasStaticNoiseData64.mat"
    pca_file  = r"C:\Users\sa-forest\Documents\GitHub\DIC-Cell-Simulator\MATLAB\Cell_PCA_data.mat"
    simulator.load_mat_files(bias_file, pca_file)

    # Instead of a Matplotlib-based for-loop, call our PyQtGraph visualizer:
    simulator.generate_and_visualize_pyqtgraph()


