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
from scipy.stats import gaussian_kde
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
        Replicates the logic in GenerateSyntheticCellSequence.m, including noise handling.
        """
        # Gather parameters
        p = self.params
        NbrFrames = p["NbrFrames"]
        imsize = p["imsize"]

        # Check that the .mat files have been loaded
        if self.bias_data is None or self.pca_data is None:
            raise ValueError("Please call load_mat_files(...) before generating the synthetic cell sequence.")

        # Extract PCA data structure
        PCA_struct = self.pca_data["PCA_data"][0, 0]

        # Generate random cell shape
        d = PCA_struct["b"].shape[0]
        x_syn_data = self.GenerateRandomCellShape(PCA_struct, d, 1)
        half = d // 2
        x, y = x_syn_data[:half, 0], x_syn_data[half:, 0]

        # Embed coordinates into image space
        I, BW = self.EmbedCoordToImageSpace(x, y, p["PixelspaceParam"])

        # Generate DIC kernel
        epsf = self.DIC_EPSF(p["epsf_M"], p["epsf_shear_angle"], p["epsf_sigma"])
        I_DIC = np.stack([convolve2d(I[:, :, f], epsf, mode='same') for f in range(NbrFrames)], axis=-1)

        # Extract random bias
        B = self.bias_data["Bias"][:, :, np.random.randint(self.bias_data["Bias"].shape[2])]

        # Generate static noise
        RAPSD = self.bias_data["RAPSD"][:, np.random.randint(self.bias_data["RAPSD"].shape[1])]
        G = self.iRadialAvgPSD(RAPSD)
        N_static = np.real(np.fft.ifft2(G * np.exp(1j * 2 * np.pi * np.random.rand(*G.shape))))

        # Generate dynamic noise
        N_dyn = np.zeros((imsize, imsize, NbrFrames))
        for f in range(NbrFrames):
            poiss_noise = np.random.poisson(p["poiss_lambda"], size=(imsize, imsize)) * p["poiss_amp"]
            gauss_noise = np.random.normal(p["gauss_mu"], p["gauss_sigma"], size=(imsize, imsize)) * p["gauss_amp"]
            N_dyn[:, :, f] = poiss_noise + gauss_noise

        # Add static and dynamic noise
        N_syn = N_static[:, :, None] + N_dyn
        I_N = I_DIC + N_syn

        # Scale signal for the desired SNR
        signal_energy = np.sum(np.linalg.norm(I_DIC, axis=(0, 1))**2) / NbrFrames
        noise_energy = np.sum(np.linalg.norm(N_syn, axis=(0, 1))**2) / NbrFrames
        scale_factor = 10**(p["SNR"] / 10) * (noise_energy / signal_energy)
        I_DIC *= scale_factor
        I_N *= scale_factor

        return I_N, BW, I, I_DIC, B


    # ---------------------------------------------------------------------
    # (3) "GenerateRandomCellShape.m"
    # ---------------------------------------------------------------------
    def GenerateRandomCellShape(self, PCA_struct, NbrCoeffs, NbrSyntheticCells):
        """
        GenerateRandomCellShape method with corrected x_bar flattening to match MATLAB behavior.
        """
        # Extract PCA components
        b = PCA_struct["b"]  # Shape: (d, N)
        V = PCA_struct["V"]  # Shape: (d, d)
        x_bar = PCA_struct["x_bar"].flatten()  # Ensure x_bar is (d,) instead of (d, 1)
        d, N = b.shape

        # Debugging log
        print(f"PCA_struct dimensions: b={b.shape}, V={V.shape}, x_bar={x_bar.shape}")

        # Initialize output array
        x_syn_data = np.zeros((d, NbrSyntheticCells))

        c = 0
        while c < NbrSyntheticCells:
            # Initialize random coefficients as a 1D array
            b_rand = np.zeros(d)

            # Sample from the distribution of coefficients for the last NbrCoeffs dimensions
            for i in range(d - NbrCoeffs, d):
                kde = gaussian_kde(b[i, :])  # Fit KDE for the i-th dimension
                b_rand[i] = kde.resample(1).item()  # Extract a single value from the KDE

            # Generate synthetic shape: x_syn = x_bar + V @ b_rand
            x_syn = x_bar + V @ b_rand  # Ensure x_bar is 1D

            # Debugging log for x_syn
            print(f"x_syn shape: {x_syn.shape}")

            # Check for crossover and only keep valid shapes
            if not self.CheckCrossOver(x_syn):
                x_syn_data[:, c] = x_syn  # Assign the 1D vector to the correct column
                c += 1

        return x_syn_data




    # ---------------------------------------------------------------------
    # (4) "EmbedCoordToImageSpace.m"
    # ---------------------------------------------------------------------
    def EmbedCoordToImageSpace(self, x, y, Param):
        """
        Enhanced EmbedCoordToImageSpace to include spectral analysis and improved noise generation.
        """
        ImSize = Param["ImSize"]
        RotationAngle = Param["RotationAngle"]
        ScalingRatio = Param["ScalingRatio"]
        Method = Param.get("Method", "fractal")
        NbrFrames = Param.get("NbrFrames", 1)

        theta = np.deg2rad(RotationAngle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy_rot = R @ np.vstack((x, y))

        tck, u = splprep([xy_rot[0, :], xy_rot[1, :]], s=0, per=True)
        u_new = np.linspace(0, 1, 200)
        x_int, y_int = splev(u_new, tck)

        max_width = np.max(np.abs([x_int, y_int]))
        x_int = x_int / max_width * ScalingRatio * (ImSize / 2)
        y_int = y_int / max_width * ScalingRatio * (ImSize / 2)

        x_centered = x_int + (ImSize / 2)
        y_centered = y_int + (ImSize / 2)
        BW = np.zeros((ImSize, ImSize), dtype=np.uint8)
        rr, cc = polygon(y_centered, x_centered, shape=BW.shape)
        BW[rr, cc] = 1

        if Method.lower() == "perlin":
            P_N = self.perlin_noise((ImSize, ImSize), Param.get("Persistence", sqrt(2)))
        elif Method.lower() == "fractal":
            P_N = self.fractal_noise((ImSize, ImSize), Param.get("Persistence", 1.0))
        else:
            P_N = np.zeros((ImSize, ImSize))

        P_N -= np.min(P_N)
        P_N /= np.max(P_N) if np.max(P_N) > 1e-12 else 1

        H = gaussian_filter(BW.astype(float), sigma=3)
        I = H + BW * P_N

        if NbrFrames == 1:
            return I, BW

        return self._create_motion_sequence(I, BW, P_N, Param)

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

        for p in range(half):
            # Reshuffle points to make p-th point the first coordinate
            x_part = np.concatenate((x_temp[p:half], x_temp[:p]))
            y_part = np.concatenate((x_temp[half + p:d], x_temp[half:half + p]))

            # Normalize by first coordinate
            x0, y0 = x_part[0], y_part[0]
            x_part -= x0
            y_part -= y0

            # Rotate such that the first two points are horizontal
            dx, dy = x_part[1] - x_part[0], y_part[1] - y_part[0]
            rot_angle = -np.arctan2(dy, dx)
            cosA, sinA = np.cos(rot_angle), np.sin(rot_angle)
            R = np.array([[cosA, -sinA], [sinA, cosA]])
            xy_rot = R @ np.vstack((x_part, y_part))
            x_part, y_part = xy_rot[0, :], xy_rot[1, :]

            # Check crossovers with subsequent line segments
            coord1 = [x_part[0], y_part[0]]
            coord2 = [x_part[1], y_part[1]]

            for k in range(2, len(x_part) - 1):  # Adjusted range
                coord3 = [x_part[k], y_part[k]]
                coord4 = [x_part[k + 1], y_part[k + 1]]

                # Slopes and intercepts
                denom1 = coord2[0] - coord1[0]
                denom2 = coord4[0] - coord3[0]
                if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
                    continue

                m1 = (coord2[1] - coord1[1]) / denom1
                c1 = coord2[1] - m1 * coord2[0]
                m2 = (coord4[1] - coord3[1]) / denom2
                c2 = coord4[1] - m2 * coord4[0]

                if abs(m1 - m2) < 1e-12:
                    continue  # Parallel lines

                # Intersection point
                x_cross = (c2 - c1) / (m1 - m2)

                # Check if the intersection is within segment bounds
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

        # Explicitly generate grid with the target dimensions
        y, x = np.linspace(-0.5, 0.5, N), np.linspace(-0.5, 0.5, M)
        X, Y = np.meshgrid(x, y)

        # Compute distance matrix
        D = np.sqrt(X**2 + Y**2)

        # Create the 1/f^p filter
        with np.errstate(divide="ignore", invalid="ignore"):
            H = 1.0 / (D**p)
        H[np.isnan(H)] = 0.0  # Replace NaNs with 0.0 to avoid issues
        if np.linalg.norm(H) > 0:
            H /= np.linalg.norm(H)  # Normalize H only if norm > 0

        # Generate random phase and apply the filter
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
    I_N, BW, I, I_DIC, B = simulator.GenerateSyntheticCellSequence()

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
        plt.imshow(I_N, cmap='gray')
        plt.title("I_N")
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
            # plt.contour(BW[:, :, f], [0.5], colors='b')

            plt.subplot(2, 2, 2)
            plt.imshow(I_DIC[:, :, f], cmap='gray')
            plt.title("I_DIC")
            plt.axis('off')
            # plt.contour(BW[:, :, f], [0.5], colors='b')

            plt.subplot(2, 2, 3)
            plt.imshow(I_N[:, :, f], cmap='gray')
            plt.title("I_N")
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
