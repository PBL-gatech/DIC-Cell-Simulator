import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.interpolate import splprep, splev
from skimage.draw import polygon

class SyntheticCellSimulator:
    def __init__(self, params=None):
        # Load default parameters if none provided
        self.params = self.default_params() if params is None else params

        # Placeholder for loaded .mat data
        self.bias_data = None
        self.pca_data = None

    def load_mat_files(self, bias_file, pca_file):
        """
        Load .mat files required for simulation. Includes error handling and validation.
        """
        try:
            self.bias_data = loadmat(bias_file)
            print(f"Successfully loaded bias file: {bias_file}")
        except Exception as e:
            print(f"Error loading bias file: {bias_file} - {e}")

        try:
            self.pca_data = loadmat(pca_file)
            print(f"Successfully loaded PCA file: {pca_file}")
        except Exception as e:
            print(f"Error loading PCA file: {pca_file} - {e}")

        # Validate expected keys for bias_data
        if self.bias_data is not None:
            required_bias_keys = ['Bias','RAPSD','RAFreq']
            for k in required_bias_keys:
                if k not in self.bias_data:
                    raise ValueError(f"Bias file is missing required key: '{k}'")

        # Validate expected keys for pca_data
        if self.pca_data is not None:
            if 'PCA_data' not in self.pca_data:
                raise ValueError("PCA file is missing required key: 'PCA_data'")

    def DIC_EPSF(self, M, shear_angle, epsf_sigma):
        """
        Generate a synthetic DIC kernel (EPSF).
        """
        xg, yg = np.meshgrid(np.arange(1, M + 1), np.arange(1, M + 1))
        xg = xg - np.mean(xg[0, :])
        yg = yg - np.mean(yg[:, 0])
        shear_angle_rad = np.deg2rad(shear_angle)
        exponent = np.exp(-(xg**2 + yg**2) / (epsf_sigma**2))
        EPSF = -xg * exponent * np.cos(shear_angle_rad) - yg * exponent * np.sin(shear_angle_rad)
        return EPSF

    def iRadialAvgPSD(self, PSD):
        """
        Inverse Radial Avg PSD: reconstruct a 2D PSD given a radial average PSD vector.
        """
        spectral_size = (len(PSD[1:])) * 2
        center = np.array([np.mean(np.arange(spectral_size)), np.mean(np.arange(spectral_size))])
        radius_space = np.arange(0, round(0.5 * spectral_size) + 1)
        X, Y = np.meshgrid(np.arange(spectral_size), np.arange(spectral_size))
        F = np.ones((spectral_size, spectral_size)) * PSD[-1]
        for r in range(len(radius_space) - 1, 0, -1):
            radius = radius_space[r]
            XY = ((X - center[1])**2 + (Y - center[0])**2) <= radius**2
            F[XY] = PSD[r]
        F = np.fft.ifftshift(F)
        F[0, 0] = PSD[0]
        return F

    def GenerateSyntheticCellSequence(self):
        param = self.params
        NbrFrames = param["NbrFrames"]

        # Extract PCA data from loaded mat
        # PCA_data is typically structured as pca_data['PCA_data'][0,0]['V'], etc.
        if self.pca_data is None:
            raise ValueError("PCA data is not loaded. Please call load_mat_files first.")
        
        PCA_struct = self.pca_data['PCA_data'][0,0]
        V = PCA_struct['V']
        x_bar = PCA_struct['x_bar']
        
        # Determine number of coefficients
        num_coeffs = V.shape[1]
        # Generate a random cell shape once for all frames, or per frame if desired
        # Here we do a single shape for simplicity
        x_syn_data = self.GenerateRandomCellShape(V, x_bar, num_coeffs, 1)
        # x_syn_data is dimension (N,1). We assume first half is x, second half is y.
        N = x_syn_data.shape[0]
        half = N // 2
        x = x_syn_data[:half,0]
        y = x_syn_data[half:,0]

        # Convert coordinates into an image and BW mask
        I, BW = self.EmbedCoordToImageSpace(x, y, param["PixelspaceParam"])

        # Apply DIC kernel to generate simulated DIC images
        EPSF = self.DIC_EPSF(param["epsf_M"], param["epsf_shear_angle"], param["epsf_sigma"])

        # Create a stack of frames
        # Currently we just replicate the same cell shape. In practice, you might vary it for each frame.
        I_stack = np.stack([I for _ in range(NbrFrames)], axis=-1)
        I_DIC_stack = np.zeros_like(I_stack, dtype=float)
        for f in range(NbrFrames):
            I_DIC_stack[:,:,f] = self.apply_convolution(I_stack[:,:,f], EPSF)

        return I_stack, BW, I_DIC_stack

    def GenerateRandomCellShape(self, V, x_bar, num_coeffs, num_cells):
        # V: shape basis
        # x_bar: mean shape
        # Generate random shape coefficients
        b_rand = np.random.randn(num_coeffs, num_cells)
        x_syn_data = x_bar[:, None] + V @ b_rand
        return x_syn_data

    def EmbedCoordToImageSpace(self, x, y, param):
        ImSize = param["ImSize"]
        RotationAngle = param["RotationAngle"]
        ScalingRatio = param["ScalingRatio"]

        # Rotate and scale coordinates
        rotation_matrix = np.array([[np.cos(np.deg2rad(RotationAngle)), -np.sin(np.deg2rad(RotationAngle))],
                                    [np.sin(np.deg2rad(RotationAngle)),  np.cos(np.deg2rad(RotationAngle))]])
        coords = np.vstack([x, y])
        rotated_coords = rotation_matrix @ coords
        scaled_coords = rotated_coords * ScalingRatio

        # Fit a spline around the cell contour
        # splprep returns (tck, u)
        (tck, u) = splprep([scaled_coords[0,:], scaled_coords[1,:]], s=0, per=True)
        # Evaluate spline at many points to get a smooth polygon
        u_new = np.linspace(0,1,200)
        x_interp, y_interp = splev(u_new, tck)
        
        # Translate coordinates to center image
        x_center = ImSize/2.0
        y_center = ImSize/2.0
        x_pix = x_interp + x_center
        y_pix = y_interp + y_center

        # Create a binary mask by filling polygon
        BW = np.zeros((ImSize, ImSize), dtype=np.uint8)
        rr, cc = polygon(y_pix, x_pix, shape=BW.shape)
        BW[rr, cc] = 1

        # Create an intensity image from BW (e.g. a blurred version of BW)
        I = gaussian_filter(BW.astype(float), sigma=3)

        return I, BW

    def apply_convolution(self, image, kernel):
        # 2D convolution
        return convolve2d(image, kernel, mode="same")

    def default_params(self):
        return {
            "imsize": 64,
            "NbrFrames": 100,
            "epsf_M": 5,
            "epsf_shear_angle": 225,
            "epsf_sigma": 0.5,
            "PixelspaceParam": {
                "ImSize": 64,
                "RotationAngle": np.random.uniform(0, 360),
                "ScalingRatio": np.random.uniform(0.7, 0.8)
            }
        }

if __name__ == "__main__":
    simulator = SyntheticCellSimulator()

    # Set custom parameters if needed
    simulator.params["NbrFrames"] = 10
    simulator.params["PixelspaceParam"]["RotationAngle"] = 45

    # Load required .mat files (update file paths accordingly)
    # Provide accessible resource, for example a GitHub repo:
    # For reference, see: https://github.com/scipy/scipy/blob/main/scipy/io/matlab/_miobase.py
    # This explains how to load MATLAB files and handle structured arrays.

    bias_file = r"C:\Users\sa-forest\Documents\GitHub\DIC-Cell-Simulator\MATLAB\BiasStaticNoiseData64.mat"
    pca_file = r"C:\Users\sa-forest\Documents\GitHub\DIC-Cell-Simulator\MATLAB\Cell_PCA_data.mat"
    simulator.load_mat_files(bias_file, pca_file)

    # Generate synthetic cell sequence
    I, BW, I_DIC = simulator.GenerateSyntheticCellSequence()

    # Display basic information about generated data
    print("Generated Images Shape:", I.shape)
    print("Generated Masks Shape:", BW.shape)
    print("Generated DIC Images Shape:", I_DIC.shape)
