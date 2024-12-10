import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

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

        # Validate expected keys
        if self.bias_data and not all(key in self.bias_data for key in ['Bias', 'RAPSD', 'RAFreq']):
            raise ValueError("Bias file is missing required keys: 'Bias', 'RAPSD', 'RAFreq'")
        if self.pca_data and 'PCA_data' not in self.pca_data:
            raise ValueError("PCA file is missing required key: 'PCA_data'")

    def DIC_EPSF(self, M, shear_angle, epsf_sigma):
        xg, yg = np.meshgrid(np.arange(1, M + 1), np.arange(1, M + 1))
        xg -= np.mean(xg[0, :])
        yg -= np.mean(yg[:, 0])
        shear_angle_rad = np.deg2rad(shear_angle)
        exponent = np.exp(-(xg**2 + yg**2) / epsf_sigma**2)
        return -xg * exponent * np.cos(shear_angle_rad) - yg * exponent * np.sin(shear_angle_rad)

    def iRadialAvgPSD(self, PSD):
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

        # Step 1: Generate Random Cell
        x_syn_data = self.GenerateRandomCellShape(self.pca_data, len(self.pca_data['PCA_data']['b']), 1)

        # Step 2: Generate Cell Surface
        x, y = np.split(x_syn_data, 2)
        I, BW = self.EmbedCoordToImageSpace(x, y, param["PixelspaceParam"])

        # Step 3: Apply DIC Kernel
        EPSF = self.DIC_EPSF(param["epsf_M"], param["epsf_shear_angle"], param["epsf_sigma"])
        I_DIC = np.stack([self.apply_convolution(I[:, :, f], EPSF) for f in range(param["NbrFrames"])])

        return I, BW, I_DIC

    def GenerateRandomCellShape(self, pca_data, num_coeffs, num_cells):
        V = pca_data['PCA_data']['V']
        x_bar = pca_data['PCA_data']['x_bar']
        b_rand = np.random.randn(V.shape[1], num_cells)
        x_syn_data = x_bar[:, None] + V @ b_rand
        return x_syn_data

    def EmbedCoordToImageSpace(self, x, y, param):
        ImSize = param["ImSize"]
        RotationAngle = param["RotationAngle"]
        ScalingRatio = param["ScalingRatio"]

        rotation_matrix = np.array([[np.cos(np.deg2rad(RotationAngle)), -np.sin(np.deg2rad(RotationAngle))],
                                     [np.sin(np.deg2rad(RotationAngle)),  np.cos(np.deg2rad(RotationAngle))]])
        coords = np.vstack([x, y])
        rotated_coords = rotation_matrix @ coords
        scaled_coords = rotated_coords * ScalingRatio

        x_interp, y_interp = splev(splprep([scaled_coords[0], scaled_coords[1]])[0])
        BW = np.zeros((ImSize, ImSize), dtype=np.uint8)
        BW[np.round(x_interp).astype(int), np.round(y_interp).astype(int)] = 1
        return BW, gaussian_filter(BW, sigma=3)

    def apply_convolution(self, image, kernel):
        return np.convolve(image, kernel, mode="same")

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
    simulator.params["NbrFrames"] = 100
    simulator.params["PixelspaceParam"]["RotationAngle"] = 45

    # Load required .mat files (update file paths accordingly)
    simulator.load_mat_files("/mnt/data/BiasStaticNoiseData64.mat", "/mnt/data/Cell_PCA_data.mat")

    # Generate synthetic cell sequence
    I, BW, I_DIC = simulator.GenerateSyntheticCellSequence()

    # Display basic information about generated data
    print("Generated Images Shape:", I.shape)
    print("Generated Masks Shape:", BW.shape)
    print("Generated DIC Images Shape:", I_DIC.shape)
