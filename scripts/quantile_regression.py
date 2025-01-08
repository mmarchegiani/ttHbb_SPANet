import os
import pickle
import argparse
import numpy as np
import awkward as ak
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams["font.size"] = 18

class WeightedQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(np.array([self.quantiles_, self.reference_quantiles_]), f)

    def load(self, filename):
        extension = os.path.splitext(filename)[1]
        if not extension == '.pkl':
            raise ValueError(f"Invalid file extension '{os.path.splitext(filename)[1]}'. Only '.pkl' files are supported.")
        self.quantiles_, self.reference_quantiles_ = np.load(filename, allow_pickle=True)

    def _weighted_quantiles(self, X, weights):
        # Calculate weighted quantiles
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        weights_sorted = weights[sorted_indices]
        cum_weights = np.cumsum(weights_sorted) / np.sum(weights_sorted)
        
        # Interpolate to get quantiles
        quantiles = np.interp(np.linspace(0, 1, self.n_quantiles), cum_weights, X_sorted)
        return quantiles
    
    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            raise ValueError("Sample weights must be provided.")
        
        self.quantiles_ = self._weighted_quantiles(X, sample_weight)
        
        if self.output_distribution == 'normal':
            self.reference_quantiles_ = norm.ppf(np.linspace(0, 1, self.n_quantiles))
        elif self.output_distribution == 'uniform':
            self.reference_quantiles_ = np.linspace(0, 1, self.n_quantiles)
        else:
            raise ValueError(f"Unknown output distribution '{self.output_distribution}'.")
        
        return self
    
    def transform(self, X):
        # Interpolate based on weighted quantiles
        transformed_X = np.interp(X, self.quantiles_, self.reference_quantiles_)
        return transformed_X

def plot_score(X, W, transformer, label, output_dir):
    transformed_score = transformer.transform(X)
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    n, bins, patches = ax.hist(X, weights=W, bins=100, histtype='step', label=label)
    n, bins, patches = ax.hist(transformed_score, weights=W, bins=100, histtype='step', label=f"{label} transformed")
    ax.legend()
    ax.set_xlabel("Score")
    ax.set_ylabel("Counts")
    plt.savefig(f"{output_dir}/{label}_score.png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantile regression of the ttHbb SPANet score on ttHbb events.")
    parser.add_argument("-i", "--input", nargs="+", help="Input H5 files", required=True)
    parser.add_argument("-o", "--output", help="Output folder for fitted quantile transformer", required=True)
    parser.add_argument("-n", "--n_quantiles", type=int, default=10000, help="Number of quantiles", required=False)
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for checking uniformity of the ttHbb distribution", required=False)
    parser.add_argument("--atol", type=float, default=0.01, help="Absolute tolerance for checking uniformity of the ttHbb distribution", required=False)
    args = parser.parse_args()
    
    for file in args.input:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file '{file}' does not exist.")
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "quantiles_regressed.pkl")

    # Read all the parquet files and merge them in a single awkward array
    df_list = []
    for file in args.input:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file '{file}' does not exist.")
        df = ak.from_parquet(file)
        df_list.append(df)
    events = ak.concatenate(df_list, axis=0)

    # Here we assume that all the events are ttHbb events
    transformer = WeightedQuantileTransformer(n_quantiles=args.n_quantiles, output_distribution='uniform')
    X = events.spanet_output.tthbb
    W = events.event.weight
    print("Fitting quantile transformer on ttHbb sample...")
    transformer.fit(X, sample_weight=W) # Fit quantile transformer on ttHbb sample only
    transformed_score = transformer.transform(X)
    plot_score(X, W, transformer, "ttHbb", args.output)
    n, bins, patches = plt.hist(transformed_score, weights=W, bins=100, histtype='step', label='ttHbb')
    assert np.allclose(n, np.ones_like(n)*n[0], atol=0.01), "The ttHbb distribution is not uniform. Please double-check the fit of the quantile transformer." # Check that the ttHbb distribution is uniform
    print(f"The ttHbb distribution is uniform after quantile transformation within {args.rtol*100}%.")

    print("Saving the fitted quantiles to", output_file)
    transformer.save(output_file)

    # Check that the fitted quantile transformer can be loaded
    print("Reading the fitted quantiles from", output_file)
    transformer_loaded = WeightedQuantileTransformer(n_quantiles=args.n_quantiles, output_distribution='uniform')
    transformer_loaded.load(output_file)

    transformed_score_loaded = transformer_loaded.transform(X)

    # Check that the loaded quantile transformer works
    assert np.allclose(transformed_score, transformed_score_loaded), "The loaded quantiles do not work. Please double-check the save/load process."
    print("The saved quantiles can be loaded and work correctly.")

