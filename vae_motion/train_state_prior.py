import os, sys
import argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import MixtureSameFamily, MultivariateNormal, Independent, Categorical, Normal

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from types import SimpleNamespace
from pathlib import Path
from matplotlib.pyplot import *

current_dir = str(Path(__file__).resolve().parents[0])
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(current_dir)

def main():
    env_path=os.path.join(str(parent_dir), "environments")
    pfnn_path = os.path.join(env_path, "PFNN_data")
    args = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(pfnn_path, "PFNN_mocap.npz"),
        save_dir=str(current_dir), # where all states and gmm
        train_states=None, # npy file with pre-loaded train states
        visualize_results=False, # MAYBE LATER
        test_results=False, # MAYBE LATER
        gmm_comps=12
    )

    gmm_out_path = os.path.join(args.save_dir, "prior_gmm.npz") # for fitting.py, gmm prior over initial state of the sequence

    raw_data = np.load(args.mocap_file)
    mocap_data = torch.from_numpy(raw_data["data"]).float().to(args.device)
    all_states = mocap_data

    print("Fitting GMM with %d components..." %(args.gmm_comps))
    gmm = GaussianMixture(n_components=args.gmm_comps,
                          covariance_type="full",
                          tol=0.001,
                          reg_covar=1e-06,
                          max_iter=200,
                          n_init=1,
                          init_params="kmeans",
                          weights_init=None,
                          means_init=None,
                          precisions_init=None,
                          random_state=0,
                          warm_start=False,
                          verbose=1,
                          verbose_interval=5
                          )

    gmm.fit(all_states)
    # print(gmm.weights_, gmm.means_, gmm.covariances_)
    # print(gmm.converged_)
    # print(gmm.weights_.shape)
    # print(gmm.means_.shape)
    # print(gmm.covariances_.shape)

    # save distribution information
    np.savez(gmm_out_path, weights=gmm.weights_, means=gmm.means_, covariances=gmm.covariances_)


def load_gmm_results(gmm_path):
    gmm_result = np.load(gmm_path)
    gmm_weights = gmm_result["weights"]
    gmm_means = gmm_result["means"]
    gmm_covs = gmm_result["covariances"]
    return gmm_weights, gmm_means, gmm_covs

def build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs):
    mix = Categorical(torch.from_numpy(gmm_weights))
    comp = MultivariateNormal(torch.from_numpy(gmm_means), covariance_matrix=torch.from_numpy(gmm_covs))
    gmm_distrib = MixtureSameFamily(mix, comp)
    return gmm_distrib

def visualize_results(data_path, gmm_path): # Evaluate likelihood of test data
    gmm_weights, gmm_means, gmm_covs = load_gmm_results(gmm_path)
    gmm_distrib = build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs)

    num_samples = 100
    sample_states = gmm_distrib.sample(torch.Size([num_samples]))

    num_samples = gmm_means.shape[0]
    sample_states = torch.from_numpy(gmm_means)
    torch_logprob = gmm_distrib.logprob(sample_states)


if __name__=="__main__":
    main()