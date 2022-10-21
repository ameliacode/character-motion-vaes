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

current_dir = str(Path(__file__).resolve().parents[0])
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

def main():
    env_path=os.path.joint(parent_dir, "environments")
    args = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(env_path),
        save_dir=str(current_dir), # where all states and gmm
        train_states=None, # npy file with pre-loaded train states
        visualize_results=False, # MAYBE LATER
        test_results=False, # MAYBE LATER
        gmm_comps=12
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_states_out_path = os.path.join(args.save_dir, "train_states.npy")
    gmm_out_path = os.path.join(args.save_dir, "prior_gmm.npz") # for fitting.py, gmm prior over initial state of the sequence

    all_states = None

    if args.train_states is not None:
        pass # MAYBE LATER
    else:
        mocap_file = 0

        # load data and input all state variable

    start_t = time.time()

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
                          verbsose=1,
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

    print("GMM time: %f s" % (time.time() - start_t))

    # print("Running evaluation on test set ...")
    # test_results(args.data, gmm_out_path)
    # print("Visualizing sampled results")
    # vis_gmm_fit_results(gmm_out_path, debug_gmm_obj=gmm, debug_data=all_states)


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

def viz_gmm_fit_results(gmm_path, debug_gmm_obj=None, debug_data=None):
    pass

def test_results(data_path, gmm_path): # Evaluate likelihood of test data

    # load in GMM result
    gmm_weights, gmm_means, gmm_covs = load_gmm_results(gmm_path)

    # build pytorch distrib
    gmm_distrib = build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs)

    # MAYBE LATER


if __name__=="__main__":
    main()