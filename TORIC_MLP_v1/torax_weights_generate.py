"""Extract parameters from sklearn network into TORAX compatible JSON."""
import numpy as np
import json
import joblib
from collections import defaultdict
from typing import Any
import os

_OUTPUT_FILE = 'toricnn.json'


def trie():
    return defaultdict(trie)


def model_to_params(scaler, mlp, pca,) -> dict[str, Any]:
    """Extract SKLearn model params and extract them.

    TORAX Flax model requires float32 weights so also do any casting.
    The keys in the `params` dictionary are those expected by TORAX.
    """
    params = trie()
    # Extract Scaler.
    params['scaler_mean'] = scaler.mean_.astype(np.float32).tolist()
    params['scaler_scale'] = scaler.scale_.astype(np.float32).tolist()
    # Extract MLP.
    for i, (coef, intercept) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
      params[f'Dense_{i}']['kernel'] = coef.astype(np.float32).tolist()
      params[f'Dense_{i}']['bias'] = intercept.astype(np.float32).tolist()
    # Extract PCA.
    params['pca_components'] = pca.components_.astype(np.float32).tolist()
    params['pca_mean'] = pca.mean_.astype(np.float32).tolist()
    return params


def extract_model_weights_for_torax(scaler, *, mlp_he3, mlp_t, mlp_e, pca_he3,
    pca_t, pca_e, input_dim=10) -> dict[str, Any]:
    params = {
        'input_dim': input_dim,
        'radial_nodes': pca_he3.components_.shape[1],
        'pca_coeffs': pca_he3.components_.shape[0],
        'hidden_sizes': list(mlp_he3.intercepts_[0].shape),
        'He3': model_to_params(scaler, mlp_he3, pca_he3),
        '2T': model_to_params(scaler, mlp_t, pca_t),
        'e': model_to_params(scaler, mlp_e, pca_e),
    }
    return params


if __name__ == '__main__':
    cwd = os.getcwd()
    scaler_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_scaler.lzma'))
    mlp_he3_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_PwHe3.lzma'))
    mlp_t_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_Pw2T.lzma'))
    mlp_e_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_PwE.lzma'))
    pca_he3_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_PCA_PwHe3.lzma'))
    pca_t_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_PCA_Pw2T.lzma'))
    pca_e_sklearn = joblib.load(os.path.join(cwd, 'TORIC_MLP_PCA_PwE.lzma'))
    params_torax = extract_model_weights_for_torax(
        scaler_sklearn,
        mlp_he3=mlp_he3_sklearn,
        mlp_t=mlp_t_sklearn,
        mlp_e=mlp_e_sklearn,
        pca_he3=pca_he3_sklearn,
        pca_t=pca_t_sklearn,
        pca_e=pca_e_sklearn,
    )
    with open(os.path.join(cwd, _OUTPUT_FILE), 'w+') as fp:
        json.dump(params_torax, fp, indent=4)
