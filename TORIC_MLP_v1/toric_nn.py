"""ToricNN JAX/Flax surrogate model for ion-cyclotron resonance heating (ICRH)."""

from collections.abc import Sequence
import dataclasses
import json
import logging
import os
import pathlib
from typing import Any, Final, Self

import flax.linen as nn
import jax
from jax import numpy as jnp
import jaxtyping as jt
import numpy as np

# Default path to the bundled TORIC MLP v1 weights and configuration JSON.
DEFAULT_MODEL_PATH: Final[str] = str(
    pathlib.Path(__file__).parent / 'toricnn.json'
)
TORIC_GRID_SIZE: Final[int] = 297
HELIUM3_ID: Final[str] = 'He3'
TRITIUM_SECOND_HARMONIC_ID: Final[str] = '2T'
ELECTRON_ID: Final[str] = 'e'


def _from_json(json_file: str | os.PathLike[str]) -> dict[str, Any]:
  """Loads the model config and weights from a JSON file."""
  expanded_path = pathlib.Path(json_file).expanduser()
  with open(expanded_path, 'r', encoding='utf-8') as file_:
    return json.load(file_)


# pylint: disable=invalid-name
# Many of the variables below are named to match the physics quantities
# as defined by the TORIC ICRF solver, so we keep their naming for consistency.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ToricNNInputs:
  """Inputs to the ToricNN model."""

  # ICRF wave frequency in MHz, training range = 119 to 121.
  frequency: jt.Float[jt.Scalar, ''] | float
  # Volume average temperature in keV, training range = 1.5 to 8.5.
  volume_average_temperature: jt.Float[jt.Scalar, ''] | float
  # Volume average density in 10^20 m^-3, training range = 1.1 to 5.1.
  volume_average_density: jt.Float[jt.Scalar, ''] | float
  # He3 minority concentration relative to the electron density in %,
  # training range = 1 to 5.
  minority_concentration: jt.Float[jt.Scalar, ''] | float
  # Distance from last closed flux surface (LCFS) to the inner wall in m,
  # training range = 0 to 0.03.
  gap_inner: jt.Float[jt.Scalar, ''] | float
  # Distance from LCFS to the outer midplane limiter in m,
  # training range = 0 to 0.05.
  gap_outer: jt.Float[jt.Scalar, ''] | float
  # Vertical position of magnetic axis in m, training range = -0.05 to 0.05.
  z0: jt.Float[jt.Scalar, ''] | float
  # Temperature profile peaking factor, training range = 2 to 3.
  temperature_peaking_factor: jt.Float[jt.Scalar, ''] | float
  # Density profile peaking factor, training range = 1.15 to 1.65.
  density_peaking_factor: jt.Float[jt.Scalar, ''] | float
  # Vacuum toroidal magnetic field at R_major [T].
  # Training range = 11.8 to 12.5.
  B_0: jt.Float[jt.Scalar, ''] | float


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ToricNNOutputs:
  """Outputs from the ToricNN model."""

  # Power deposition on helium-3 in MW/m^3/MW_{abs}.
  power_deposition_He3: jt.Float[jt.Array, '_']
  # Power deposition on tritium (second harmonic) in MW/m^3/MW_{abs}.
  power_deposition_2T: jt.Float[jt.Array, '_']
  # Power deposition on electrons in MW/m^3/MW_{abs}.
  power_deposition_e: jt.Float[jt.Array, '_']


class ToricNN(nn.Module):
  """Surrogate heating model trained on TORIC ICRF solver simulation.

  This model takes input parameters from the `ToricNNInputs` class and outputs
  power deposition profiles for helium-3, tritium (second harmonic) and
  electrons on a radial grid.

  This Flax module is not intended to be used directly but rather through the
  `ToricNNWrapper` class.

  The modelling approach is described in:
  https://iopscience.iop.org/article/10.1088/1741-4326/ad645d/pdf. The model
  is trained on regression outputs from the TORIC ICRF solver. PCA is applied
  to the outputs of the solver to reduce the dimensionality of the model.

  The structure of the model consists of:
  - Scaling and normalisation of the input parameters.
  - An MLP transforming the scaled inputs.
  - A projection back to true values using the PCA coefficients.
  """

  # Hidden layer sizes for the MLP.
  hidden_sizes: Sequence[int]
  # Number of PCA coefficients used by ToricNN.
  pca_coeffs: int
  # Input dimensionality of the ToricNN model.
  input_dim: int
  # Number of radial nodes in output of the ToricNN model.
  radial_nodes: int

  def setup(self):
    """Setup the parameters of the ToricNN model."""
    self.scaler_mean = self.param(
        'scaler_mean',
        jax.random.normal,
        (self.input_dim,),
    )
    self.scaler_scale = self.param(
        'scaler_scale',
        jax.random.normal,
        (self.input_dim,),
    )
    self.pca_components = self.param(
        'pca_components',
        jax.random.normal,
        (
            self.pca_coeffs,
            self.radial_nodes,
        ),
    )
    self.pca_mean = self.param(
        'pca_mean',
        jax.random.normal,
        (self.radial_nodes,),
    )

  @nn.compact
  def __call__(
      self,
      x: jt.Float[jt.Array, 'B* {self.input_dim}'],
  ) -> jt.Float[jt.Array, 'B* {self.radial_nodes}']:
    """Run a forward pass of the ToricNN model."""
    # Scale and normalise inputs.
    x = (x - self.scaler_mean) / self.scaler_scale

    # MLP.
    for hidden_size in self.hidden_sizes:
      x = nn.Dense(
          hidden_size,
      )(x)
      x = nn.relu(x)
    x = nn.Dense(
        self.pca_coeffs,
    )(x)

    x = x @ self.pca_components + self.pca_mean  # Project back to true values.
    x = x * (x > 0)  # Eliminate non-physical values for power deposition.
    return x


class ToricNNWrapper:
  """Wrapper for the Toric NN model.

  This wrapper is the main interface for interacting with the Toric NN model
  for making predictions of heating power deposition profiles given
  `ToricNNInputs`.

  The wrapper constructs 3 separate instances of the `ToricNN` class, one for
  each simulated output (Helium-3, 2nd-harmonic tritium and electrons).
  """

  def __init__(self, path: str | os.PathLike[str] = DEFAULT_MODEL_PATH):
    self._path = str(path)
    logging.info('Loading ToricNN model from %s', self._path)
    self.model_config = _from_json(self._path)

    self.power_deposition_network = self._load_network()
    self.power_deposition_He3_params = self._load_params(HELIUM3_ID)
    self.power_deposition_2T_params = self._load_params(
        TRITIUM_SECOND_HARMONIC_ID
    )
    self.power_deposition_e_params = self._load_params(ELECTRON_ID)
    logging.info('Loaded ToricNN model from %s', self._path)

  def _load_network(self) -> ToricNN:
    return ToricNN(
        hidden_sizes=self.model_config['hidden_sizes'],
        pca_coeffs=self.model_config['pca_coeffs'],
        input_dim=self.model_config['input_dim'],
        radial_nodes=self.model_config['radial_nodes'],
    )

  def _load_params(self, network_name: str) -> dict[str, Any]:
    """Load a ToricNN network and its parameters."""
    params = {}
    params['params'] = dict(self.model_config[f'{network_name}'])
    for i in range(len(self.model_config['hidden_sizes']) + 1):
      params['params'][f'Dense_{i}'] = {
          'kernel': np.asarray(
              self.model_config[f'{network_name}'][f'Dense_{i}']['kernel']
          ),
          'bias': np.asarray(
              self.model_config[f'{network_name}'][f'Dense_{i}']['bias']
          ),
      }
    params['params']['pca_components'] = np.asarray(
        self.model_config[f'{network_name}']['pca_components']
    )
    params['params']['pca_mean'] = np.asarray(
        self.model_config[f'{network_name}']['pca_mean']
    )
    params['params']['scaler_mean'] = np.asarray(
        self.model_config[f'{network_name}']['scaler_mean']
    )
    params['params']['scaler_scale'] = np.asarray(
        self.model_config[f'{network_name}']['scaler_scale']
    )
    return params

  def __hash__(self) -> int:
    return hash(self._path)

  def __eq__(self, other: Self) -> bool:
    return isinstance(other, ToricNNWrapper) and self._path == other._path


def _get_dtype() -> Any:
  precision = os.getenv('JAX_PRECISION', 'f64')
  return jnp.float64 if precision == 'f64' else jnp.float32


@jax.jit(static_argnames='toric_nn')
def predict(
    toric_nn: ToricNNWrapper,
    inputs: ToricNNInputs,
) -> ToricNNOutputs:
  """Makes a prediction of power deposition profiles given the inputs."""
  input_array = jnp.asarray(
      [
          inputs.frequency,
          inputs.volume_average_temperature,
          inputs.volume_average_density,
          inputs.minority_concentration,
          inputs.gap_inner,
          inputs.gap_outer,
          inputs.z0,
          inputs.temperature_peaking_factor,
          inputs.density_peaking_factor,
          inputs.B_0,
      ],
      dtype=_get_dtype(),
  )
  outputs_He3 = toric_nn.power_deposition_network.apply(
      toric_nn.power_deposition_He3_params, input_array
  )
  outputs_2T = toric_nn.power_deposition_network.apply(
      toric_nn.power_deposition_2T_params, input_array
  )
  outputs_e = toric_nn.power_deposition_network.apply(
      toric_nn.power_deposition_e_params, input_array
  )
  return ToricNNOutputs(
      power_deposition_He3=outputs_He3,
      power_deposition_2T=outputs_2T,
      power_deposition_e=outputs_e,
  )

