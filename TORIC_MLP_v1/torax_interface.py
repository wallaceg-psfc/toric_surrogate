"""TORAX source model interface for the ToricNN ion-cyclotron heating model.

To use this source model in TORAX, register it with TORAX's source registry
before building your `ToraxConfig`:

```python
from torax import sources as torax_sources
from toric_surrogate.TORIC_MLP_v1 import torax_interface

torax_sources.register_source_model_config(
    torax_interface.ToricNNIonCyclotronSourceConfig,
    'icrh',
)
```
"""

import dataclasses
import functools
import logging
from typing import Annotated, Literal, Self

import chex
import jax
from jax import numpy as jnp
import pydantic
import torax
from torax import sources as torax_sources
from torax._src import array_typing
from torax._src import math_utils
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import cell_variable
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.physics import fast_ion_utils
from torax._src.sources.ion_cyclotron_source import base as icrh_base
from torax._src.torax_pydantic import torax_pydantic
from toric_surrogate.TORIC_MLP_v1 import toric_nn as toric_nn_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax_sources.RuntimeParams):
  """Runtime parameters for the ToricNN ICRH source model."""

  frequency: array_typing.FloatScalar
  minority_concentration: array_typing.FloatScalar | None
  P_total: array_typing.FloatScalar
  absorption_fraction: array_typing.FloatScalar
  wall_inner: float
  wall_outer: float
  minority_species: str | None = dataclasses.field(
      default=None, metadata={'static': True}
  )


def _get_minority_concentration_from_composition(
    plasma_composition: plasma_composition_lib.RuntimeParams,
    core_profiles: torax.CoreProfiles,
    minority_species: str,
) -> jax.Array:
  """Extracts minority species concentration from core profiles.

  Args:
    plasma_composition: Runtime parameters of plasma composition.
    core_profiles: Core plasma profiles.
    minority_species: Symbol of the minority species (e.g., 'He3', 'D', 'T').

  Returns:
    Minority species fractional concentration relative to electron density.

  Raises:
    ValueError: If minority_species is not found in plasma composition.
  """
  if minority_species in plasma_composition.main_ion_names:
    fraction = core_profiles.main_ion_fractions[minority_species]
    return core_profiles.n_i.value * fraction / core_profiles.n_e.value  # pyrefly: ignore[bad-return]

  if minority_species in plasma_composition.impurity_names:
    impurity_fractions = core_profiles.impurity_fractions
    impurity_density_scaling = core_profiles.impurity_density_scaling
    fraction = impurity_fractions[minority_species]
    n_imp_species = (
        fraction * core_profiles.n_impurity.value * impurity_density_scaling
    )
    return n_imp_species / core_profiles.n_e.value  # pyrefly: ignore[bad-return]

  raise ValueError(
      f'Minority species {minority_species} not found in plasma composition.'
      f' Available main ions: {plasma_composition.main_ion_names},'
      f' impurities: {plasma_composition.impurity_names}'
  )


def icrh_model_func(
    runtime_params: torax.RuntimeParams,
    geo: torax.Geometry,
    source_name: str,
    core_profiles: torax.CoreProfiles,
    unused_calculated_source_profiles: torax_sources.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
    toric_nn: toric_nn_lib.ToricNNWrapper,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    tuple[fast_ion_lib.FastIon, ...],
]:
  """Computes ion/electron heat source terms and fast ions using ToricNN."""
  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, RuntimeParams)

  # Get minority concentration: either from plasma composition or from params.
  if source_params.minority_species is not None:
    minority_concentration_profile = (
        _get_minority_concentration_from_composition(
            runtime_params.plasma_composition,
            core_profiles,
            source_params.minority_species,
        )
    )
    # Use first cell point value as assumption for relevant locations for
    # extracting scalar ToricNN input, since deposition tends to be near-axis.
    minority_concentration_scalar = minority_concentration_profile[0]
  else:
    # Use legacy parameter (backward compatibility).
    # TODO(b/434175938): Remove backward compatibility in V2.
    minority_concentration_scalar = source_params.minority_concentration
    minority_concentration_profile = source_params.minority_concentration

  # Construct inputs for ToricNN.
  volume_average_temperature = math_utils.volume_average(
      core_profiles.T_e.value, geo  # pyrefly: ignore[bad-argument-type]
  )
  volume_average_density = math_utils.volume_average(
      core_profiles.n_e.value, geo  # pyrefly: ignore[bad-argument-type]
  )

  # Peaking factors are core w.r.t volume averages.
  temperature_peaking_factor = (
      core_profiles.T_e.value[0] / volume_average_temperature  # pyrefly: ignore[bad-index]
  )
  density_peaking_factor = core_profiles.n_e.value[0] / volume_average_density  # pyrefly: ignore[bad-index]
  Router = geo.R_out_face[-1]  # Use LCFS outboard radius.
  Rinner = geo.R_in_face[-1]  # Use LCFS inboard radius.
  # Assumption: inner and outer gaps are not functions of z0.
  # This is a good assumption for the inner gap but perhaps less good for the
  # outer gap where there is significant curvature to the outer limiter.
  gap_inner = Rinner - source_params.wall_inner
  gap_outer = source_params.wall_outer - Router
  toric_inputs = toric_nn_lib.ToricNNInputs(
      frequency=source_params.frequency,
      volume_average_temperature=volume_average_temperature,
      volume_average_density=volume_average_density
      / 1e20,  # convert to 10^20 m^-3
      minority_concentration=minority_concentration_scalar  # pyrefly: ignore[unsupported-operation]
      * 100,  # Convert to percentage.
      gap_inner=gap_inner,
      gap_outer=gap_outer,
      z0=geo.z_magnetic_axis(),  # pyrefly: ignore[bad-argument-type]
      temperature_peaking_factor=temperature_peaking_factor,
      density_peaking_factor=density_peaking_factor,
      B_0=geo.B_0,
  )

  toric_nn_outputs = toric_nn_lib.predict(toric_nn, toric_inputs)
  toric_grid = jnp.linspace(0.0, 1.0, toric_nn_lib.TORIC_GRID_SIZE)

  # Ideally total ICRH power should equal one, but normalise if not.
  power_deposition_he3 = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_He3,
  )
  power_deposition_e = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_e,
  )
  power_deposition_2T = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_2T,
  )
  power_deposition_all = (
      power_deposition_2T + power_deposition_e + power_deposition_he3
  )

  total_power_deposition = math_utils.volume_integration(
      power_deposition_all, geo
  )
  power_deposition_he3 /= total_power_deposition
  power_deposition_e /= total_power_deposition
  power_deposition_2T /= total_power_deposition

  # Computing fast ion density and temperatures for He3.
  he3_atomic_mass = 3.016
  he3_charge_number = 2

  n_tail, T_tail = fast_ion_utils.bimaxwellian_split(
      power_deposition=power_deposition_he3,
      T_e=core_profiles.T_e.value,  # pyrefly: ignore[bad-argument-type]
      n_e=core_profiles.n_e.value,  # pyrefly: ignore[bad-argument-type]
      T_i=core_profiles.T_i.value,  # pyrefly: ignore[bad-argument-type]
      n_i=core_profiles.n_i.value,  # pyrefly: ignore[bad-argument-type]
      minority_concentration=minority_concentration_profile,  # pyrefly: ignore[bad-argument-type]
      P_total_W=source_params.P_total,  # pyrefly: ignore[bad-argument-type]
      charge_number=he3_charge_number,
      mass_number=he3_atomic_mass,
      bulk_ion_mass=core_profiles.A_i,  # pyrefly: ignore[bad-argument-type]
      Z_i=core_profiles.Z_i,  # pyrefly: ignore[bad-argument-type]
      n_impurity=core_profiles.n_impurity.value,  # pyrefly: ignore[bad-argument-type]
      Z_impurity=core_profiles.Z_impurity,  # pyrefly: ignore[bad-argument-type]
      A_impurity=core_profiles.A_impurity,  # pyrefly: ignore[bad-argument-type]
  )

  # Build fast ion output for all supported species.
  # Only He3 has non-zero values from the model; others are zeroed.
  he3_fast_ion = fast_ion_lib.FastIon(
      species='He3',
      source=source_name,
      n=cell_variable.CellVariable(
          value=n_tail,
          face_centers=geo.rho_face_norm,
          right_face_grad_constraint=None,
          right_face_constraint=jnp.zeros(()),
      ),
      T=cell_variable.CellVariable(
          value=T_tail,
          face_centers=geo.rho_face_norm,
          right_face_grad_constraint=None,
          right_face_constraint=core_profiles.T_i.right_face_constraint,
      ),
  )
  fast_ions = icrh_base.build_fast_ions(
      source_name=source_name,
      geo=geo,
      fast_ions=[he3_fast_ion],
  )

  frac_ion_heating = collisions.fast_ion_fractional_heating_formula(
      T_tail,
      core_profiles.T_e.value,  # pyrefly: ignore[bad-argument-type]
      he3_atomic_mass,
  )
  absorbed_power = source_params.P_total * source_params.absorption_fraction
  source_ion = power_deposition_he3 * frac_ion_heating * absorbed_power
  source_el = power_deposition_he3 * (1 - frac_ion_heating) * absorbed_power

  # Assume that all the power from the electron power profile goes to electrons.
  source_el += power_deposition_e * absorbed_power

  # Assume that all the power from the tritium power profile goes to ions.
  source_ion += power_deposition_2T * absorbed_power

  return (source_ion, source_el, fast_ions)


# Cache the result of this function to avoid re-creating the partial function
# every time it is called and ensure we hit the same JAX compile cache (as
# model_func is part of the key).
# maxsize=1 is sufficient as the ToricNNWrapper only changes if a new path
# is provided. This is not expected to happen very often.
@functools.lru_cache(maxsize=1)
def _icrh_model_func_with_toric_nn(
    model_path: str | None,
) -> torax_sources.SourceProfileFunction:
  """Returns a function that computes the ICRH source terms given a ToricNN."""
  toric_nn = toric_nn_lib.ToricNNWrapper(model_path)
  return functools.partial(
      icrh_model_func,
      toric_nn=toric_nn,
  )


class ToricNNIonCyclotronSourceConfig(icrh_base.IonCyclotronSourceConfig):
  """Configuration for the IonCyclotronSource using the ToricNN model.

  Attributes:
    model_name: Discriminator literal identifying the ToricNN model.
    model_path: Path to JSON weights and model config of ToricNN model.
    wall_inner: Inner radial location of first wall at plasma midplane level
      [m].
    wall_outer: Outer radial location of first wall at plasma midplane level
      [m].
    frequency: ICRF wave frequency [Hz].
    minority_concentration: He3 minority fractional concentration relative to
      the electron density. This parameter is used when minority_species is not
      specified (legacy mode). When minority_species is set, the concentration
      is automatically extracted from plasma_composition.
  """

  model_name: Annotated[Literal['toric_nn'], torax_pydantic.JAX_STATIC] = (
      'toric_nn'
  )
  model_path: Annotated[str | None, torax_pydantic.JAX_STATIC] = None
  wall_inner: torax_pydantic.Meter = 1.24
  wall_outer: torax_pydantic.Meter = 2.43
  frequency: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  # TODO(b/434175938): Remove in V2. Will be set in plasma_composition.
  minority_concentration: torax_pydantic.TimeVaryingScalar | None = (
      torax_pydantic.ValidatedDefault(0.03)
  )

  @property
  def model_func(self) -> torax_sources.SourceProfileFunction:
    return _icrh_model_func_with_toric_nn(self.model_path)

  def build_source(self) -> icrh_base.IonCyclotronSource:
    """Builds the source, only loading the ToricNN model in MODEL_BASED mode.

    When the mode is PRESCRIBED or ZERO, the model function is never called,
    so we avoid loading the ToricNN model JSON file. This allows users who
    don't have access to the toric_nn model file to still use the ICRH source
    in prescribed mode.

    Returns:
      An IonCyclotronSource object.
    """
    if self.mode == torax_sources.Mode.MODEL_BASED:
      return icrh_base.IonCyclotronSource(model_func=self.model_func)
    return icrh_base.IonCyclotronSource(model_func=None)

  @pydantic.model_validator(mode='after')
  def _validate_minority_species(self) -> Self:
    if self.minority_species is not None and self.minority_species != 'He3':
      raise ValueError(
          "Minority species must be 'He3' if specified. Got:"
          f' {self.minority_species}. '
          'The current ToricNN model supports only He3.'
      )
    return self

  @pydantic.model_validator(mode='after')
  def _log_warning_for_used_minority_concentration(
      self,
  ) -> Self:
    """Logs a warning if minority_concentration is provided."""
    if self.minority_concentration is not None:
      logging.warning(
          'minority_concentration is provided in ion_cyclotron_source. '
          'This is a deprecated parameter. It is recommended to explicitly set '
          'minority_species to None and then define the minority density in '
          'plasma_composition instead. Note that defining '
          'non None minority_concentration in ion_cyclotron_source overrides '
          'any plasma_composition values for ICRH model input.'
      )
    return self

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        wall_inner=self.wall_inner,
        wall_outer=self.wall_outer,
        frequency=self.frequency.get_value(t),
        minority_concentration=(
            self.minority_concentration.get_value(t)
            if self.minority_concentration is not None
            else None
        ),
        minority_species=self.minority_species,
        P_total=self.P_total.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
    )
