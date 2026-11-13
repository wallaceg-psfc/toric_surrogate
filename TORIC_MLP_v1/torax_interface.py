"""TORAX source model interface for the ToricNN ion-cyclotron heating model.

To use this source model in TORAX, register it with TORAX's source registry
before building your `ToraxConfig`:

```python
from torax import sources as torax_sources
from toric_surrogate.TORIC_MLP_v1 import torax_interface

torax_sources.register_source_config(
    torax_interface.ToricNNIonCyclotronSourceConfig
)
```
"""

import dataclasses
import functools
from typing import Annotated, ClassVar, Literal

import chex
import jax
from jax import numpy as jnp
import pydantic
import torax
from torax import math_utils
from torax import sources as torax_sources
from torax import torax_pydantic
from torax.experimental import fast_ion as fast_ion_lib
from toric_surrogate.TORIC_MLP_v1 import fast_ion_utils
from toric_surrogate.TORIC_MLP_v1 import toric_nn as toric_nn_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax_sources.RuntimeParams):
  """Runtime parameters for the ToricNN ICRH source model."""

  P_total: jax.Array
  absorption_fraction: jax.Array
  frequency: jax.Array
  wall_inner: float
  wall_outer: float
  minority_species: str = jax.tree.static()


# Cache the result of this function to avoid reloading the ToricNN weights
# every time the source is evaluated.
# maxsize=1 is sufficient as the ToricNNWrapper only changes if a new path
# is provided. This is not expected to happen very often.
@functools.lru_cache(maxsize=1)
def _load_toric_nn(
    model_path: str = toric_nn_lib.DEFAULT_MODEL_PATH,
) -> toric_nn_lib.ToricNNWrapper:
  """Returns the (cached) ToricNN model loaded from `model_path`."""
  return toric_nn_lib.ToricNNWrapper(model_path)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ToricNNIonCyclotronSource(torax_sources.Source):
  """Ion cyclotron source using the ToricNN surrogate model."""

  SOURCE_ID: ClassVar[str] = 'icrh'
  POPULATED_SOURCE_OUTPUTS: ClassVar[
      tuple[torax_sources.PopulatedSourceOutput, ...]
  ] = (
      torax_sources.PopulatedSourceOutput.POWER_DENSITY_I,
      torax_sources.PopulatedSourceOutput.POWER_DENSITY_E,
  )
  PRODUCES_FAST_IONS: ClassVar[bool] = True
  model_path: str = toric_nn_lib.DEFAULT_MODEL_PATH

  def _get_model_value(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      solver_profiles: torax.SolverProfiles,
      source_state: torax_sources.SourceState,
  ) -> tuple[torax_sources.SourceOutput, torax_sources.SourceState]:
    """Computes ion/electron heat source terms and fast ions using ToricNN."""
    toric_nn = _load_toric_nn(self.model_path)
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)

    # Extract minority concentration from plasma composition.
    minority_concentration_profile = solver_profiles.get_species_concentration(
        source_params.minority_species
    )
    # Use first cell point value as assumption for relevant locations for
    # extracting scalar ToricNN input, since deposition tends to be near-axis.
    minority_concentration_scalar = minority_concentration_profile[0]

    # Construct inputs for ToricNN.
    volume_average_temperature = (
        math_utils.cell_integration(
            solver_profiles.kinetic.T_e.value * geo.vpr, geo
        )
        / geo.volume_face[-1]
    )
    volume_average_density = (
        math_utils.cell_integration(
            solver_profiles.kinetic.n_e.value * geo.vpr, geo
        )
        / geo.volume_face[-1]
    )

    # Peaking factors are core w.r.t volume averages.
    temperature_peaking_factor = (
        solver_profiles.kinetic.T_e.left_face_value[0]
        / volume_average_temperature
    )
    density_peaking_factor = (
        solver_profiles.kinetic.n_e.left_face_value[0] / volume_average_density
    )
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
        minority_concentration=minority_concentration_scalar
        * 100,  # Convert to percentage.
        gap_inner=gap_inner,
        gap_outer=gap_outer,
        z0=geo.z_magnetic_axis(),
        temperature_peaking_factor=temperature_peaking_factor,
        density_peaking_factor=density_peaking_factor,
        B_0=geo.B_0,
    )

    toric_nn_outputs = toric_nn_lib.predict(toric_nn, toric_inputs)
    toric_grid = jnp.linspace(0.0, 1.0, toric_nn_lib.TORIC_GRID_SIZE)

    # Ideally total ICRH power should equal one, but normalise if not.
    power_deposition_he3 = jnp.interp(
        geo.rho_norm,
        toric_grid,
        toric_nn_outputs.power_deposition_He3,
    )
    power_deposition_e = jnp.interp(
        geo.rho_norm,
        toric_grid,
        toric_nn_outputs.power_deposition_e,
    )
    power_deposition_2T = jnp.interp(
        geo.rho_norm,
        toric_grid,
        toric_nn_outputs.power_deposition_2T,
    )
    power_deposition_all = (
        power_deposition_2T + power_deposition_e + power_deposition_he3
    )

    total_power_deposition = math_utils.cell_integration(
        power_deposition_all * geo.vpr, geo
    )
    power_deposition_he3 /= total_power_deposition
    power_deposition_e /= total_power_deposition
    power_deposition_2T /= total_power_deposition

    # Computing fast ion density and temperatures for He3.
    he3_atomic_mass = 3.016
    he3_charge_number = 2

    absorbed_power = source_params.P_total * source_params.absorption_fraction

    n_tail, T_tail = fast_ion_utils.bimaxwellian_split(
        power_deposition=power_deposition_he3,
        T_e=solver_profiles.kinetic.T_e.value,
        n_e=solver_profiles.kinetic.n_e.value,
        T_i=solver_profiles.kinetic.T_i.value,
        n_i=solver_profiles.kinetic.n_i.value,
        minority_concentration=minority_concentration_profile,
        P_total_W=absorbed_power,
        charge_number=he3_charge_number,
        mass_number=he3_atomic_mass,
        bulk_ion_mass=solver_profiles.ions.A_i,
        Z_i=solver_profiles.ions.Z_i,
        n_impurity=solver_profiles.n_impurity_bundled.value,
        Z_impurity=solver_profiles.ions.Z_impurity_bundled,
        A_impurity=solver_profiles.ions.A_impurity,
    )

    he3_fast_ion = fast_ion_lib.FastIon.from_arrays(
        species='He3',
        source_name=self.SOURCE_ID,
        geo=geo,
        n=n_tail,
        T=T_tail,
        T_right_face_constraint=solver_profiles.kinetic.T_i.right_face_constraint,
    )

    frac_ion_heating = fast_ion_lib.mikkelsen_fast_ion_heating_fraction(
        T_tail,
        solver_profiles.kinetic.T_e.value,
        he3_atomic_mass,
    )
    source_ion = power_deposition_he3 * frac_ion_heating * absorbed_power
    source_el = power_deposition_he3 * (1 - frac_ion_heating) * absorbed_power

    # Assume that all the power from the electron power profile goes to
    # electrons.
    source_el += power_deposition_e * absorbed_power

    # Assume that all the power from the tritium power profile goes to ions.
    source_ion += power_deposition_2T * absorbed_power

    return (
        torax_sources.SourceOutput(
            power_density_i=source_ion,
            power_density_e=source_el,
            fast_ions=(he3_fast_ion,),
        ),
        source_state,
    )


class ToricNNIonCyclotronSourceConfig(torax_sources.SourceConfigBase):
  """Configuration for the IonCyclotronSource using the ToricNN model.

  Attributes:
    model_name: Discriminator literal identifying the ToricNN model.
    P_total: Total heating power [W].
    absorption_fraction: Fraction of absorbed power.
    minority_species: Symbol of the minority species. Currently only 'He3' is
      supported by the ToricNN model.
    model_path: Path to JSON weights and model config of ToricNN model. Defaults
      to the bundled `toricnn.json` weights (`toric_nn.DEFAULT_MODEL_PATH`).
    wall_inner: Inner radial location of first wall at plasma midplane level
      [m].
    wall_outer: Outer radial location of first wall at plasma midplane level
      [m].
    frequency: ICRF wave frequency [Hz].
  """

  model_name: Annotated[Literal['toric_nn'], torax_pydantic.JAX_STATIC] = (
      'toric_nn'
  )
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      10e6
  )
  absorption_fraction: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  minority_species: Annotated[Literal['He3'], torax_pydantic.JAX_STATIC] = 'He3'
  model_path: Annotated[str, torax_pydantic.JAX_STATIC] = (
      toric_nn_lib.DEFAULT_MODEL_PATH
  )
  wall_inner: pydantic.PositiveFloat = 1.24
  wall_outer: pydantic.PositiveFloat = 2.43
  frequency: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )

  def build_source(self) -> ToricNNIonCyclotronSource:
    """Builds the ToricNNIonCyclotronSource and eagerly validates `model_path`."""
    _load_toric_nn(self.model_path)
    return ToricNNIonCyclotronSource(model_path=self.model_path)

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_params),
        P_total=self.P_total.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
        minority_species=self.minority_species,
        wall_inner=self.wall_inner,
        wall_outer=self.wall_outer,
        frequency=self.frequency.get_value(t),
    )
