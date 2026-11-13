"""Tests for the TORAX ToricNN ion-cyclotron source interface."""

import dataclasses
import json
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import pydantic
import torax
from torax import sources as torax_sources
from torax.experimental import fast_ion as fast_ion_lib
from toric_surrogate.TORIC_MLP_v1 import torax_interface
from toric_surrogate.TORIC_MLP_v1 import toric_nn as toric_nn_lib


class ToricNNTest(parameterized.TestCase):
  """Test cases for the ToricNN ICRH model and TORAX interface."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torax_sources.register_source_config(
        torax_interface.ToricNNIonCyclotronSourceConfig
    )

  def setUp(self):
    super().setUp()
    # pylint: disable=protected-access
    torax_interface._load_toric_nn.cache_clear()
    # pylint: enable=protected-access
    self._dummy_model_path = self.create_tempfile('toricnn.json').full_path
    toric_nn = toric_nn_lib.ToricNN(
        hidden_sizes=[3],
        pca_coeffs=4,
        input_dim=10,
        radial_nodes=toric_nn_lib.TORIC_GRID_SIZE,
    )
    model_input = toric_nn_lib.ToricNNInputs(
        frequency=120.0e6,
        volume_average_temperature=6,
        volume_average_density=5,
        minority_concentration=2.0,  # In percentage.
        gap_inner=0.01,
        gap_outer=0.01,
        z0=0.0,
        temperature_peaking_factor=2,
        density_peaking_factor=2,
        B_0=12.2,
    )
    toric_input = jnp.array([
        model_input.frequency,
        model_input.volume_average_temperature,
        model_input.volume_average_density,
        model_input.minority_concentration,
        model_input.gap_inner,
        model_input.gap_outer,
        model_input.z0,
        model_input.temperature_peaking_factor,
        model_input.density_peaking_factor,
        model_input.B_0,
    ])
    model_output, params = toric_nn.init_with_output(
        jax.random.PRNGKey(0), toric_input
    )
    config = dataclasses.asdict(toric_nn)
    config[toric_nn_lib.HELIUM3_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params['params']
    )
    config[toric_nn_lib.TRITIUM_SECOND_HARMONIC_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params['params']
    )
    config[toric_nn_lib.ELECTRON_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params['params']
    )
    with open(self._dummy_model_path, 'w', encoding='utf-8') as f:
      json.dump(config, f, indent=4, separators=(',', ':'))
    self.dummy_input = model_input
    self.dummy_output = model_output
    self._source_name = torax_interface.ToricNNIonCyclotronSource.SOURCE_ID

  def _get_default_config_dict(
      self, minority_concentration: float = 0.03
  ) -> dict[str, Any]:
    return {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'psi_source': 'j',
        },
        'plasma_composition': {
            'main_ion': {'D': 0.5, 'T': 0.5},
            'impurity': {
                'impurity_mode': 'n_e_ratios',
                'species': {'He3': minority_concentration},
            },
        },
        'numerics': {},
        'geometry': {'model_name': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

  def _make_initial_inputs(
      self, torax_config: torax.ToraxConfig
  ) -> tuple[torax.CoreProfiles, torax.RuntimeParams, torax.Geometry]:
    step_fn = torax.experimental.make_step_fn(torax_config)
    state, _ = torax.experimental.get_initial_state_and_post_processed_outputs(
        step_fn
    )
    runtime_params = step_fn.runtime_params_provider(
        t=torax_config.time_step_calculator.t_initial
    )
    return state.core_profiles, runtime_params, state.geometry

  def test_build_runtime_params(self):
    config = self._get_default_config_dict()
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source = torax_config.sources.icrh
    self.assertIsInstance(
        source, torax_interface.ToricNNIonCyclotronSourceConfig
    )
    runtime_params = source.build_runtime_params(t=0.0)
    self.assertIsInstance(runtime_params, torax_sources.RuntimeParams)

  @parameterized.product(
      mode=(
          torax_sources.Mode.ZERO,
          torax_sources.Mode.MODEL_BASED,
          torax_sources.Mode.PRESCRIBED,
      ),
      is_explicit=(True, False),
  )
  def test_runtime_params_builds_dynamic_params(
      self, mode: torax_sources.Mode, is_explicit: bool
  ):
    config = self._get_default_config_dict()
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'mode': mode,
            'is_explicit': is_explicit,
            'model_path': self._dummy_model_path,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_config = torax_config.sources.icrh
    self.assertIsInstance(
        source_config, torax_interface.ToricNNIonCyclotronSourceConfig
    )
    dynamic_params = source_config.build_runtime_params(t=0.0)
    self.assertIsInstance(dynamic_params, torax_sources.RuntimeParams)
    self.assertEqual(dynamic_params.mode, mode)
    self.assertEqual(dynamic_params.is_explicit, is_explicit)

  def test_config_raises_if_model_path_does_not_exist(self):
    config = torax_interface.ToricNNIonCyclotronSourceConfig(
        model_path='/tmp/non_existent_file.json'
    )
    with self.assertRaises(FileNotFoundError):
      _ = config.build_source()

  @parameterized.product(
      total_power=[1e6, 20e6, 120e6],
      frequency=[119e6, 120e6, 121e6],
      minority_concentration=[0.01, 0.02, 0.03, 0.04, 0.05],
  )
  def test_icrh_output_matches_total_power(
      self,
      total_power: float,
      frequency: float,
      minority_concentration: float,
  ):
    """Tests source outputs match total heating power."""
    config = self._get_default_config_dict(
        minority_concentration=minority_concentration
    )
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'P_total': total_power,
            'frequency': frequency,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.assertIsInstance(
        torax_config.sources.icrh,
        torax_interface.ToricNNIonCyclotronSourceConfig,
    )
    source_models = torax_config.sources.build_models()
    icrh_source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    icrh_output, _ = icrh_source.get_value(
        runtime_params,
        geo,
        core_profiles,
        source_state=icrh_source.initial_state(runtime_params),
    )
    assert icrh_output.power_density_i is not None
    assert icrh_output.power_density_e is not None
    ion_el_total = icrh_output.power_density_i + icrh_output.power_density_e
    integrated_power = jnp.sum(ion_el_total * geo.vpr * geo.drho_norm)

    np.testing.assert_allclose(
        integrated_power,
        total_power,
    )

  def test_toric_nn_loads_and_predicts(self):
    """Tests that the ToricNN model loads default weights and predicts."""
    expected_output_path = pathlib.Path(__file__).parent / 'data'
    expected_output_helium3 = np.load(
        expected_output_path / 'toricnn_He3_expected_output.npy'
    )
    expected_output_tritium = np.load(
        expected_output_path / 'toricnn_2T_expected_output.npy'
    )
    expected_output_e = np.load(
        expected_output_path / 'toricnn_e_expected_output.npy'
    )
    toric_nn = toric_nn_lib.ToricNNWrapper()
    model_input = toric_nn_lib.ToricNNInputs(
        frequency=120.0e6,
        volume_average_temperature=6,
        volume_average_density=5,
        minority_concentration=2.0,
        gap_inner=0.01,
        gap_outer=0.01,
        z0=0.0,
        temperature_peaking_factor=2,
        density_peaking_factor=2,
        B_0=12.2,
    )

    output = toric_nn_lib.predict(toric_nn, model_input)

    np.testing.assert_array_almost_equal(
        output.power_deposition_He3, expected_output_helium3[0]
    )
    np.testing.assert_array_almost_equal(
        output.power_deposition_2T, expected_output_tritium[0]
    )
    np.testing.assert_array_almost_equal(
        output.power_deposition_e, expected_output_e[0]
    )

    # Test that the model predicts correctly when B_0 is changed.
    model_input = dataclasses.replace(model_input, B_0=11.8)
    output = toric_nn_lib.predict(toric_nn, model_input)
    np.testing.assert_array_almost_equal(
        output.power_deposition_He3, expected_output_helium3[1]
    )
    np.testing.assert_array_almost_equal(
        output.power_deposition_2T, expected_output_tritium[1]
    )
    np.testing.assert_array_almost_equal(
        output.power_deposition_e, expected_output_e[1]
    )

  def test_toric_nn_loads_and_predicts_with_dummy_model(self):
    """Tests that the ToricNNWrapper loads and predicts consistently."""
    toric_wrapper = toric_nn_lib.ToricNNWrapper(path=self._dummy_model_path)
    wrapper_output = toric_nn_lib.predict(toric_wrapper, self.dummy_input)
    np.testing.assert_array_almost_equal(
        wrapper_output.power_deposition_He3,
        self.dummy_output,
    )
    np.testing.assert_array_almost_equal(
        wrapper_output.power_deposition_2T,
        self.dummy_output,
    )
    np.testing.assert_array_almost_equal(
        wrapper_output.power_deposition_e,
        self.dummy_output,
    )

  def test_source_value(self):
    """Tests that the registered source can provide a value."""
    config = self._get_default_config_dict()
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.assertIsInstance(
        torax_config.sources.icrh,
        torax_interface.ToricNNIonCyclotronSourceConfig,
    )
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    self.assertIsInstance(source, torax_sources.Source)
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    icrh_out, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    assert icrh_out.power_density_i is not None
    assert icrh_out.power_density_e is not None
    self.assertEqual(icrh_out.power_density_i.shape, geo.rho.shape)
    self.assertEqual(icrh_out.power_density_e.shape, geo.rho.shape)

  def test_source_with_minority_species_from_composition(self):
    """Tests ICRH source with minority_species reading from composition."""
    config = self._get_default_config_dict()
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'minority_species': 'He3',
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    self.assertIsInstance(source, torax_sources.Source)
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    icrh_params = runtime_params.sources[self._source_name]
    self.assertIsInstance(icrh_params, torax_interface.RuntimeParams)
    self.assertEqual(icrh_params.minority_species, 'He3')
    self.assertIn('He3', runtime_params.plasma_composition.impurity_names)
    icrh_out, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    assert icrh_out.power_density_i is not None
    assert icrh_out.power_density_e is not None
    self.assertEqual(icrh_out.power_density_i.shape, geo.rho.shape)
    self.assertEqual(icrh_out.power_density_e.shape, geo.rho.shape)

  def test_source_with_minority_species_main_ion(self):
    """Tests ICRH source with minority_species as a main ion."""
    config = self._get_default_config_dict()
    config['plasma_composition'] = {
        'main_ion': {
            'D': 0.47,
            'T': 0.5,
            'He3': 0.03,
        },
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'Ne': 0.0},
        },
    }
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'minority_species': 'He3',
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)

    icrh_out, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )

    assert icrh_out.power_density_i is not None
    self.assertEqual(icrh_out.power_density_i.shape, geo.rho.shape)

  def test_invalid_minority_species_raises(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Input should be 'He3'",
    ):
      torax_interface.ToricNNIonCyclotronSourceConfig(minority_species='D')

  def test_source_raises_if_minority_species_not_in_composition(self):
    """Tests that a ValidationError is raised if minority species is missing."""
    config = self._get_default_config_dict()
    config['plasma_composition'] = {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {'impurity_mode': 'n_e_ratios', 'species': {'Ne': 0.01}},
    }
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'minority_species': 'He3',
        }
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'The ICRH source requires "He3" to be present',
    ):
      torax.ToraxConfig.from_dict(config)

  def test_icrh_returns_fast_ion_data(self):
    config = self._get_default_config_dict()
    config['numerics'] = {'enable_fast_ions': True}
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'P_total': 10e6,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    assert output.power_density_i is not None
    assert output.power_density_e is not None
    fast_ion_data = output.fast_ions
    self.assertIsInstance(fast_ion_data, tuple)
    self.assertLen(fast_ion_data, 1)
    fi = fast_ion_data[0]
    self.assertIsInstance(fi, fast_ion_lib.FastIon)
    self.assertEqual(fi.species, 'He3')
    self.assertEqual(fi.source, self._source_name)
    self.assertEqual(fi.n.value.shape, geo.rho.shape)
    self.assertEqual(fi.T.value.shape, geo.rho.shape)

  def test_fast_ion_density_conservation(self):
    minority_conc = 0.03
    config = self._get_default_config_dict(minority_concentration=minority_conc)
    config['numerics'] = {'enable_fast_ions': True}
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'P_total': 10e6,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    fast_ion_data = output.fast_ions
    n_tail = fast_ion_data[0].n.value
    n_total = core_profiles.kinetic.n_e.value * minority_conc
    self.assertTrue(jnp.all(n_tail >= 0))
    self.assertTrue(jnp.all(n_tail <= n_total))

  def test_fast_ion_tail_temperature_above_electron_temp(self):
    config = self._get_default_config_dict()
    config['numerics'] = {'enable_fast_ions': True}
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'P_total': 10e6,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    fast_ion_data = output.fast_ions
    temperature_tail = fast_ion_data[0].T.value
    self.assertTrue(
        jnp.all(temperature_tail >= core_profiles.kinetic.T_e.value)
    )

  def test_zero_power_produces_zero_fast_ions(self):
    config = self._get_default_config_dict()
    config['numerics'] = {'enable_fast_ions': True}
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'P_total': 0.0,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params, geo = self._make_initial_inputs(torax_config)
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    fast_ion_data = output.fast_ions
    np.testing.assert_allclose(fast_ion_data[0].n.value, 0.0, atol=1e-9)
    np.testing.assert_allclose(
        fast_ion_data[0].T.value, core_profiles.kinetic.T_i.value
    )

  @parameterized.parameters(
      (20e6, 0.5),
      (40e6, 0.25),
  )
  def test_absorption_fraction_scales_heating_and_fast_ions(
      self, p_total: float, absorption_fraction: float
  ):
    config = self._get_default_config_dict()
    config['numerics'] = {'enable_fast_ions': True}
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'P_total': p_total,
            'absorption_fraction': absorption_fraction,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    core_profiles, runtime_params_partial, geo = self._make_initial_inputs(
        torax_config
    )
    output_partial, _ = source.get_value(
        runtime_params=runtime_params_partial,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params_partial),
    )
    source_ion_partial = output_partial.power_density_i
    source_el_partial = output_partial.power_density_e
    fast_ions_partial = output_partial.fast_ions

    torax_config.update_fields({
        f'sources.{self._source_name}.P_total': p_total * absorption_fraction,
        f'sources.{self._source_name}.absorption_fraction': 1.0,
    })
    runtime_params_full = torax.experimental.make_step_fn(
        torax_config
    ).runtime_params_provider(t=torax_config.time_step_calculator.t_initial)
    output_full, _ = source.get_value(
        runtime_params=runtime_params_full,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params_full),
    )
    source_ion_full = output_full.power_density_i
    source_el_full = output_full.power_density_e
    fast_ions_full = output_full.fast_ions

    np.testing.assert_allclose(source_ion_partial, source_ion_full)
    np.testing.assert_allclose(source_el_partial, source_el_full)
    np.testing.assert_allclose(
        fast_ions_partial[0].n.value,
        fast_ions_full[0].n.value,
    )
    np.testing.assert_allclose(
        fast_ions_partial[0].T.value,
        fast_ions_full[0].T.value,
    )


if __name__ == '__main__':
  absltest.main()
