"""Tests for the TORAX ToricNN ion-cyclotron source interface."""

import dataclasses
import json
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import pydantic
import torax
from torax import sources as torax_sources
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources.ion_cyclotron_source import base as icrh_base
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic
from toric_surrogate.TORIC_MLP_v1 import torax_interface
from toric_surrogate.TORIC_MLP_v1 import toric_nn as toric_nn_lib

# Index of He3 in _FAST_ION_SPECIES.
_HE3_INDEX = 3


class ToricNNTest(test_lib.SourceTestCase):
  """Test cases for the ToricNN ICRH model and TORAX interface."""

  source_name = icrh_base.IonCyclotronSource.SOURCE_NAME
  source_config_class = torax_interface.ToricNNIonCyclotronSourceConfig

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torax_sources.register_source_model_config(
        torax_interface.ToricNNIonCyclotronSourceConfig,
        icrh_base.IonCyclotronSource.SOURCE_NAME,
    )

  def setUp(self):
    super().setUp()
    # pylint: disable=protected-access
    torax_interface._icrh_model_func_with_toric_nn.cache_clear()
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
    self._source_name = icrh_base.IonCyclotronSource.SOURCE_NAME

  def test_build_runtime_params(self):
    source = torax_interface.ToricNNIonCyclotronSourceConfig.from_dict(
        {'model_path': self._dummy_model_path}
    )
    self.assertIsInstance(
        source, torax_interface.ToricNNIonCyclotronSourceConfig
    )
    face_centers = interpolated_param_2d.get_face_centers(nx=4)
    torax_pydantic.set_grid(
        source, torax_pydantic.Grid1D(face_centers=face_centers)
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
    source_config = torax_interface.ToricNNIonCyclotronSourceConfig.from_dict({
        'mode': mode,
        'is_explicit': is_explicit,
        'model_path': self._dummy_model_path,
    })
    face_centers = interpolated_param_2d.get_face_centers(nx=4)
    torax_pydantic.set_grid(
        source_config,
        torax_pydantic.Grid1D(face_centers=face_centers),
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
      _ = config.model_func

  @parameterized.product(
      total_power=[1e6, 20e6, 120e6],
      frequency=[119e6, 120e6, 121e6],
      minority_concentration=[1, 2, 3, 4, 5],
  )
  def test_icrh_output_matches_total_power(
      self,
      total_power: float,
      frequency: float,
      minority_concentration: float,
  ):
    """Tests source outputs match total heating power."""
    config = default_configs.get_default_config_dict()
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'P_total': total_power,
            'frequency': frequency,
            'minority_concentration': minority_concentration,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.assertIsInstance(
        torax_config.sources.icrh,
        torax_interface.ToricNNIonCyclotronSourceConfig,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    icrh_source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    icrh_output = icrh_source.get_value(
        runtime_params,
        geo,
        core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    ion_el_total = icrh_output[0] + icrh_output[1]  # pyrefly: ignore[unsupported-operation]
    integrated_power = jnp.sum(ion_el_total * geo.vpr * geo.drho_norm)  # pyrefly: ignore[bad-argument-type]

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
    config = default_configs.get_default_config_dict()
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
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(source, torax_sources.Source)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    icrh_out = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(icrh_out, 3)
    self.assertEqual(icrh_out[0].shape, geo.rho.shape)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(icrh_out[1].shape, geo.rho.shape)  # pyrefly: ignore[missing-attribute]

  def test_source_with_minority_species_from_composition(self):
    """Tests ICRH source with minority_species reading from composition."""
    config = default_configs.get_default_config_dict()
    config['plasma_composition'] = {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'He3': 0.03},
        },
    }
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'minority_species': 'He3',
            'minority_concentration': 0.01,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(source, torax_sources.Source)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    icrh_params = runtime_params.sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(icrh_params, torax_interface.RuntimeParams)
    self.assertEqual(icrh_params.minority_species, 'He3')
    self.assertIn('He3', runtime_params.plasma_composition.impurity_names)
    icrh_out = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(icrh_out, 3)
    self.assertEqual(icrh_out[0].shape, geo.rho.shape)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(icrh_out[1].shape, geo.rho.shape)  # pyrefly: ignore[missing-attribute]

  def test_source_with_minority_species_main_ion(self):
    """Tests ICRH source with minority_species as a main ion."""
    config = default_configs.get_default_config_dict()
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
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]

    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    icrh_out = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )

    self.assertLen(icrh_out, 3)
    self.assertEqual(icrh_out[0].shape, geo.rho.shape)  # pyrefly: ignore[missing-attribute]

  def test_source_raises_if_minority_species_not_he3(self):
    """Tests that a ValidationError is raised if minority_species != 'He3'."""
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Minority species must be 'He3' if specified",
    ):
      torax_interface.ToricNNIonCyclotronSourceConfig(minority_species='D')

  def test_source_raises_if_minority_species_not_in_composition(self):
    """Tests that a ValueError is raised if minority species is missing."""
    config = default_configs.get_default_config_dict()
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
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    with self.assertRaisesRegex(
        ValueError,
        'Minority species He3 not found in plasma composition',
    ):
      core_profiles = initialization.initial_core_profiles(
          runtime_params=runtime_params,
          geo=geo,
          source_models=source_models,
          neoclassical_models=neoclassical_models,
      )
      source.get_value(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          calculated_source_profiles=None,
          conductivity=None,
      )

  def test_minority_concentration_warning_by_default(self):
    with self.assertLogs(level='WARNING') as cm:
      torax_interface.ToricNNIonCyclotronSourceConfig()
    self.assertTrue(
        any('minority_concentration is provided' in o for o in cm.output)
    )

  def test_minority_concentration_warning_when_explicitly_set_to_non_none(self):
    with self.assertLogs(level='WARNING') as cm:
      torax_interface.ToricNNIonCyclotronSourceConfig(
          minority_concentration=0.05
      )
    self.assertTrue(
        any('minority_concentration is provided' in o for o in cm.output)
    )

  def test_minority_concentration_no_warning_when_none(self):
    with self.assertNoLogs(level='WARNING'):
      torax_interface.ToricNNIonCyclotronSourceConfig(
          minority_concentration=None
      )

  def test_icrh_returns_fast_ion_data(self):
    config = default_configs.get_default_config_dict()
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
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(output, 3)
    fast_ion_data = output[2]
    self.assertIsInstance(fast_ion_data, tuple)
    self.assertLen(fast_ion_data, 5)
    fi = fast_ion_data[_HE3_INDEX]
    self.assertIsInstance(fi, fast_ion_lib.FastIon)
    self.assertEqual(fi.species, 'He3')
    self.assertEqual(fi.source, icrh_base.IonCyclotronSource.SOURCE_NAME)
    self.assertEqual(fi.n.value.shape, geo.rho.shape)
    self.assertEqual(fi.T.value.shape, geo.rho.shape)

  def test_fast_ion_density_conservation(self):
    config = default_configs.get_default_config_dict()
    config['numerics'] = {'enable_fast_ions': True}
    minority_conc = 0.03
    config['sources'] = {
        self._source_name: {
            'model_name': 'toric_nn',
            'model_path': self._dummy_model_path,
            'P_total': 10e6,
            'minority_concentration': minority_conc,
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output[2]
    n_tail = fast_ion_data[_HE3_INDEX].n.value  # pyrefly: ignore[missing-attribute]
    n_total = core_profiles.n_e.value * minority_conc
    self.assertTrue(jnp.all(n_tail >= 0))
    self.assertTrue(jnp.all(n_tail <= n_total))

  def test_fast_ion_tail_temperature_above_electron_temp(self):
    config = default_configs.get_default_config_dict()
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
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output[2]
    temperature_tail = fast_ion_data[_HE3_INDEX].T.value  # pyrefly: ignore[missing-attribute]
    self.assertTrue(jnp.all(temperature_tail >= core_profiles.T_e.value))

  def test_zero_power_produces_zero_fast_ions(self):
    config = default_configs.get_default_config_dict()
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
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output[2]
    np.testing.assert_allclose(
        fast_ion_data[_HE3_INDEX].n.value, 0.0, atol=1e-9  # pyrefly: ignore[missing-attribute]
    )
    np.testing.assert_allclose(
        fast_ion_data[_HE3_INDEX].T.value, core_profiles.T_i.value  # pyrefly: ignore[missing-attribute]
    )


if __name__ == '__main__':
  absltest.main()
