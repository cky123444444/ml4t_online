import unittest
from unittest.mock import patch

from src.serving import server


class TestServerPipelineRouting(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_method_mapping_process_2(self):
        with patch.object(server.PIPELINE_RUNNER, 'run', return_value={'ok': True}) as mock_run:
            resp = self.client.post('/predict', json={'method': 'process_2', 'model': 'dragonnet'})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.get_json()['result']['ok'])
            self.assertEqual(mock_run.call_args[0][0], 'p_binance_sql_dump')

    def test_pipeline_has_higher_priority_than_method(self):
        with patch.object(server.PIPELINE_RUNNER, 'run', return_value={'ok': True}) as mock_run:
            resp = self.client.post(
                '/predict',
                json={
                    'pipeline': 'p_hdf_local_strategy',
                    'method': 'process_2',
                    'model': 'dragonnet',
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(mock_run.call_args[0][0], 'p_hdf_local_strategy')

    def test_default_pipeline_when_none_provided(self):
        with patch.object(server.PIPELINE_RUNNER, 'run', return_value={'ok': True}) as mock_run:
            resp = self.client.post('/predict', json={'model': 'dragonnet'})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(mock_run.call_args[0][0], server.PIPELINE_CONFIG['default_pipeline'])

    def test_unknown_pipeline_returns_400(self):
        resp = self.client.post('/predict', json={'pipeline': 'not_exist'})
        self.assertEqual(resp.status_code, 400)
        self.assertIn('Unknown pipeline', resp.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
