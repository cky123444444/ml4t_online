import os
import sys
import uuid
from typing import Any, Dict

from flask import Flask, jsonify, request
import requests

# Ensure project-relative imports work with existing modules that use absolute names like `utils` and `Alpha102`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
CALC_DIR = os.path.join(SRC_DIR, 'feature', 'calculator')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CALC_DIR not in sys.path:
    sys.path.insert(0, CALC_DIR)

from src.ops.dumper.sql_dumper import get_sql_dumper
from src.serving.pipeline_runtime import (
    PipelineConfigError,
    PipelineSettings,
    create_pipeline_runtime,
)
from src.utils.logger import (
    clear_request_id,
    get_request_id,
    normalize_request_id,
    set_request_id,
    setup_logger,
)

logger = setup_logger('server')
app = Flask(__name__)

# -----------------------------
# Configuration (env-overridable)
# -----------------------------
TS_BASE_URL = os.environ.get('TS_BASE_URL', 'http://127.0.0.1:8080')
DEFAULT_MODEL = os.environ.get('TS_MODEL', 'dragonnet')

HDF_PATH = os.environ.get('HDF_PATH', './torchserve/data/last_max_window_10000_ochlv.hdf')
HDF_KEY = os.environ.get('HDF_KEY', 'my_data')

NUM_FEATURES = int(os.environ.get('NUM_FEATURES', '111'))
HISTORY_WINDOW = int(os.environ.get('HISTORY_WINDOW', '14000'))
REQUEST_TIMEOUT = float(os.environ.get('TS_TIMEOUT', '30'))

OFFLINE_DEBUG_MODE = os.environ.get('OFFLINE_DEBUG_MODE', '0') == '1'
DEBUG_OUTPUT_DIR = os.environ.get('DEBUG_OUTPUT_DIR', '/app/debug_output')

ENABLE_FEATURE_DUMP = os.environ.get('ENABLE_FEATURE_DUMP', '1') == '1'
FEATURE_DUMP_SYMBOL = os.environ.get('FEATURE_DUMP_SYMBOL', 'BTCUSDT')

PIPELINE_CONFIG_PATH = os.environ.get(
    'PIPELINE_CONFIG_PATH',
    os.path.join(os.path.dirname(__file__), 'pipelines.json'),
)

PIPELINE_RUNTIME = create_pipeline_runtime(
    PipelineSettings(
        ts_base_url=TS_BASE_URL,
        request_timeout=REQUEST_TIMEOUT,
        default_model=DEFAULT_MODEL,
        hdf_path=HDF_PATH,
        hdf_key=HDF_KEY,
        num_features=NUM_FEATURES,
        history_window=HISTORY_WINDOW,
        offline_debug_mode=OFFLINE_DEBUG_MODE,
        debug_output_dir=DEBUG_OUTPUT_DIR,
        enable_feature_dump=ENABLE_FEATURE_DUMP,
        feature_dump_symbol=FEATURE_DUMP_SYMBOL,
        pipeline_config_path=PIPELINE_CONFIG_PATH,
    )
)
PIPELINE_CONFIG = PIPELINE_RUNTIME.config
PIPELINE_RUNNER = PIPELINE_RUNTIME.runner


def _new_request_id() -> str:
    return str(uuid.uuid4())[:8]


@app.before_request
def _bind_request_id():
    incoming_req_id = normalize_request_id(request.headers.get('X-Request-ID'))
    set_request_id(incoming_req_id or _new_request_id())


@app.after_request
def _add_request_id_header(response):
    response.headers['X-Request-ID'] = get_request_id()
    return response


@app.teardown_request
def _cleanup_request_id(_exc):
    clear_request_id()


def process_1(req_params: Dict[str, Any]):
    return PIPELINE_RUNNER.run('p_hdf_local_strategy', req_params)


def process_2(req_params: Dict[str, Any]):
    return PIPELINE_RUNNER.run('p_binance_sql_dump', req_params)


def process_3(req_params: Dict[str, Any]):
    return PIPELINE_RUNNER.run('p_binance_hdf_strategy_dump', req_params)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        model_name = (payload.get('model') or DEFAULT_MODEL).strip()

        request_id = normalize_request_id(payload.get('request_id')) or get_request_id()
        set_request_id(request_id)

        payload['request_id'] = request_id
        payload['model'] = model_name

        pipeline_name = (payload.get('pipeline') or '').strip()
        method = (payload.get('method') or '').strip()

        if pipeline_name:
            selected_pipeline = pipeline_name
        elif method:
            selected_pipeline = PIPELINE_CONFIG.get('method_mapping', {}).get(method)
            if not selected_pipeline:
                raise PipelineConfigError(f'Unknown method: {method}')
        else:
            selected_pipeline = PIPELINE_CONFIG['default_pipeline']

        logger.info(
            f'Received predict request: model={model_name}, pipeline={selected_pipeline}, method={method or "<none>"}'
        )

        model_response = PIPELINE_RUNNER.run(selected_pipeline, payload)

        logger.info('Predict request completed successfully')
        return jsonify({'model': model_name, 'request_id': request_id, 'result': model_response})
    except PipelineConfigError as e:
        logger.error(f'Invalid pipeline request: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 400
    except requests.HTTPError as http_err:
        response = getattr(http_err, 'response', None)
        logger.error(f'TorchServe HTTP error: {http_err}', exc_info=True)
        return jsonify({
            'error': 'TorchServe HTTP error',
            'details': str(http_err),
            'response': response.text if response else None,
        }), 502
    except Exception as e:
        logger.error(f'Predict request failed: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    logger.debug('Health check requested')
    return jsonify({
        'status': 'ok',
        'torchserve': TS_BASE_URL,
        'default_model': DEFAULT_MODEL,
        'hdf_path': HDF_PATH,
        'hdf_key': HDF_KEY or '<auto>',
        'num_features': NUM_FEATURES,
        'history_window': HISTORY_WINDOW,
        'default_pipeline': PIPELINE_CONFIG.get('default_pipeline'),
    })


@app.route('/feature_dump/status', methods=['GET'])
def feature_dump_status():
    dumper = get_sql_dumper()
    stats = dumper.get_db_stats()
    return jsonify({
        'enabled': ENABLE_FEATURE_DUMP,
        'db_path': dumper.db_path,
        'queue_size': stats.get('queue_size', 0),
        'record_count': stats.get('record_count', 0),
        'db_size_bytes': stats.get('db_size_bytes', 0),
        'db_size_human': stats.get('db_size_human', '0 B'),
    })


@app.route('/config', methods=['GET'])
def config_info():
    logger.debug('Config info requested')
    return jsonify({
        'TS_BASE_URL': TS_BASE_URL,
        'TS_MODEL': DEFAULT_MODEL,
        'HDF_PATH': HDF_PATH,
        'HDF_KEY': HDF_KEY or '<auto>',
        'NUM_FEATURES': NUM_FEATURES,
        'HISTORY_WINDOW': HISTORY_WINDOW,
        'TS_TIMEOUT': REQUEST_TIMEOUT,
        'DEBUG_OUTPUT_DIR': DEBUG_OUTPUT_DIR,
        'ENABLE_FEATURE_DUMP': ENABLE_FEATURE_DUMP,
        'FEATURE_DUMP_SYMBOL': FEATURE_DUMP_SYMBOL,
        'PIPELINE_CONFIG_PATH': PIPELINE_CONFIG_PATH,
        'DEFAULT_PIPELINE': PIPELINE_CONFIG.get('default_pipeline'),
        'AVAILABLE_PIPELINES': sorted(list(PIPELINE_CONFIG.get('pipelines', {}).keys())),
    })


if __name__ == '__main__':
    from src.ops.scheduler.background_polling import start_polling_thread

    logger.info('=' * 50)
    logger.info('Starting Flask server...')
    logger.info('Configuration:')
    logger.info(f'  - OFFLINE_DEBUG_MODE: {OFFLINE_DEBUG_MODE}')
    logger.info(f'  - TorchServe URL: {TS_BASE_URL}')
    logger.info(f'  - Default model: {DEFAULT_MODEL}')
    logger.info(f'  - HDF path: {HDF_PATH}')
    logger.info(f'  - Feature dump enabled: {ENABLE_FEATURE_DUMP}')
    logger.info(f'  - Pipeline config path: {PIPELINE_CONFIG_PATH}')
    logger.info(f"  - Default pipeline: {PIPELINE_CONFIG.get('default_pipeline')}")
    logger.info('=' * 50)

    start_polling_thread(interval_sec=3)
    app.run(host='0.0.0.0', port=8000, debug=OFFLINE_DEBUG_MODE, use_reloader=False)
