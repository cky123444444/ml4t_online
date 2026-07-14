import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping

import requests

from src.ops.adaptor.binance_adaptor import BinanceAdaptor
from src.ops.calculator.sample_calculator import SampleCalculator
from src.ops.dumper.hdf_dumper import get_hdf_dumper
from src.ops.dumper.sql_dumper import get_sql_dumper
from src.ops.retriever.binance_cached_retriever import BinanceCachedRetriever
from src.ops.retriever.hdf_file_retriever import HDFFileRetriever
from src.ops.strategy.first_strategy import FirstStrategy
from src.ops.strategy.ple_strategy import PleStrategy
from src.utils.data_helper import get_latest_close
from src.utils.debug_utils import save_debug_data
from src.utils.logger import setup_logger

logger = setup_logger('pipeline_runtime')


class PipelineConfigError(ValueError):
    pass


class Node:
    def execute(self, ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        raise NotImplementedError


class FunctionNode(Node):
    def __init__(self, fn: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        self.fn = fn

    def execute(self, ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        self.fn(ctx, node_cfg)


class PipelineRunner:
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        node_registry: Mapping[str, Node],
        history_window: int,
        default_model: str,
    ):
        self.pipeline_config = pipeline_config
        self.node_registry = node_registry
        self.history_window = history_window
        self.default_model = default_model
        self._validate_config()

    def _validate_node(self, path: str, node_cfg: Any) -> None:
        if not isinstance(node_cfg, dict):
            raise PipelineConfigError(f'{path} must be an object')
        node_type = node_cfg.get('type')
        if not isinstance(node_type, str) or not node_type:
            raise PipelineConfigError(f'{path}.type must be a non-empty string')
        if node_type not in self.node_registry:
            raise PipelineConfigError(f'{path}.type unknown: {node_type}')

    def _validate_config(self) -> None:
        if not isinstance(self.pipeline_config, dict):
            raise PipelineConfigError('pipeline config must be a JSON object')

        pipelines = self.pipeline_config.get('pipelines')
        if not isinstance(pipelines, dict) or not pipelines:
            raise PipelineConfigError('pipelines must be a non-empty object')

        default_pipeline = self.pipeline_config.get('default_pipeline')
        if not isinstance(default_pipeline, str) or default_pipeline not in pipelines:
            raise PipelineConfigError('default_pipeline missing or not found in pipelines')

        method_mapping = self.pipeline_config.get('method_mapping', {})
        if not isinstance(method_mapping, dict):
            raise PipelineConfigError('method_mapping must be an object')

        for method_name, pipeline_name in method_mapping.items():
            if pipeline_name not in pipelines:
                raise PipelineConfigError(
                    f'method_mapping.{method_name} points to unknown pipeline: {pipeline_name}'
                )

        for pipeline_name, pipeline_cfg in pipelines.items():
            if not isinstance(pipeline_cfg, dict):
                raise PipelineConfigError(f'pipelines.{pipeline_name} must be an object')

            steps = pipeline_cfg.get('steps')
            if not isinstance(steps, list) or not steps:
                raise PipelineConfigError(f'pipelines.{pipeline_name}.steps must be a non-empty array')

            for i, step_cfg in enumerate(steps):
                self._validate_node(f'pipelines.{pipeline_name}.steps[{i}]', step_cfg)

    def run(self, pipeline_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        pipeline_cfg = self.pipeline_config['pipelines'].get(pipeline_name)
        if not pipeline_cfg:
            raise PipelineConfigError(f'Unknown pipeline: {pipeline_name}')

        ctx: Dict[str, Any] = {
            'pipeline_name': pipeline_name,
            'pipeline_cfg': pipeline_cfg,
            'payload': payload,
            'request_id': payload['request_id'],
            'model_name': payload.get('model', self.default_model),
            'req_debug_mode': bool(payload.get('req_debug_mode', False)),
            'raw_data': None,
            'df': None,
            'features': None,
            'model_response': None,
            'end_time': None,
            'history_window': self.history_window,
        }

        for step_cfg in pipeline_cfg['steps']:
            self.node_registry[step_cfg['type']].execute(ctx, step_cfg)

        return ctx['model_response']


@dataclass(frozen=True)
class PipelineRuntime:
    config_path: str
    config: Dict[str, Any]
    runner: PipelineRunner


@dataclass(frozen=True)
class PipelineSettings:
    ts_base_url: str
    request_timeout: float
    default_model: str
    hdf_path: str
    hdf_key: str
    num_features: int
    history_window: int
    offline_debug_mode: bool
    debug_output_dir: str
    enable_feature_dump: bool
    feature_dump_symbol: str
    pipeline_config_path: str


def _call_torchserve(ts_base_url: str, request_timeout: float, model_name: str, features: List[List[float]]):
    url = f'{ts_base_url}/predictions/{model_name}'
    headers = {'Content-Type': 'application/json'}
    logger.debug(f'Calling TorchServe: {url} with {len(features)} feature rows')
    resp = requests.post(url, headers=headers, json=features, timeout=request_timeout)
    resp.raise_for_status()
    try:
        result = resp.json()
        logger.debug('TorchServe response received successfully')
        return result
    except json.JSONDecodeError:
        logger.warning(f'TorchServe returned non-JSON response: {resp.text[:100]}')
        return {'raw': resp.text}


def _load_pipeline_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise PipelineConfigError(f'Pipeline config file not found: {config_path}')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise PipelineConfigError(f'Invalid pipeline config JSON: {config_path}: {e}') from e


def create_pipeline_runtime(settings: PipelineSettings) -> PipelineRuntime:
    def _save_debug_if_needed(ctx: Dict[str, Any], debug_key: str, data: Any) -> None:
        if not ctx['req_debug_mode']:
            return
        filename = ctx['pipeline_cfg'].get('debug_files', {}).get(debug_key)
        if filename:
            save_debug_data(data, filename, settings.debug_output_dir)

    def _node_retrieve_hdf(ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        hdf_path = node_cfg.get('hdf_path', settings.hdf_path)
        hdf_key = node_cfg.get('hdf_key', settings.hdf_key)
        df = HDFFileRetriever(hdf_path, hdf_key).execute()
        ctx['df'] = df
        logger.info(f'Loaded {len(df)} rows from HDF file')
        _save_debug_if_needed(ctx, 'raw_dataframe', df)

    def _node_retrieve_binance_cached(ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        symbol = node_cfg.get('symbol', settings.feature_dump_symbol)
        retriever = BinanceCachedRetriever(symbol=symbol)

        end_time = datetime.now(timezone.utc)
        if 'start_minutes_ago' in node_cfg:
            minutes_ago = int(node_cfg['start_minutes_ago'])
        elif node_cfg.get('start_minutes_ago_from_context') == 'history_window':
            minutes_ago = int(ctx['history_window'])
        else:
            raise PipelineConfigError(
                f"Pipeline {ctx['pipeline_name']} retrieve node requires start_minutes_ago"
            )

        start_time = end_time - timedelta(minutes=minutes_ago)
        logger.info(f'正在获取币安数据: {start_time} 到 {end_time}')
        raw_data = retriever.execute(start_time=start_time, end_time=end_time)

        if not raw_data:
            logger.error('未能从币安获取到数据')
            raise ValueError('未能从币安获取到数据')

        logger.info(f'获取到 {len(raw_data)} 条原始K线数据')
        ctx['raw_data'] = raw_data
        ctx['end_time'] = end_time

    def _node_adapt_binance(ctx: Dict[str, Any], _node_cfg: Dict[str, Any]) -> None:
        raw_data = ctx.get('raw_data')
        df = BinanceAdaptor(raw_data).execute()
        logger.info(f'转换后得到 {len(df)} 条DataFrame记录')
        if df.empty:
            logger.error('数据转换后为空')
            raise ValueError('数据转换后为空')

        ctx['df'] = df
        _save_debug_if_needed(ctx, 'adapted_dataframe', df)

    def _node_calculate_sample(ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        df = ctx.get('df')
        if df is None:
            raise ValueError('No dataframe available for calculation')

        if settings.offline_debug_mode and node_cfg.get('offline_debug_limit_rows'):
            limit_rows = int(node_cfg['offline_debug_limit_rows'])
            df = df.head(limit_rows)
            logger.debug(f'离线调试模式，只计算前 {len(df)} 条数据的特征')

        num_features = int(node_cfg.get('num_features', settings.num_features))
        max_window_size = int(node_cfg.get('max_window_size', settings.history_window))
        features = SampleCalculator(df, num_features=num_features, max_window_size=max_window_size).execute()

        logger.info(f'计算得到 {len(features)} 行特征, 维度: {len(features[0]) if features else 0}')
        ctx['features'] = features
        _save_debug_if_needed(ctx, 'features', features)

    def _node_predict_torchserve(ctx: Dict[str, Any], _node_cfg: Dict[str, Any]) -> None:
        model_response = _call_torchserve(
            settings.ts_base_url,
            settings.request_timeout,
            ctx['model_name'],
            ctx['features'],
        )
        ctx['model_response'] = model_response
        _save_debug_if_needed(ctx, 'model_result', model_response)

    def _post_action_inject_cur_price(ctx: Dict[str, Any], _node_cfg: Dict[str, Any]) -> None:
        raw_data = ctx.get('raw_data')
        if not raw_data:
            raise ValueError('No raw_data available for cur_price injection')

        cur_price = get_latest_close(raw_data)
        logger.info(f'print cur_price: {cur_price}')
        ctx['model_response']['cur_price'] = str(cur_price)

    def _post_action_debug_override_trigger(ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        if not ctx['req_debug_mode']:
            return

        filename = node_cfg.get('filename', 'trigger_long.json')
        trigger_file = os.path.join(settings.debug_output_dir, filename)
        if os.path.exists(trigger_file):
            with open(trigger_file, 'r', encoding='utf-8') as f:
                trigger_data = json.load(f)
            ctx['model_response'].update(trigger_data)
            logger.info(
                f"{ctx['pipeline_name']}: loaded trigger data from {trigger_file}: {trigger_data}"
            )

    def _post_action_ple_strategy(ctx: Dict[str, Any], _node_cfg: Dict[str, Any]) -> None:
        logger.info(f"{ctx['pipeline_name']}: call strategy with model_response: {ctx['model_response']}")
        PleStrategy(ctx['model_response']).execute()

    def _post_action_first_strategy(ctx: Dict[str, Any], _node_cfg: Dict[str, Any]) -> None:
        logger.info(f"{ctx['pipeline_name']}: call strategy with model_response: {ctx['model_response']}")
        FirstStrategy(ctx['model_response']).execute()

    def _post_action_feature_dump_sql(ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        if not settings.enable_feature_dump:
            return

        raw_data = ctx.get('raw_data')
        if raw_data is None:
            return

        if settings.offline_debug_mode and node_cfg.get('offline_tail_size'):
            tail_size = int(node_cfg['offline_tail_size'])
            ohlcv_snapshot = raw_data[-tail_size:] if len(raw_data) > tail_size else raw_data
        elif node_cfg.get('snapshot') == 'tail':
            tail_size = int(node_cfg.get('tail_size', 100))
            ohlcv_snapshot = raw_data[-tail_size:] if len(raw_data) > tail_size else raw_data
        else:
            ohlcv_snapshot = raw_data

        get_sql_dumper().dump_features(
            request_id=ctx['request_id'],
            timestamp=ctx['end_time'] or datetime.now(timezone.utc),
            symbol=node_cfg.get('symbol', settings.feature_dump_symbol),
            ohlcv_data=ohlcv_snapshot,
            features=ctx['features'],
            model_name=ctx['model_name'],
            model_output=ctx['model_response'],
        )
        logger.debug(f"Feature dump queued for request_id={ctx['request_id']}")

    def _post_action_feature_dump_hdf(ctx: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
        if not settings.enable_feature_dump:
            return

        raw_data = ctx.get('raw_data')
        if raw_data is None:
            return

        tail_size = int(node_cfg.get('tail_size', 100))
        ohlcv_snapshot = raw_data[-tail_size:] if len(raw_data) > tail_size else raw_data

        get_hdf_dumper().dump_features(
            request_id=ctx['request_id'],
            timestamp=ctx['end_time'] or datetime.now(timezone.utc),
            symbol=node_cfg.get('symbol', settings.feature_dump_symbol),
            ohlcv_data=ohlcv_snapshot,
            features=ctx['features'],
            model_name=ctx['model_name'],
            model_output=ctx['model_response'],
        )
        logger.debug(f"HDF5 feature dump queued for request_id={ctx['request_id']}")

    config = _load_pipeline_config(settings.pipeline_config_path)

    node_registry: Dict[str, Node] = {
        'hdf_file': FunctionNode(_node_retrieve_hdf),
        'binance_cached': FunctionNode(_node_retrieve_binance_cached),
        'binance_adaptor': FunctionNode(_node_adapt_binance),
        'sample': FunctionNode(_node_calculate_sample),
        'torchserve': FunctionNode(_node_predict_torchserve),
        'inject_cur_price': FunctionNode(_post_action_inject_cur_price),
        'debug_override_trigger': FunctionNode(_post_action_debug_override_trigger),
        'ple_strategy': FunctionNode(_post_action_ple_strategy),
        'first_strategy': FunctionNode(_post_action_first_strategy),
        'feature_dump_sql': FunctionNode(_post_action_feature_dump_sql),
        'feature_dump_hdf': FunctionNode(_post_action_feature_dump_hdf),
    }

    runner = PipelineRunner(
        pipeline_config=config,
        node_registry=node_registry,
        history_window=settings.history_window,
        default_model=settings.default_model,
    )
    return PipelineRuntime(config_path=settings.pipeline_config_path, config=config, runner=runner)
