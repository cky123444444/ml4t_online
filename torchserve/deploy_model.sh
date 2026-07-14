torch-model-archiver \
  --model-name ple \
  --version 1.0 \
  --model-file ./model_files/ple.py \
  --serialized-file ./raw_models/ple_double_t_2025_12_31_w_log1p.pt \
  --handler ./model_files/default_handler.py \
  --extra-files ./model_files/common_models.py,./model_files/model_utils.py \
  --export-path model_store \
  --force

