To test torchserve, run the following commands.
- For the first time:
    - ./init_env.sh
- On updating raw_models
    - ./re_deploy_models.sh
- To stop:
    - ./stop_torchserve.sh
- To do an inference, results are saved to `data/result.json`.
    - ./do_inference.sh
