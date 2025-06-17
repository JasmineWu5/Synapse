import argparse
import time
import os
import glob

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from core.config import ConfigManager
from core.logger import EnhancedLogger
from core.model_module import ModelModule, SaveTestOutputs
from core.data_module import DataModule


def main(_args: argparse.Namespace):
    config_manager = ConfigManager(
        data_cfg_file=_args.data_config,
        model_cfg_file=_args.model_config,
        run_cfg_file=_args.run_config
    )
    data_config = config_manager.data
    model_config = config_manager.model
    run_config = config_manager.run

    os.makedirs(run_config.run_dir, exist_ok=True)

    # Initialize the logger
    logger_config = run_config.logger_config

    def update_file_path(log_file_path: str, replace_auto: str = "", suffix: str="") -> str:
        dirname = os.path.dirname(log_file_path)
        if dirname:
            if os.path.isabs(dirname):
                log_file_path = log_file_path
            else:
                log_file_path = os.path.join(run_config.run_dir, log_file_path)
        else:
            log_file_path = os.path.join(run_config.run_dir, log_file_path)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        if '{auto}' in log_file_path:
            if replace_auto == "":
                replace_auto = time.strftime('%Y%m%d_%H%M%S')
            log_file_path = log_file_path.replace('{auto}', replace_auto + f'_{suffix}')
        return log_file_path

    run_info_str = (f"{time.strftime('%Y%m%d_%H%M%S')}_{model_config.model.split('.')[-1]}"
                    f"_epochs{run_config.epochs}_start{run_config.start_epoch}"
                    f"_bs{run_config.batch_size}_lr{model_config.start_lr}"
                    f"_opt_{model_config.optimizer}_lrSche_{model_config.lr_scheduler}")

    if logger_config.get('log_file'):
        logger_config['log_file'] = update_file_path(logger_config['log_file'], run_info_str, 'info')
    if logger_config.get('debug_file'):
        logger_config['debug_file'] = update_file_path(logger_config['debug_file'], run_info_str, 'debug')

    _logger = EnhancedLogger.from_config(logger_config).get_logger()

    _logger.info("Starting Synapse...")

    deterministic = False
    if run_config.seed:
        L.seed_everything(run_config.seed, workers=True, verbose=False)
        _logger.info(f"Set random seed to {run_config.seed}")
        deterministic = True

    tb_logger = TensorBoardLogger(save_dir=run_config.run_dir, name=f"TensorBoardLogs_{run_info_str}")

    trainer_callbacks = []
    if run_config.get("test_output"):
        test_output = update_file_path(run_config.test_output, run_info_str)
        test_output_callback = SaveTestOutputs(
            data_cfg=data_config,
            model_cfg=model_config,
            run_cfg=run_config,
            output_filepath=test_output
        )
        trainer_callbacks.append(test_output_callback)

    model = ModelModule(
        run_cfg=run_config,
        model_class=model_config.model,
        model_params=model_config.model_params,
        loss_fn=model_config.loss_function['name'],
        loss_params=model_config.loss_function['params'],
        optimizer=model_config.optimizer,
        start_lr=model_config.start_lr,
        lr_scheduler=model_config.lr_scheduler,
        metrics=model_config.metrics,
    )
    # TODO: customized checkpoint callback (save ckpt to another place, not tb logger dir), model_summary callback.
    trainer = L.Trainer(
        accelerator=run_config.device,
        devices=run_config.n_devices,
        deterministic=deterministic,
        default_root_dir=run_config.run_dir,
        enable_checkpointing=True,
        max_epochs=run_config.epochs,
        logger=tb_logger,
        precision= '16-mixed' if run_config.use_amp else '32-true',
        enable_progress_bar=True,
        enable_model_summary=True,
        inference_mode=True,
        callbacks= trainer_callbacks,
    )

    # TODO: cross-validation support
    train_file_paths = []
    for file_path in data_config.train_files:
        train_file_paths.extend(glob.glob(file_path))
    val_file_paths = []
    for file_path in data_config.val_files:
        val_file_paths.extend(glob.glob(file_path))
    test_file_paths = []
    for file_path in data_config.test_files:
        test_file_paths.extend(glob.glob(file_path))

    data_module = DataModule(
        data_cfg=data_config,
        run_cfg=run_config,
        train_file_list=train_file_paths,
        val_file_list=val_file_paths,
        test_file_list=test_file_paths
    )

    if run_config.run_mode == 'train':
        _logger.info("Running in training mode...")
        data_module.setup('fit')
        trainer.fit(model=model, datamodule=data_module)
    elif run_config.run_mode == 'test':
        _logger.info("Running in test mode...")
        trainer.test(model=model, datamodule=data_module)
    #TODO: checkpoint loading support


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Synapse")
    parser.add_argument('--data_config', type=str, required=True, help='Data configuration file path')
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration file path')
    parser.add_argument('--run_config', type=str, required=True, help='Run configuration file path')

    args = parser.parse_args()
    main(args)