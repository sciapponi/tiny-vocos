from vocos.experiment import VocosExp
from vocos.dataset import VocosDataModule, DataConfig
from vocos.models import VocosBackbone, XiVocosBackboneFixedChannels, PhiBackbone
from vocos.heads import ISTFTHead
from vocos.feature_extractors import MelSpectrogramFeatures
from torch import nn
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import CSVLogger

EPOCHS = 5
BATCHSIZE = 16

def get_size_MB(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb


def objective(trial:optuna.trial.Trial):
    
    # FIXED PARAMETERS:
    hop_length = 256
    sample_rate=24000
    # SELECTION:
    # Backbone Type
    backbone_type = trial.suggest_categorical(name="backone_type", choices=["vocos", "xinet", "phinet"])
    # Hidden Dim (Only doing 1 Hidden layer for vocos since it saves up a lot of memory/computation)
    hidden_dim = trial.suggest_categorical(name="hidden_dim", choices=[64,128,256,512])
    # n_fft
    n_fft = trial.suggest_categorical(name="n_fft", choices=[512,1024])
    # learning rate
    lr = trial.suggest_uniform("lr", 1e-5, 1e-2) #log uniform

    # Backbone Selection + number of layers
    if backbone_type == "vocos":
        num_layers = trial.suggest_int('num_layers', 1, 4)
        backbone = VocosBackbone(input_channels=100, dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=num_layers, linear=False)
    elif backbone_type == "phinet":
        num_layers = trial.suggest_int('num_layers', 1, 8)
        backbone = PhiBackbone(num_layers=num_layers, freqs=100, dim=hidden_dim)
    elif backbone_type == "xinet":
        num_layers = trial.suggest_int('num_layers', 1, 8)
        backbone = XiVocosBackboneFixedChannels(num_layers=num_layers, freqs=100, dim=hidden_dim)

    
    # ISTFT Head
    head = ISTFTHead(dim=hidden_dim, 
                     n_fft=n_fft,
                     hop_length=hop_length)
    
    if get_size_MB(nn.Sequential(backbone,head)) > 1.6:
        return 0
    
    #Feature Extractor
    feature_extractor = MelSpectrogramFeatures(n_fft=n_fft)


    #Model Creation
    model = VocosExp(feature_extractor=feature_extractor,
                     backbone=backbone,
                     head=head,
                     sample_rate=sample_rate,
                     initial_learning_rate=lr,
                     evaluate_pesq=True
                     )
    
    # DATA
    trainDataConfig = DataConfig(filelist_path="ljfilelist.train",
                            sampling_rate=24000,
                            num_samples=16384,
                            batch_size=16,
                            num_workers=8)
    valDataConfig = DataConfig(filelist_path="ljfilelist.val",
                            sampling_rate=24000,
                            num_samples=16384,
                            batch_size=16,
                            num_workers=8)
    datamodule =  VocosDataModule(train_params=trainDataConfig, val_params=valDataConfig) 
    
    #LOGGER
    logger = CSVLogger("hp_logs", name=f"{backbone_type}_{hidden_dim}_{n_fft}_{num_layers}_{lr}")
    # TRAINER
    trainer = pl.Trainer(
                        logger=logger,
                        # limit_val_batches=PERCENT_VALID_EXAMPLES,
                        enable_checkpointing=False,
                        max_epochs=EPOCHS,
                        accelerator="auto",
                        devices=4,
                        callbacks=[PyTorchLightningPruningCallback(trial, monitor="pesq_score")],
                        )
    
    
    
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["pesq_score"].item()


if __name__== "__main__":
    pruner = optuna.pruners.MedianPruner() #if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=192)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))