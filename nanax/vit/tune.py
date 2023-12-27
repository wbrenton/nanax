import optuna
from rich.pretty import pprint
from vit import Args, train

def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1028])
    embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
    depth = trial.suggest_categorical("depth", [1, 2, 3])

    args = Args(
        learning_rate=learning_rate,
        batch_size=batch_size,
        embed_dim=embed_dim,
        depth=depth,
    )

    return train(args)

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=42),
    direction='maximize'
    )
study.optimize(objective, n_trials=1000)