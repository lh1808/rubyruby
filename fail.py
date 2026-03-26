rface can change in the future.
  sampler = optuna.samplers.TPESampler(
19:38:11 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=389988 rows, indices=389984, X_task=(389984, 124), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
/mnt/rubin/rubin/tuning_optuna.py:372: ExperimentalWarning: Argument ``multivariate`` is an experimental feature. The interface can change in the future.
  sampler = optuna.samplers.TPESampler(
/mnt/rubin/rubin/tuning_optuna.py:372: ExperimentalWarning: Argument ``constant_liar`` is an experimental feature. The interface can change in the future.
  sampler = optuna.samplers.TPESampler(
19:40:51 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
/mnt/rubin/rubin/tuning_optuna.py:372: ExperimentalWarning: Argument ``multivariate`` is an experimental feature. The interface can change in the future.
  sampler = optuna.samplers.TPESampler(
/mnt/rubin/rubin/tuning_optuna.py:372: ExperimentalWarning: Argument ``constant_liar`` is an experimental feature. The interface can change in the future.
  sampler = optuna.samplers.TPESampler(
