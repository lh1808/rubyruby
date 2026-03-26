○
HTML-Report
Pipeline-Logs
132 Zeilen
Ausblenden
[rubin] Step 1/8: Daten laden & Preprocessing
[rubin] Step 2/8: Feature-Selektion
[rubin] Step 3/8: Base-Learner-Tuning
[rubin] Step 4/8: Training & Cross-Predictions
[rubin] Step 5/8: Evaluation & Metriken
16:07:00 INFO [rubin.analysis] [rubin] Step 1/8: Daten laden & Preprocessing
16:07:00 INFO [rubin.analysis] Historischer Score: 5 NaN-Werte durch 0 ersetzt.
16:07:02 INFO [rubin.analysis] Memory-Reduktion: 1085.7 MB → 826.8 MB (24% gespart).
16:07:02 INFO [rubin.analysis] Daten geladen: X=(389988, 696), T=(389988,) (unique=[0, 1]), Y=(389988,) (unique=[0, 1]), S=(389988,)
16:07:02 INFO [rubin.analysis] [rubin] Step 2/8: Feature-Selektion
16:07:03 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
16:11:20 INFO [rubin.feature_selection] CausalForest FS: X=(389988, 696) (dtypes: 696 numeric, 0 category), T=(389988,) (unique=2), Y=(389988,), n_jobs=-1, in_thread=False
16:11:21 INFO [rubin.feature_selection] CausalForest FS: Subsampling 389988 → 99999 Zeilen (stratifiziert nach T).
16:11:21 INFO [rubin.feature_selection] CausalForest FS: fit(99999×696, T unique=2, n_estimators=100, n_jobs=-1)...
16:11:51 INFO [rubin.feature_selection] Feature-Selection 'lgbm_importance': Top-15% = 105 / 696 Features.
16:11:51 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 105 / 696 Features.
16:11:51 INFO [rubin.feature_selection] Feature-Selection Union: 152 / 696 Features behalten, 544 entfernt.
16:12:54 INFO [rubin.analysis] [rubin] Step 3/8: Base-Learner-Tuning
16:12:54 INFO [rubin.analysis] Starte Tuning: X=(389988, 124), Y=(389988,) (unique=[0, 1]), T=(389988,) (unique=[0, 1])
16:12:54 INFO [rubin.tuning] tune_all gestartet: models=['XLearner', 'DRLearner', 'NonParamDML', 'TLearner', 'CausalForestDML'], X=(389988, 124), Y=(389988,) (unique=[0, 1]), T=(389988,) (unique=[0, 1]), cv_splits=5, n_trials=80, parallel_trials=16
16:12:54 INFO [rubin.tuning] Tuning-Task 'lgbm__outcome__classifier__all__no_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
16:14:11 INFO [rubin.tuning] Tuning-Task 'lgbm__outcome_regression__regressor__all__no_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
16:15:05 INFO [rubin.tuning] Tuning-Task 'lgbm__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=389988 rows, indices=389984, X_task=(389984, 124), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
16:16:39 INFO [rubin.tuning] Tuning-Task 'lgbm__propensity__classifier__all__no_t__t': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
16:18:01 INFO [rubin.tuning] Tuning-Task 'lgbm__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X_input=389988 rows, indices=389984, X_task=(389984, 124), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=D, objective=pseudo_effect
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
16:39:53 INFO [rubin.analysis] [rubin] Step 4/8: Training & Cross-Predictions
16:39:53 INFO [rubin.training] XLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:40:46 INFO [rubin.analysis] Predictions_XLearner: CATE min=-0.0362055, median=0.000625949, max=0.0409281, std=0.00137388, unique=389431/389988, non-zero=389988/389988
16:40:46 INFO [rubin.analysis] DRLearner model_final effektive Params: {'n_estimators': 54, 'num_leaves': 21, 'max_depth': 2, 'min_child_samples': 303, 'min_child_weight': 3.07633272350389, 'colsample_bytree': 0.26842441331437217, 'reg_alpha': 1.0480719959544251, 'reg_lambda': 13.14811096117303, 'path_smooth': 21.6490543411484} (explicit_tuned=ja, fmt_fixed=False)
16:40:47 INFO [rubin.training] DRLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:43:09 WARNING [rubin.analysis] WARNUNG: Predictions_DRLearner hat nur 5 distinkte Werte bei 5 Folds (Range=2.34e-05, Mean=8.41e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
16:43:09 INFO [rubin.analysis] NonParamDML model_final effektive Params: {'n_estimators': 65, 'num_leaves': 6, 'max_depth': 5, 'min_child_samples': 965, 'min_child_weight': 6.449148722228649, 'colsample_bytree': 0.34537937044605627, 'reg_alpha': 44.69715087603946, 'reg_lambda': 3.9394402449744, 'path_smooth': 16.094041222406258} (explicit_tuned=ja, fmt_fixed=False)
16:43:09 INFO [rubin.training] NonParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
16:44:59 WARNING [rubin.analysis] WARNUNG: Predictions_NonParamDML hat nur 5 distinkte Werte bei 5 Folds (Range=1.68e-05, Mean=8.26e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
16:45:00 INFO [rubin.training] TLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:45:21 INFO [rubin.analysis] Predictions_TLearner: CATE min=-0.0565335, median=0.000499056, max=0.0484579, std=0.00243547, unique=375621/389988, non-zero=389988/389988
16:50:02 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
16:50:02 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
16:56:44 INFO [rubin.analysis] Predictions_CausalForestDML: CATE min=-0.00547924, median=0.000720607, max=0.00860707, std=0.000398421, unique=389444/389988, non-zero=389988/389988
16:56:44 INFO [rubin.analysis] [rubin] Step 5/8: Evaluation & Metriken


[rubin] Step 1/8: Daten laden & Preprocessing
[rubin] Step 2/8: Feature-Selektion
[rubin] Step 3/8: Base-Learner-Tuning
[rubin] Step 4/8: Training & Cross-Predictions
[rubin] Step 5/8: Evaluation & Metriken
[rubin] Step 6/8: Surrogate-Tree
[rubin] Step 7/8: Bundle-Export
[rubin] Step 8/8: HTML-Report
[rubin] Step 8/8: Fertig
07:12:15 INFO [rubin.analysis] [rubin] Step 1/8: Daten laden & Preprocessing
07:12:17 INFO [rubin.analysis] Memory-Reduktion: 1316.5 MB → 1002.5 MB (24% gespart).
07:12:17 INFO [rubin.analysis] Daten geladen: X=(472875, 696), T=(472875,) (unique=[0, 1]), Y=(472875,) (unique=[0, 1]), S=(472875,)
07:12:18 INFO [rubin.analysis] [rubin] Step 2/8: Feature-Selektion
07:12:18 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
07:16:23 INFO [rubin.feature_selection] CausalForest FS: X=(472875, 696) (dtypes: 696 numeric, 0 category), T=(472875,) (unique=2), Y=(472875,), n_jobs=-1, in_thread=False
07:16:25 INFO [rubin.feature_selection] CausalForest FS: Subsampling 472875 → 99999 Zeilen (stratifiziert nach T).
07:16:25 INFO [rubin.feature_selection] CausalForest FS: fit(99999×696, T unique=2, n_estimators=100, n_jobs=-1)...
07:16:53 INFO [rubin.feature_selection] Feature-Selection 'lgbm_importance': Top-15% = 105 / 696 Features.
07:16:53 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 105 / 696 Features.
07:16:53 INFO [rubin.feature_selection] Feature-Selection Union: 153 / 696 Features behalten, 543 entfernt.
07:18:10 INFO [rubin.analysis] [rubin] Step 3/8: Base-Learner-Tuning
07:18:10 INFO [rubin.analysis] Starte Tuning: X=(472875, 129), Y=(472875,) (unique=[0, 1]), T=(472875,) (unique=[0, 1])
07:18:10 INFO [rubin.tuning] tune_all gestartet: models=['TLearner', 'DRLearner', 'NonParamDML', 'CausalForestDML'], X=(472875, 129), Y=(472875,) (unique=[0, 1]), T=(472875,) (unique=[0, 1]), cv_splits=5, n_trials=80, parallel_trials=16
07:18:11 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=472875 rows, indices=472875, X_task=(472875, 129), target=(472875,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
07:22:26 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__no_t__y': X_input=472875 rows, indices=472875, X_task=(472875, 129), target=(472875,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
07:25:04 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=472875 rows, indices=389998, X_task=(389998, 129), target=(389998,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
07:27:34 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': X_input=472875 rows, indices=472875, X_task=(472875, 129), target=(472875,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
07:52:33 INFO [rubin.analysis] [rubin] Step 4/8: Training & Cross-Predictions
07:52:34 INFO [rubin.training] TLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
07:52:59 INFO [rubin.analysis] Predictions_TLearner: CATE min=-0.362981, median=0.000753776, max=0.253531, std=0.00260195, unique=472272/472875, non-zero=472875/472875
07:52:59 INFO [rubin.analysis] DRLearner model_final effektive Params: {'iterations': 50, 'min_data_in_leaf': 295} (explicit_tuned=ja, fmt_fixed=False)
07:52:59 INFO [rubin.training] DRLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
07:56:34 INFO [rubin.analysis] Predictions_DRLearner: CATE min=-1.13981, median=0.000674828, max=0.581783, std=0.006639, unique=461702/472875, non-zero=472875/472875
07:56:34 INFO [rubin.analysis] NonParamDML model_final effektive Params: {'iterations': 50, 'min_data_in_leaf': 366} (explicit_tuned=ja, fmt_fixed=False)
07:56:35 INFO [rubin.training] NonParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
08:01:14 INFO [rubin.analysis] Predictions_NonParamDML: CATE min=-0.715154, median=0.000663523, max=1.13144, std=0.00649879, unique=448886/472875, non-zero=472875/472875
08:08:10 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
08:08:10 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
08:23:12 INFO [rubin.analysis] Predictions_CausalForestDML: CATE min=-0.000455436, median=0.000761368, max=0.00271738, std=0.000404139, unique=472272/472875, non-zero=472875/472875
08:23:12 INFO [rubin.analysis] [rubin] Step 5/8: Evaluation & Metriken
08:23:38 INFO [rubin.analysis] DRTester Nuisance einmalig gefittet (BT, cv=3, n_est≤100). Wird für alle Modelle wiederverwendet.
08:23:38 INFO [rubin.analysis] Evaluation Predictions_TLearner: n=472875, min=-0.362981, median=0.000753776, max=0.253531, std=0.00260195, non-zero=472875/472875, unique=999
08:23:39 INFO [rubin.analysis] Evaluation Predictions_DRLearner: n=472875, min=-1.13981, median=0.000674828, max=0.581783, std=0.006639, non-zero=472875/472875, unique=999
08:23:40 INFO [rubin.analysis] Evaluation Predictions_NonParamDML: n=472875, min=-0.715154, median=0.000663523, max=1.13144, std=0.00649879, non-zero=472875/472875, unique=999
08:23:40 INFO [rubin.analysis] Evaluation Predictions_CausalForestDML: n=472875, min=-0.000455436, median=0.000761368, max=0.00271738, std=0.000404139, non-zero=472875/472875, unique=999
08:23:41 INFO [rubin.analysis] Metriken für 4 Modelle berechnet. Vorläufiger Champion: CausalForestDML. Diagnostik-Plots: Champion + Challenger
08:55:22 WARNING [rubin.evaluation.drtester_plots] DRTester evaluate_all fehlgeschlagen: exog contains inf or nans
Traceback (most recent call last):
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 644, in evaluate_cate_with_plots
    res = tester.evaluate_all(X_val.values, X_train.values if X_train is not None else None, n_groups=n_groups, n_bootstrap=n_bootstrap)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 269, in evaluate_all
    blp_res = self.evaluate_blp()
              ^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py", line 482, in evaluate_blp
    reg = OLS(self.dr_val_, add_constant(self.cate_preds_val_)).fit()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/regression/linear_model.py", line 921, in __init__
    super().__init__(endog, exog, missing=missing,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/regression/linear_model.py", line 746, in __init__
    super().__init__(endog, exog, missing=missing,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/regression/linear_model.py", line 200, in __init__
    super().__init__(endog, exog, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/model.py", line 270, in __init__
    super().__init__(endog, exog, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/model.py", line 95, in __init__
    self.data = self._handle_data(endog, exog, missing, hasconst,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/model.py", line 135, in _handle_data
    data = handle_data(endog, exog, missing, hasconst, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/data.py", line 694, in handle_data
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/data.py", line 90, in __init__
    self._handle_constant(hasconst)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/data.py", line 139, in _handle_constant
    raise MissingDataError("exog contains inf or nans")
statsmodels.tools.sm_exceptions.MissingDataError: exog contains inf or nans
09:05:02 WARNING [rubin.analysis] DRTester/SkLift-Plots für historischen Score fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 1133, in _evaluate_historical_score
    _log_temp_artifact(mlflow, lambda p, _b=bundle_h: save_dataframe_as_png(_b.summary, p), f"summary__{hist_name}.png")
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 88, in _log_temp_artifact
    content_fn(path)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 1133, in <lambda>
    _log_temp_artifact(mlflow, lambda p, _b=bundle_h: save_dataframe_as_png(_b.summary, p), f"summary__{hist_name}.png")
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 96, in save_dataframe_as_png
    tbl = table(ax, df, loc="center", cellLoc="center", colWidths=[0.1] * len(df.columns))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/plotting/_misc.py", line 62, in table
    return plot_backend.table(
           ^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/plotting/_matplotlib/tools.py", line 83, in table
    return matplotlib.table.table(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/matplotlib/table.py", line 761, in table
    cols = len(cellText[0])
               ~~~~~~~~^^^
IndexError: index 0 is out of bounds for axis 0 with size 0
09:05:02 WARNING [rubin.analysis] Policy-Value-Vergleich fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 1178, in _evaluate_historical_score
    figs = policy_value_comparison_plots(policy_values_dict, comparison_model_name=hist_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 850, in policy_value_comparison_plots
    ref_values = ref["policy_value"]
                 ~~~^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/core/indexes/range.py", line 417, in get_loc
    raise KeyError(key)
KeyError: 'policy_value'
09:05:03 INFO [rubin.analysis] [rubin] Step 6/8: Surrogate-Tree
09:05:06 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.00022390792849648504, 'auuc': 0.0007465267284536511, 'uplift_at_10pct': 0.00023851643273150967, 'uplift_at_20pct': 0.00044080402691202647, 'uplift_at_50pct': 0.0008635323716823421, 'policy_value_treat_positive': 0.0010452375999190067}
09:13:48 INFO [rubin.analysis] RAM-Optimierung: gc.collect() nach Surrogate.
09:13:48 INFO [rubin.analysis] [rubin] Step 7/8: Bundle-Export
09:20:36 INFO [rubin.analysis] Surrogate-Einzelbaum exportiert (Typ=catboost, Tiefe=6, Blätter=64, trainiert auf 472875 Zeilen).
09:20:36 INFO [rubin.analysis] RAM-Optimierung: Modelle, Predictions und X_full freigegeben.
09:20:36 INFO [rubin.analysis] [rubin] Step 8/8: HTML-Report
09:20:36 INFO [rubin.reporting] HTML-Report geschrieben: output/analysis_report.html


              ^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py", line 482, in evaluate_blp
    reg = OLS(self.dr_val_, add_constant(self.cate_preds_val_)).fit()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/regression/linear_model.py", line 921, in __init__
    super().__init__(endog, exog, missing=missing,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/regression/linear_model.py", line 746, in __init__
    super().__init__(endog, exog, missing=missing,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/regression/linear_model.py", line 200, in __init__
    super().__init__(endog, exog, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/model.py", line 270, in __init__
    super().__init__(endog, exog, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/model.py", line 95, in __init__
    self.data = self._handle_data(endog, exog, missing, hasconst,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/model.py", line 135, in _handle_data
    data = handle_data(endog, exog, missing, hasconst, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/data.py", line 694, in handle_data
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/data.py", line 90, in __init__
    self._handle_constant(hasconst)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/statsmodels/base/data.py", line 139, in _handle_constant
    raise MissingDataError("exog contains inf or nans")
statsmodels.tools.sm_exceptions.MissingDataError: exog contains inf or nans
