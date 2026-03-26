[rubin] Step 1/8: Daten laden & Preprocessing
[rubin] Step 2/8: Feature-Selektion
[rubin] Step 3/8: Base-Learner-Tuning
[rubin] Step 4/8: Training & Cross-Predictions
[rubin] Step 5/8: Evaluation & Metriken
[rubin] Step 6/8: Surrogate-Tree
[rubin] Step 7/8: Bundle-Export
[rubin] Step 8/8: HTML-Report
[rubin] Step 8/8: Fertig
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
17:39:16 INFO [rubin.analysis] DRTester Nuisance einmalig gefittet (BT, cv=3, n_est≤100). Wird für alle Modelle wiederverwendet.
17:39:16 INFO [rubin.analysis] Evaluation Predictions_XLearner: n=389988, min=-0.0362055, median=0.000625949, max=0.0409281, std=0.00137388, non-zero=389988/389988, unique=999
17:39:16 INFO [rubin.analysis] Evaluation Predictions_DRLearner: n=389988, min=0.000827544, median=0.00084219, max=0.000850949, std=8.14947e-06, non-zero=389988/389988, unique=5
17:39:17 INFO [rubin.analysis] Evaluation Predictions_NonParamDML: n=389988, min=0.000817894, median=0.000829547, max=0.000834671, std=6.78026e-06, non-zero=389988/389988, unique=5
17:39:18 INFO [rubin.analysis] Evaluation Predictions_TLearner: n=389988, min=-0.0565335, median=0.000499056, max=0.0484579, std=0.00243547, non-zero=389988/389988, unique=999
17:39:18 INFO [rubin.analysis] Evaluation Predictions_CausalForestDML: n=389988, min=-0.00547924, median=0.000720607, max=0.00860707, std=0.000398421, non-zero=389988/389988, unique=999
17:39:19 INFO [rubin.analysis] Metriken für 5 Modelle berechnet. Vorläufiger Champion: TLearner. Diagnostik-Plots: Champion + Challenger
18:05:17 INFO [rubin.analysis] [rubin] Step 6/8: Surrogate-Tree
18:05:17 INFO [rubin.analysis] Trainiere Surrogate auf CausalForestDML (immer, unabhängig von Champion).
18:05:17 INFO [rubin.analysis] CausalForestDML Rang: 3 → DRTester-Plots: nein
18:05:17 WARNING [rubin.analysis] SurrogateTree_CausalForestDML fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2170, in run
    fitted_tester=fitted_tester_bt,
                  ^^^^^^^^^^^^^^^^
NameError: name 'fitted_tester_bt' is not defined
18:05:17 INFO [rubin.analysis] Trainiere Surrogate auf Champion TLearner.
18:05:43 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.00034930760793360887, 'auuc': 0.0007775453479073333, 'uplift_at_10pct': 0.00048318486772785333, 'uplift_at_20pct': 0.0006370808417097568, 'uplift_at_50pct': 0.0008523636600969063, 'policy_value_treat_positive': 0.0009617831432560263}
18:14:21 INFO [rubin.analysis] RAM-Optimierung: gc.collect() nach Surrogate.
18:14:21 INFO [rubin.analysis] [rubin] Step 7/8: Bundle-Export
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
18:17:56 INFO [rubin.analysis] Surrogate-Einzelbaum exportiert (Typ=lgbm, Tiefe=None, Blätter=31, trainiert auf 389988 Zeilen).
18:17:56 INFO [rubin.analysis] RAM-Optimierung: Modelle, Predictions und X_full freigegeben.
18:17:56 INFO [rubin.analysis] [rubin] Step 8/8: HTML-Report
18:17:56 INFO [rubin.reporting] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
18:17:56 INFO [rubin.analysis] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
