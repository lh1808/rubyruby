[rubin] Step 1/8: Daten laden & Preprocessing
[rubin] Step 2/8: Feature-Selektion
[rubin] Step 3/8: Base-Learner-Tuning
[rubin] Step 4/8: Training & Cross-Predictions
[rubin] Step 5/8: Evaluation & Metriken
[rubin] Step 6/8: Surrogate-Tree
[rubin] Step 7/8: Bundle-Export
[rubin] Step 8/8: HTML-Report
[rubin] Step 8/8: Fertig
19:49:35 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/data/processed/dataprep_config.yml
19:49:35 INFO [rubin.analysis] [rubin] Step 1/8: Daten laden & Preprocessing
19:49:35 INFO [rubin.analysis] Historischer Score: 5 NaN-Werte durch 0 ersetzt.
19:49:37 INFO [rubin.analysis] Memory-Reduktion: 1085.7 MB → 826.8 MB (24% gespart).
19:49:37 INFO [rubin.analysis] Daten geladen: X=(389988, 696), T=(389988,) (unique=[0, 1]), Y=(389988,) (unique=[0, 1]), S=(389988,)
19:49:38 INFO [rubin.analysis] [rubin] Step 2/8: Feature-Selektion
19:49:38 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
19:53:53 INFO [rubin.feature_selection] CausalForest FS: X=(389988, 696) (dtypes: 696 numeric, 0 category), T=(389988,) (unique=2), Y=(389988,), n_jobs=-1, in_thread=False
19:53:54 INFO [rubin.feature_selection] CausalForest FS: Subsampling 389988 → 99999 Zeilen (stratifiziert nach T).
19:53:54 INFO [rubin.feature_selection] CausalForest FS: fit(99999×696, T unique=2, n_estimators=100, n_jobs=-1)...
19:54:24 INFO [rubin.feature_selection] Feature-Selection 'lgbm_importance': Top-15% = 105 / 696 Features.
19:54:24 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 105 / 696 Features.
19:54:24 INFO [rubin.feature_selection] Feature-Selection Union: 152 / 696 Features behalten, 544 entfernt.
19:55:19 INFO [rubin.analysis] [rubin] Step 3/8: Base-Learner-Tuning
19:55:19 INFO [rubin.analysis] Starte Tuning: X=(389988, 124), Y=(389988,) (unique=[0, 1]), T=(389988,) (unique=[0, 1])
19:55:19 INFO [rubin.tuning] tune_all gestartet: models=['SLearner', 'TLearner', 'DRLearner', 'NonParamDML', 'XLearner', 'ParamDML', 'CausalForestDML'], X=(389988, 124), Y=(389988,) (unique=[0, 1]), T=(389988,) (unique=[0, 1]), cv_splits=5, n_trials=80, parallel_trials=16
19:55:20 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
19:58:30 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__with_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 125), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
20:01:28 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__no_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
20:04:40 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=389988 rows, indices=389984, X_task=(389984, 124), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
20:07:33 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': X_input=389988 rows, indices=389988, X_task=(389988, 124), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
20:11:06 INFO [rubin.tuning] Tuning-Task 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X_input=389988 rows, indices=389984, X_task=(389984, 124), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=D, objective=pseudo_effect
20:40:36 INFO [rubin.analysis] [rubin] Step 4/8: Training & Cross-Predictions
20:40:36 INFO [rubin.training] SLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:40:53 INFO [rubin.analysis] Predictions_SLearner: CATE min=-0.0425264, median=0.000107347, max=0.0286913, std=0.000258557, unique=21318/389988, non-zero=389988/389988
20:40:54 INFO [rubin.training] TLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:41:23 INFO [rubin.analysis] Predictions_TLearner: CATE min=-0.420284, median=0.000595737, max=0.598723, std=0.00367522, unique=389448/389988, non-zero=389988/389988
20:41:23 INFO [rubin.analysis] DRLearner model_final effektive Params: {'iterations': 53, 'depth': 4, 'l2_leaf_reg': 65.42191762821476, 'rsm': 0.3251253072582496, 'min_data_in_leaf': 658, 'model_size_reg': 4.905426336169818} (explicit_tuned=ja, fmt_fixed=False)
20:41:24 INFO [rubin.training] DRLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:43:23 INFO [rubin.analysis] Predictions_DRLearner: CATE min=-0.00536922, median=0.000832813, max=0.00721606, std=0.000115331, unique=382352/389988, non-zero=389988/389988
20:43:23 INFO [rubin.analysis] NonParamDML model_final effektive Params: {'iterations': 152, 'depth': 4, 'l2_leaf_reg': 50.42098768426331, 'rsm': 0.5252183844922764, 'min_data_in_leaf': 235, 'model_size_reg': 0.13340590902918337} (explicit_tuned=ja, fmt_fixed=False)
20:43:23 INFO [rubin.training] NonParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:46:38 INFO [rubin.analysis] Predictions_NonParamDML: CATE min=-0.021193, median=0.000828244, max=0.0110057, std=0.000208021, unique=389244/389988, non-zero=389988/389988
20:46:39 INFO [rubin.training] XLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:47:42 INFO [rubin.analysis] Predictions_XLearner: CATE min=-0.209343, median=0.000720491, max=0.248351, std=0.00141115, unique=389448/389988, non-zero=389988/389988
20:47:43 INFO [rubin.training] ParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:50:58 INFO [rubin.analysis] Predictions_ParamDML: CATE min=-0.143624, median=0.000650767, max=0.202019, std=0.0031356, unique=389451/389988, non-zero=389988/389988
20:56:19 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
20:56:19 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
21:07:19 INFO [rubin.analysis] Predictions_CausalForestDML: CATE min=-0.000379461, median=0.000752343, max=0.0028939, std=0.000458208, unique=389448/389988, non-zero=389988/389988
21:07:19 INFO [rubin.analysis] [rubin] Step 5/8: Evaluation & Metriken
21:07:33 INFO [rubin.analysis] DRTester Nuisance einmalig gefittet (BT, cv=3, n_est≤100). Wird für alle Modelle wiederverwendet.
21:07:33 INFO [rubin.analysis] Evaluation Predictions_SLearner: n=389988, min=-0.0425264, median=0.000107347, max=0.0286913, std=0.000258557, non-zero=389988/389988, unique=999
21:07:33 INFO [rubin.analysis] Evaluation Predictions_TLearner: n=389988, min=-0.420284, median=0.000595737, max=0.598723, std=0.00367522, non-zero=389988/389988, unique=999
21:07:34 INFO [rubin.analysis] Evaluation Predictions_DRLearner: n=389988, min=-0.00536922, median=0.000832813, max=0.00721606, std=0.000115331, non-zero=389988/389988, unique=999
21:07:35 INFO [rubin.analysis] Evaluation Predictions_NonParamDML: n=389988, min=-0.021193, median=0.000828244, max=0.0110057, std=0.000208021, non-zero=389988/389988, unique=999
21:07:36 INFO [rubin.analysis] Evaluation Predictions_XLearner: n=389988, min=-0.209343, median=0.000720491, max=0.248351, std=0.00141115, non-zero=389988/389988, unique=999
21:07:36 INFO [rubin.analysis] Evaluation Predictions_ParamDML: n=389988, min=-0.143624, median=0.000650767, max=0.202019, std=0.0031356, non-zero=389988/389988, unique=999
21:07:37 INFO [rubin.analysis] Evaluation Predictions_CausalForestDML: n=389988, min=-0.000379461, median=0.000752343, max=0.0028939, std=0.000458208, non-zero=389988/389988, unique=999
21:07:38 INFO [rubin.analysis] Metriken für 7 Modelle berechnet. Vorläufiger Champion: ParamDML. Diagnostik-Plots: Champion + Challenger
21:07:38 WARNING [rubin.evaluation.drtester_plots] evaluate_cal fehlgeschlagen: Must fit nuisance models on training sample data to use calibration test
21:07:38 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(qini) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:07:38 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(toc) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:07:38 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
21:07:38 WARNING [rubin.evaluation.drtester_plots] DRTester plot_cal fehlgeschlagen: 'NoneType' object has no attribute 'plot_cal'
21:07:38 WARNING [rubin.evaluation.drtester_plots] DRTester plot_qini fehlgeschlagen: 'NoneType' object has no attribute 'plot_uplift'
21:07:38 WARNING [rubin.evaluation.drtester_plots] DRTester plot_toc fehlgeschlagen: 'NoneType' object has no attribute 'plot_uplift'
21:07:38 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:38 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:40 WARNING [rubin.evaluation.drtester_plots] evaluate_cal fehlgeschlagen: Must fit nuisance models on training sample data to use calibration test
21:07:40 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(qini) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:07:40 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(toc) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:07:40 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
21:07:40 WARNING [rubin.evaluation.drtester_plots] DRTester plot_cal fehlgeschlagen: 'NoneType' object has no attribute 'plot_cal'
21:07:40 WARNING [rubin.evaluation.drtester_plots] DRTester plot_qini fehlgeschlagen: 'NoneType' object has no attribute 'plot_uplift'
21:07:40 WARNING [rubin.evaluation.drtester_plots] DRTester plot_toc fehlgeschlagen: 'NoneType' object has no attribute 'plot_uplift'
21:07:40 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:40 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:42 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:42 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:43 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:43 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:45 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:45 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:46 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:46 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:48 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:48 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:50 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:50 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:07:51 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:07:51 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
/mnt/rubin/rubin/evaluation/drtester_plots.py:573: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
21:07:53 WARNING [rubin.evaluation.drtester_plots] evaluate_cal fehlgeschlagen: Must fit nuisance models on training sample data to use calibration test
21:07:53 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(qini) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:07:53 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(toc) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:07:53 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
21:07:53 WARNING [rubin.evaluation.drtester_plots] DRTester plot_cal fehlgeschlagen: 'NoneType' object has no attribute 'plot_cal'
21:07:53 WARNING [rubin.evaluation.drtester_plots] DRTester plot_qini fehlgeschlagen: 'NoneType' object has no attribute 'plot_uplift'
21:07:53 WARNING [rubin.evaluation.drtester_plots] DRTester plot_toc fehlgeschlagen: 'NoneType' object has no attribute 'plot_uplift'
21:07:53 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
/mnt/rubin/rubin/evaluation/drtester_plots.py:530: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
21:07:54 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
/mnt/rubin/rubin/pipelines/analysis_pipeline.py:1382: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(10, 6))
21:08:29 INFO [rubin.analysis] [rubin] Step 6/8: Surrogate-Tree
21:08:29 INFO [rubin.analysis] Trainiere Surrogate auf CausalForestDML (immer, unabhängig von Champion).
21:08:29 INFO [rubin.analysis] CausalForestDML Rang: 4 → DRTester-Plots: nein
21:08:31 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.00023688313894898032, 'auuc': 0.0006651208789227047, 'uplift_at_10pct': 0.0002229220860541558, 'uplift_at_20pct': 0.00040794414030472805, 'uplift_at_50pct': 0.000758173623850414, 'policy_value_treat_positive': 0.0008564754799530801}
/mnt/rubin/rubin/evaluation/drtester_plots.py:462: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4.5))
/mnt/rubin/rubin/evaluation/drtester_plots.py:610: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig_qini, ax_qini = plt.subplots(figsize=(10, 6))
21:08:31 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:08:31 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:08:32 INFO [rubin.analysis] Trainiere Surrogate auf Champion ParamDML.
21:08:34 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.00020175837100031406, 'auuc': 0.0006299961109740384, 'uplift_at_10pct': 0.0002762633693418934, 'uplift_at_20pct': 0.0004058339244152247, 'uplift_at_50pct': 0.0006329534043062781, 'policy_value_treat_positive': 0.0008930081685348443}
21:08:35 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen: plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
21:08:35 WARNING [rubin.evaluation.drtester_plots] sklift Treatment-Balance fehlgeschlagen: plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
21:08:36 INFO [rubin.analysis] RAM-Optimierung: gc.collect() nach Surrogate.
21:08:36 INFO [rubin.analysis] [rubin] Step 7/8: Bundle-Export
21:15:17 INFO [rubin.analysis] Surrogate-Einzelbaum exportiert (Typ=catboost, Tiefe=6, Blätter=64, trainiert auf 389988 Zeilen).
21:15:18 INFO [rubin.analysis] RAM-Optimierung: Modelle, Predictions und X_full freigegeben.
21:15:18 INFO [rubin.analysis] [rubin] Step 8/8: HTML-Report
21:15:18 INFO [rubin.reporting] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
21:15:18 INFO [rubin.analysis] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
