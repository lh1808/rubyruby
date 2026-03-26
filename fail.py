Ich kann in der Datenvorbereitung die Felder von Treatment-Spalte, Target-Spalte etc. nicht mehr manuell anpassen. Schau bitte, wo es da überall Probleme gibt und behebe diese.
Zusätzlich wird nun zwar mein Feature Dictionary erkannt, jedoch gibt es keinen Button mit dem ich dessen Ausführung und die Reduktion der Spalten und Anpassung der Datentypen vornehmen kann. Integriere das bitte und verknüpfe immer alles mit dem Backend.


Analyse fehlgeschlagen: Fehlgeschlagen (Exit 1)

Details:
15:09:36 INFO [rubin.analysis] [rubin] Step 1/8: Daten laden & Preprocessing
15:10:02 INFO [rubin.analysis] Memory-Reduktion: 2124.2 MB → 292.1 MB (86% gespart).
15:10:02 INFO [rubin.analysis] Daten geladen: X=(435995, 608), T=(435995,) (unique=[0, 1]), Y=(435995,) (unique=[0, 1]), S=None
15:10:03 INFO [rubin.analysis] [rubin] Step 2/8: Feature-Selektion
15:10:03 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
15:13:53 INFO [rubin.feature_selection] CausalForest FS: X=(435995, 608) (dtypes: 608 numeric, 0 category), T=(435995,) (unique=2), Y=(435995,), n_jobs=-1, in_thread=False
15:13:54 INFO [rubin.feature_selection] CausalForest FS: Subsampling 435995 → 99999 Zeilen (stratifiziert nach T).
15:13:54 INFO [rubin.feature_selection] CausalForest FS: fit(99999×608, T unique=2, n_estimators=100, n_jobs=-1)...
15:14:19 INFO [rubin.feature_selection] Feature-Selection 'lgbm_importance': Top-15% = 92 / 608 Features.
15:14:19 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 92 / 608 Features.
15:14:19 INFO [rubin.feature_selection] Feature-Selection Union: 132 / 608 Features behalten, 476 entfernt.
15:15:13 INFO [rubin.analysis] [rubin] Step 3/8: Base-Learner-Tuning
15:15:13 INFO [rubin.analysis] Starte Tuning: X=(435995, 116), Y=(435995,) (unique=[0, 1]), T=(435995,) (unique=[0, 1])
15:15:13 INFO [rubin.tuning] tune_all gestartet: models=['TLearner', 'DRLearner', 'NonParamDML', 'CausalForestDML'], X=(435995, 116), Y=(435995,) (unique=[0, 1]), T=(435995,) (unique=[0, 1]), cv_splits=5, n_trials=80, parallel_trials=16
15:15:13 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=435995 rows, indices=435995, X_task=(435995, 116), target=(435995,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
15:24:22 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__no_t__y': X_input=435995 rows, indices=435995, X_task=(435995, 116), target=(435995,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
15:30:01 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=435995 rows, indices=435990, X_task=(435990, 116), target=(435990,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
15:34:41 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': X_input=435995 rows, indices=435995, X_task=(435995, 116), target=(435995,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 117, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 113, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 1826, in run
    tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 333, in _run_tuning
    add = final_tuner.tune_final_model(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: FinalModelTuner.tune_final_model() missing 1 required positional argument: 'fmt_fixed_params'

Backend nicht erreichbar: Unexpected token '<', "<!doctype "... is not valid JSON



Excel file format cannot be determined, you must specify an engine manually.


Backend nicht erreichbar: Unexpected token 'N', ..."L":{"max":NaN,"mean""... is not valid JSON. Prüfe die Verbindung in der Sidebar.


Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl.


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
