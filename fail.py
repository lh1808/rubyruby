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
