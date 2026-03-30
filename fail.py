Analyse fehlgeschlagen: Fehlgeschlagen (Exit -11)

Details:
09:32:13 INFO [rubin.analysis] MLflow-Experiment 'rubin_WG_all' (identisch mit DataPrep).
09:32:13 INFO [rubin.analysis] Run-Name-Suffix 'kühler-wolf' aus DataPrep übernommen.
09:32:13 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/data/processed/dataprep_config.yml
09:32:13 INFO [rubin.analysis] [rubin] Step 1/9: Daten laden & Preprocessing
09:32:13 INFO [rubin.analysis] Historischer Score: 5 NaN-Werte durch 0 ersetzt.
09:32:15 INFO [rubin.analysis] Memory-Reduktion: 1085.7 MB → 826.8 MB (24% gespart).
09:32:15 INFO [rubin.analysis] Daten geladen: X=(389988, 696), T=(389988,) (unique=[0, 1]), Y=(389988,) (unique=[0, 1]), S=(389988,)
09:32:16 INFO [rubin.analysis] [rubin] Step 2/9: Feature-Selektion
09:32:16 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
09:37:37 INFO [rubin.feature_selection] CausalForest FS: X=(389988, 696) (dtypes: 696 numeric, 0 category), T=(389988,) (unique=2), Y=(389988,), n_jobs=-1, in_thread=False
09:37:39 INFO [rubin.feature_selection] CausalForest FS: Subsampling 389988 → 99999 Zeilen (stratifiziert nach T).
09:37:39 INFO [rubin.feature_selection] CausalForest FS: fit(99999×696, T unique=2, n_estimators=100, n_jobs=-1)...
09:50:04 INFO [rubin.feature_selection] Korrelationsfilter (|r| > 0.90, importance-gesteuert): 319 Features entfernt, 377 verbleiben.
09:50:04 INFO [rubin.analysis] Importance-Umverteilung: 319 entfernte Features → Importance auf Partner übertragen.
09:50:04 INFO [rubin.feature_selection] Feature-Selection 'lgbm_importance': Top-15% = 57 / 377 Features.
09:50:04 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 57 / 377 Features.
09:50:04 INFO [rubin.feature_selection] Feature-Selection Union: 78 / 377 Features behalten, 299 entfernt.
09:50:04 INFO [rubin.analysis] Feature-Selektion gesamt: 696 → 78 Features (Korrelation: −319, Importance: −299)
09:50:04 INFO [rubin.analysis] [rubin] Step 3/9: Base-Learner-Tuning
09:50:04 INFO [rubin.analysis] Starte Tuning: X=(389988, 78), Y=(389988,) (unique=[0, 1]), T=(389988,) (unique=[0, 1])
09:50:04 INFO [rubin.tuning] tune_all gestartet: models=['TLearner', 'DRLearner', 'NonParamDML', 'XLearner', 'ParamDML', 'CausalForestDML'], X=(389988, 78), Y=(389988,) (unique=[0, 1]), T=(389988,) (unique=[0, 1]), cv_splits=5, n_trials=80, parallel_trials=16
09:50:04 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 78), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
09:52:39 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__with_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 79), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
09:54:49 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=389988 rows, indices=389984, X_task=(389984, 78), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
09:56:55 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': X_input=389988 rows, indices=389988, X_task=(389988, 78), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
09:59:28 INFO [rubin.tuning] Tuning-Task 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X_input=389988 rows, indices=389984, X_task=(389984, 78), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=D, objective=pseudo_effect
