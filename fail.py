/mnt/rubin/rubin/tuning_optuna.py:1037: ExperimentalWarning: Argument ``constant_liar`` is an experimental feature. The interface can change in the future.
  sampler = optuna.samplers.TPESampler(
20:16:15 INFO [rubin.analysis] [rubin] Step 4/8: Training & Cross-Predictions
20:16:15 INFO [rubin.training] TLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:16:34 INFO [rubin.analysis] Predictions_TLearner: CATE min=-0.353035, median=0.000605967, max=0.363495, std=0.00242156, unique=389448/389988, non-zero=389988/389988
20:16:34 INFO [rubin.analysis] DRLearner model_final effektive Params: {'iterations': 73, 'depth': 5, 'l2_leaf_reg': 85.19927721193956, 'rsm': 0.5113972499673406, 'min_data_in_leaf': 897, 'model_size_reg': 3.566326989678074} (explicit_tuned=ja, fmt_fixed=False)
20:16:35 INFO [rubin.training] DRLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:20:33 INFO [rubin.analysis] Predictions_DRLearner: CATE min=-0.00819597, median=0.000848075, max=0.00994161, std=0.000104333, unique=318937/389988, non-zero=389988/389988
20:20:33 INFO [rubin.analysis] NonParamDML model_final effektive Params: {'iterations': 80, 'depth': 5, 'l2_leaf_reg': 95.23860292728172, 'rsm': 0.4191820805173687, 'min_data_in_leaf': 217, 'model_size_reg': 14.827183377651869} (explicit_tuned=ja, fmt_fixed=False)
20:20:33 INFO [rubin.training] NonParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:23:54 INFO [rubin.analysis] Predictions_NonParamDML: CATE min=-0.00857699, median=0.00084059, max=0.00436814, std=6.77741e-05, unique=345370/389988, non-zero=389988/389988
20:23:54 INFO [rubin.training] XLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
20:25:06 INFO [rubin.analysis] Predictions_XLearner: CATE min=-0.284246, median=0.000711965, max=0.312233, std=0.00179129, unique=389448/389988, non-zero=389988/389988
20:30:30 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
20:30:30 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
20:41:41 INFO [rubin.analysis] Predictions_CausalForestDML: CATE min=-0.000470676, median=0.000753678, max=0.00283732, std=0.000465358, unique=389448/389988, non-zero=389988/389988
20:41:41 INFO [rubin.analysis] [rubin] Step 5/8: Evaluation & Metriken
20:42:16 INFO [rubin.analysis] DRTester Nuisance einmalig gefittet (BT, cv=3, n_est≤100). Wird für alle Modelle wiederverwendet.
20:42:16 INFO [rubin.analysis] Evaluation Predictions_TLearner: n=389988, min=-0.353035, median=0.000605967, max=0.363495, std=0.00242156, non-zero=389988/389988, unique=999
20:42:16 INFO [rubin.analysis] Evaluation Predictions_DRLearner: n=389988, min=-0.00819597, median=0.000848075, max=0.00994161, std=0.000104333, non-zero=389988/389988, unique=999
20:42:17 INFO [rubin.analysis] Evaluation Predictions_NonParamDML: n=389988, min=-0.00857699, median=0.00084059, max=0.00436814, std=6.77741e-05, non-zero=389988/389988, unique=999
20:42:17 INFO [rubin.analysis] Evaluation Predictions_XLearner: n=389988, min=-0.284246, median=0.000711965, max=0.312233, std=0.00179129, non-zero=389988/389988, unique=999
20:42:18 INFO [rubin.analysis] Evaluation Predictions_CausalForestDML: n=389988, min=-0.000470676, median=0.000753678, max=0.00283732, std=0.000465358, non-zero=389988/389988, unique=999
20:42:19 INFO [rubin.analysis] Metriken für 5 Modelle berechnet. Vorläufiger Champion: XLearner. Diagnostik-Plots: Champion + Challenger
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py:428: RuntimeWarning: divide by zero encountered in scalar divide
  cal_r_squared[k] = 1 - (cal_score_g / cal_score_o)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/utils.py:106: RuntimeWarning: invalid value encountered in divide
  mboot = (toc_psi / toc_std.reshape(-1, 1)) @ w / n
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py:581: RuntimeWarning: invalid value encountered in scalar divide
  pvals = [st.norm.sf(abs(q / e)) for q, e in zip(coeffs, errs)]
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py:428: RuntimeWarning: divide by zero encountered in scalar divide
  cal_r_squared[k] = 1 - (cal_score_g / cal_score_o)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/utils.py:106: RuntimeWarning: invalid value encountered in divide
  mboot = (toc_psi / toc_std.reshape(-1, 1)) @ w / n
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py:581: RuntimeWarning: invalid value encountered in scalar divide
  pvals = [st.norm.sf(abs(q / e)) for q, e in zip(coeffs, errs)]
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py:428: RuntimeWarning: divide by zero encountered in scalar divide
  cal_r_squared[k] = 1 - (cal_score_g / cal_score_o)
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/utils.py:106: RuntimeWarning: invalid value encountered in divide
  mboot = (toc_psi / toc_std.reshape(-1, 1)) @ w / n
/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py:581: RuntimeWarning: invalid value encountered in scalar divide
  pvals = [st.norm.sf(abs(q / e)) for q, e in zip(coeffs, errs)]
20:58:44 INFO [rubin.analysis] [rubin] Step 6/8: Surrogate-Tree
20:58:44 INFO [rubin.analysis] Trainiere Surrogate auf CausalForestDML (immer, unabhängig von Champion).
20:58:44 INFO [rubin.analysis] CausalForestDML Rang: 2 → DRTester-Plots: ja
20:58:47 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.0002200149791185733, 'auuc': 0.0006482527190922976, 'uplift_at_10pct': 0.0002038830275714333, 'uplift_at_20pct': 0.00041057887931692416, 'uplift_at_50pct': 0.0007419426538343781, 'policy_value_treat_positive': 0.0008564754799530801}
20:59:48 WARNING [rubin.evaluation.drtester_plots] evaluate_cal fehlgeschlagen: 'NoneType' object is not subscriptable
20:59:48 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(qini) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
20:59:48 WARNING [rubin.evaluation.drtester_plots] evaluate_uplift(toc) fehlgeschlagen: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
21:00:33 INFO [rubin.analysis] DRTester-Plots für SurrogateTree_CausalForestDML erzeugt.
21:00:33 INFO [rubin.analysis] Trainiere Surrogate auf Champion XLearner.
21:00:35 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.0001230329498002115, 'auuc': 0.0005512706897739359, 'uplift_at_10pct': 0.00014552400374057337, 'uplift_at_20pct': 0.0003254783062120586, 'uplift_at_50pct': 0.0005822565321207735, 'policy_value_treat_positive': 0.0008730632270598686}
21:03:01 INFO [rubin.analysis] RAM-Optimierung: gc.collect() nach Surrogate.
21:03:01 INFO [rubin.analysis] [rubin] Step 7/8: Bundle-Export
21:08:57 INFO [rubin.analysis] Surrogate-Einzelbaum exportiert (Typ=catboost, Tiefe=6, Blätter=64, trainiert auf 389988 Zeilen).
21:08:58 INFO [rubin.analysis] RAM-Optimierung: Modelle, Predictions und X_full freigegeben.
21:08:58 INFO [rubin.analysis] [rubin] Step 8/8: HTML-Report
21:08:58 INFO [rubin.reporting] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
21:08:58 INFO [rubin.analysis] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
