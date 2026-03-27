22:22:25 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
22:23:10 INFO [rubin.analysis] Predictions_CausalForestDML: CATE min=-0.0322499, median=2.48663e-05, max=0.0632119, std=0.00476095, unique=38957/38999, non-zero=38999/38999
22:23:10 INFO [rubin.analysis] [rubin] Step 5/7: Evaluation & Metriken
22:26:45 INFO [rubin.analysis] DRTester Nuisance einmalig gefittet (BT, cv=3, n_est≤100). Wird für alle Modelle wiederverwendet.
22:26:45 INFO [rubin.analysis] Evaluation Predictions_TLearner: n=38999, min=-0.0244167, median=0.000803286, max=0.00650789, std=0.00141861, non-zero=38999/38999, unique=241
22:26:45 INFO [rubin.analysis] Evaluation Predictions_NonParamDML: n=38999, min=-0.534393, median=0.000124924, max=0.415577, std=0.0206684, non-zero=38999/38999, unique=999
22:26:46 INFO [rubin.analysis] Evaluation Predictions_XLearner: n=38999, min=-0.00864352, median=0.000949606, max=0.00102796, std=0.000328094, non-zero=38999/38999, unique=999
22:26:46 INFO [rubin.analysis] Evaluation Predictions_ParamDML: n=38999, min=-0.113158, median=0.000500625, max=0.176208, std=0.00876847, non-zero=38999/38999, unique=999
22:26:47 INFO [rubin.analysis] Evaluation Predictions_CausalForestDML: n=38999, min=-0.0322499, median=2.48663e-05, max=0.0632119, std=0.00476095, non-zero=38999/38999, unique=999
22:26:47 INFO [rubin.analysis] Metriken für 5 Modelle berechnet. Vorläufiger Champion: ParamDML. Diagnostik-Plots: Champion + Challenger
22:26:47 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
22:26:47 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:49 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
22:26:49 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:50 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:50 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:51 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:52 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:53 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:54 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
22:26:55 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:26:59 INFO [rubin.analysis] [rubin] Step 6/7: Surrogate-Tree
22:26:59 INFO [rubin.analysis] Trainiere Surrogate auf CausalForestDML (immer, unabhängig von Champion).
22:26:59 INFO [rubin.analysis] CausalForestDML Rang: 2 → DRTester-Plots: ja
22:27:22 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.0007632861789515562, 'auuc': 0.00124623167751254, 'uplift_at_10pct': 0.0012058783329094863, 'uplift_at_20pct': 0.0012808507807255149, 'uplift_at_50pct': 0.0014073056371907706, 'policy_value_treat_positive': 0.0018320434521013112}
22:27:22 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:27:23 WARNING [rubin.evaluation.drtester_plots] DRTester summary() fehlgeschlagen: 'NoneType' object has no attribute 'summary'
22:27:23 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:27:23 INFO [rubin.analysis] DRTester-Plots für SurrogateTree_CausalForestDML erzeugt.
22:27:23 INFO [rubin.analysis] Trainiere Surrogate auf Champion ParamDML.
22:27:46 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.0007669850614037948, 'auuc': 0.0012499305599647788, 'uplift_at_10pct': 0.0010424477058349495, 'uplift_at_20pct': 0.0012659850816265786, 'uplift_at_50pct': 0.0012734174885099948, 'policy_value_treat_positive': 0.0026064539017760266}
22:27:46 WARNING [rubin.evaluation.drtester_plots] sklift Uplift-by-Percentile fehlgeschlagen, nutze Fallback: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6,) + inhomogeneous part.
22:27:48 INFO [rubin.analysis] RAM-Optimierung: gc.collect() nach Surrogate.
22:27:48 INFO [rubin.analysis] RAM-Optimierung: Modelle, Predictions und X_full freigegeben.
22:27:48 INFO [rubin.analysis] [rubin] Step 7/7: HTML-Report
22:27:48 INFO [rubin.reporting] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
22:27:48 INFO [rubin.analysis] HTML-Report geschrieben: /mnt/rubin/.rubin_cache/analysis_report.html
