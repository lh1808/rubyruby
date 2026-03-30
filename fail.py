12:53:26 INFO [rubin.analysis] [rubin] Step 8/9: Explainability
12:53:26 INFO [rubin.analysis] Explainability für Champion 'ParamDML' (max 10000 Samples).
12:53:26 WARNING [rubin.analysis] Explainability-Schritt fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2410, in run
    self._run_explainability(cfg, X, models, eval_summary, mlflow, report)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 1915, in _run_explainability
    raw_uplift = np.asarray(_predict_effect(model, X_expl))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 45, in _predict_effect
    pred = model.const_marginal_effect(X)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 998, in const_marginal_effect
    self._check_fitted_dims(X)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 650, in _check_fitted_dims
    assert self._d_x == X.shape[1:], "Dimension mis-match of X with fitted X"
           ^^^^^^^^^
AttributeError: 'LinearDML' object has no attribute '_d_x'
