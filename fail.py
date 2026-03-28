14:10:31 WARNING [rubin.evaluation.drtester_plots] DRTester evaluate_all fehlgeschlagen: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Traceback (most recent call last):
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 804, in evaluate_cate_with_plots
    policy_values = res.get_policy_values(1)
                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 177, in get_policy_values
    if not self.qini.curves or not self.qini.treatments:
                                   ^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
