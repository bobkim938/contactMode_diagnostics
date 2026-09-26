## TEST dataset analysis
To confirm whether the change has same direction as validation in all 5 seeds
- Models: seeds 0, 1, 2, 3, 42, both conditions (10 checkpoints, no retraining)
- Metric: episode-balanced $v_x$ RMSE, mm/s
- Regions: near recontact ($\pm$ 5 ms), shallow contact (0–1 mm, >20 ms from transitions), free flight (>20 ms from transitions)
- Windows: $\pm$ 5 ms primary; $\pm$ 2, 10, 20 ms sensitivity
