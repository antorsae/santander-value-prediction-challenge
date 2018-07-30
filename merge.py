import pandas as pd
import numpy as np
df_test = pd.read_csv('baseline_submission_with_leaks_all_1000.csv')
sub = pd.read_csv('csv/sub_du_columngroups_drop_n20_dagilprst_m36mul2_sNone_rx_f5_b1_RMSE1.3335216-LB050.csv')
sub['target'] = sub.apply(lambda row: df_test.target[row.name] if df_test.target[row.name] > 0 else row['target'], axis=1)
sub.to_csv('sub_du_columngroups_drop_n20_dagilprst_m36mul2_sNone_rx_f5_b1_RMSE1.3335216-LB050-merged_42groups_leak.csv', index=False)

