python -m cmds.generate_table_data    --dbname amazon    --task_name user-churn \
   --sample_size 100000      --dfs    --table_output_dir ./data/dfs-flatten-table --n_timedelta 1


python -m cmds.generate_table_data    --dbname amazon    --task_name item-churn \
   --sample_size 100000      --dfs    --table_output_dir ./data/dfs-flatten-table --n_timedelta 1


python -m cmds.generate_table_data    --dbname amazon    --task_name user-ltv \
   --sample_size 100000      --dfs    --table_output_dir ./data/dfs-flatten-table --n_timedelta 1


python -m cmds.generate_table_data    --dbname amazon    --task_name item-ltv \
   --sample_size 100000      --dfs    --table_output_dir ./data/dfs-flatten-table --n_timedelta 1