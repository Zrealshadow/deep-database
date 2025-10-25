python -m cmds.generate_table_data    --dbname f1    --task_name driver-dnf \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname f1    --task_name driver-f1 \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname avito    --task_name user-clicks \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data


python -m cmds.generate_table_data    --dbname ratebeer    --task_name user-active \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname ratebeer    --task_name place-positive \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname trial    --task_name site-success \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname trial    --task_name study-adverse \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname event    --task_name user-attendance \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data

python -m cmds.generate_table_data    --dbname ratebeer    --task_name beer-positive \
   --sample_size 100000      --dfs     --selection     --table_output_dir ./data/dfs-fs-data


python -m cmds.generate_table_data    --dbname hm    --task_name user-churn\
   --sample_size 100000      --dfs     --selection   --n_timedelta 4  --table_output_dir ./data/dfs-fs-data 

python -m cmds.generate_table_data    --dbname hm    --task_name item-sales\
   --sample_size 100000      --dfs     --selection   --n_timedelta 4  --table_output_dir ./data/dfs-fs-data 
