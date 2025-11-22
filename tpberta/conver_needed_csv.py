from utils.data import DatabaseFactory

tasks = [
    ("hm", "user-churn", "rel-hm"),
    ("event", "user-repeat", "rel-event"),
    ("trial", "study-outcome", "rel-trial"),
    ("ratebeer", "user-active", "ratebeer"),
    ("avito", "user-clicks", "rel-avito"),

    ("hm", "item-sales", "rel-hm"),
    ("event", "user-attendance", "rel-event"),
    ("trial", "site-success", "rel-trial"),
    ("ratebeer", "beer-positive", "ratebeer"),
    ("avito", "ad-ctr", "rel-avito")
]

cache_dir_root = "/home/lingze/.cache/relbench/"
for ele in tasks:
    db_name = ele[0]
    task_name = ele[1]
    cache_dir_db_name = ele[2]
    cache_dir = cache_dir_root + cache_dir_db_name
    print(f"cache_dir={cache_dir} for {db_name}")

    db = DatabaseFactory.get_db(db_name,
                                cache_dir=cache_dir,
                                upto_test_timestamp=False,
                                with_text_compress=True)
    dataset = DatabaseFactory.get_dataset(db_name, cache_dir)
    task = DatabaseFactory.get_task(db_name, task_name, dataset)
    entity_table = task.entity_table
    used_df = db.table_dict[entity_table]
    #
    # process_csv_rows_to_embeddings(
    #     csv_rows=test_csv_rows,
    #     pretrain_dir=pretrain_dir,
    #     delimiter=";",
    #     device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    # )
