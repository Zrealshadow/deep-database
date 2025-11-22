from utils.data import DatabaseFactory

db_name = "event"
db = DatabaseFactory.get_db(db_name,
                            upto_test_timestamp=False,
                            with_text_compress=True)
dataset = DatabaseFactory.get_dataset(db_name)

cache_dir = "/home/lingze/embedding_fusion/data/rel-event-tensor-frame"

task_name = "user-repeat"
task = DatabaseFactory.get_task(db_name, task_name, dataset)

entity_table = task.entity_table

print(db.table_dict[entity_table])
