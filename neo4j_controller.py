import time

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"

class Neo4jController():

    def __init__(self, database = "neo4j"):
        self.driver = GraphDatabase.driver(URI)
        self.database = database
        self._create_database_if_not_exists()

    def reset_db(self):
        query = "MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r"
        self.excecute_query(query)

    def excecute_query(self, query):
        return self.driver.execute_query(query, database_=self.database)

    def _create_database_if_not_exists(self):
        self.driver.execute_query(f"CREATE DATABASE {self.database} IF NOT EXISTS")
    
    def excecute_query_and_compare_with_label(self, query, label, expect_multiple_rows = False):
        records, _, keys = self.excecute_query(query)
        if not expect_multiple_rows and len(records) != 1:
            raise Exception(f"Expected record to contain exactly 1 row, but got {records} instead.")
        if expect_multiple_rows and len(records) < 1:
            raise Exception(f"Expected record to contain at least 1 row.")
        if len(keys) != 1:
            raise Exception(f"Expected record to contain exactly 1 key, but got {keys} instead.")
        if not expect_multiple_rows:
            record = records[0][keys[0]]
            if record != label:
                raise Exception(f"Expected record to be {label}, but got {record} instead.")
            return record
        else:
            records = list(map(lambda record: record[keys[0]], records))
            if label not in records:
                raise Exception(f"Expected {label} to be in {records}.")
            return label
