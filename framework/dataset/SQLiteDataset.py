import json
import pickle
import sqlite3
from typing import Any, Iterator

import networkx as nx

from framework.core.graph import BatchWriter, FrameworkGraph, Update

TABLE_NAME = "graphs"
FEATURES_COLUMN_NAME = "features"
GRAPH_DATA_COLUMN_NAME = "data"
ID_COLUMN_NAME = "id"


class SQLiteGraphLoader:
    id: int
    conn: sqlite3.Connection
    _cached_graph: nx.Graph | None

    def __init__(self, id: int, conn: sqlite3.Connection):
        self.id = id
        self.conn = conn
        self._cached_graph = None

    def load(self) -> nx.Graph:
        if self._cached_graph is not None:
            return self._cached_graph

        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT {GRAPH_DATA_COLUMN_NAME} FROM {TABLE_NAME} WHERE {ID_COLUMN_NAME} = ?",
            (self.id,),
        )
        blob = cursor.fetchone()[0]
        self._cached_graph = pickle.loads(blob)
        return self._cached_graph  # type: ignore

    def unload(self) -> None:
        self._cached_graph = None


def create_sqlite_graph(
    id: int, conn: sqlite3.Connection, features: dict[str, Any]
) -> FrameworkGraph:
    loader = SQLiteGraphLoader(id, conn)
    return FrameworkGraph(id=id, features=features, loader=loader)


class SQLiteDataset:
    conn: sqlite3.Connection
    graphs: list[FrameworkGraph]
    writer_batch_size: int

    def __init__(self, conn: sqlite3.Connection, writer_batch_size: int = 1000):
        self.conn = conn
        self.writer_batch_size = writer_batch_size
        self.ensure_table_exists()
        self.graphs = self.load_graphs()

    def ensure_table_exists(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                {ID_COLUMN_NAME} INTEGER PRIMARY KEY,
                {FEATURES_COLUMN_NAME} TEXT,
                {GRAPH_DATA_COLUMN_NAME} BLOB
            )
            """
        )
        self.conn.commit()

    def load_graphs(self) -> list[FrameworkGraph]:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {ID_COLUMN_NAME}, features FROM {TABLE_NAME}")
        rows = cursor.fetchall()
        graphs = []
        for row in rows:
            id = row[0]
            features = json.loads(row[1]) if row[1] else {}
            graph = create_sqlite_graph(id, self.conn, features)
            graphs.append(graph)
        return graphs

    def __getitem__(self, index: int) -> FrameworkGraph:
        return self.graphs[index]

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self) -> Iterator[FrameworkGraph]:
        return iter(self.graphs)

    def writer(self, batch_size: int = 1000) -> BatchWriter:
        return BatchWriter(
            save_callback=self._batch_save_callback, batch_size=batch_size
        )

    def _batch_save_callback(self, updates: list[Update]) -> None:
        cursor = self.conn.cursor()

        add_updates = [u for u in updates if u.update_type == "add"]
        if add_updates:
            add_values = []
            for update in add_updates:
                features = json.dumps(update.graph.features)
                with update.graph as g:
                    graph_data = pickle.dumps(g)
                add_values.append((features, graph_data))
            cursor.executemany(
                f"INSERT INTO {TABLE_NAME} (features, {GRAPH_DATA_COLUMN_NAME}) VALUES (?, ?)",
                add_values,
            )

        feature_updates = [u for u in updates if u.update_type == "feature_update"]
        if feature_updates:
            feature_values = []
            for update in feature_updates:
                features = json.dumps(update.graph.features)
                feature_values.append((features, update.graph.id))
            cursor.executemany(
                f"UPDATE {TABLE_NAME} SET {FEATURES_COLUMN_NAME} = ? WHERE {ID_COLUMN_NAME} = ?",
                feature_values,
            )

        self.conn.commit()

    @classmethod
    def from_file(
        cls, file_path: str, writer_batch_size: int = 1000
    ) -> "SQLiteDataset":
        conn = sqlite3.connect(file_path)
        return cls(conn, writer_batch_size)
