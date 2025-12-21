import json
import pickle
import sqlite3
from typing import Any

import networkx as nx

from framework.core.graph import FrameworkGraph, Writer

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


class SQLiteWriter:
    conn: sqlite3.Connection
    batch_size: int
    updates: list

    def __init__(self, conn: sqlite3.Connection, batch_size: int = 1000):
        self.conn = conn
        self.batch_size = batch_size
        self.updates = []

    def update(self, graph: FrameworkGraph) -> None:
        features = json.dumps(graph.features)
        self.updates.append((features, graph.id))
        if len(self.updates) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        cursor = self.conn.cursor()
        cursor.executemany(
            f"UPDATE {TABLE_NAME} SET features = ? WHERE {ID_COLUMN_NAME} = ?",
            self.updates,
        )
        self.conn.commit()
        self.updates = []

    def __enter__(self) -> "SQLiteWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.updates:
            self._flush()


class SQLiteDataset:
    conn: sqlite3.Connection
    graphs: list[FrameworkGraph]
    writer_batch_size: int

    def __init__(self, conn: sqlite3.Connection, writer_batch_size: int = 1000):
        self.conn = conn
        self.writer_batch_size = writer_batch_size
        self.graphs = self.load_graphs()

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

    def __iter__(self):
        return iter(self.graphs)

    def append(self, graph: FrameworkGraph) -> None:
        cursor = self.conn.cursor()
        features = json.dumps(graph.features)
        graph_data = pickle.dumps(nx.Graph())
        response = cursor.execute(
            f"INSERT INTO {TABLE_NAME} (features, {GRAPH_DATA_COLUMN_NAME}) VALUES (?, ?) RETURNING {ID_COLUMN_NAME}",
            (features, graph_data),
        )
        new_graph_id = response.fetchone()[0]
        new_graph = create_sqlite_graph(new_graph_id, self.conn, graph.features)
        self.graphs.append(new_graph)

    def extend(self, graphs: list[FrameworkGraph]) -> None:
        for graph in graphs:
            self.append(graph)

    def writer(self) -> Writer:
        return SQLiteWriter(self.conn, self.writer_batch_size)
