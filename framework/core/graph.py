import json
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx

from framework.core.constants import (
    GRAPH_DATA_COLUMN_NAME,
    ID_COLUMN_NAME,
    TABLE_NAME,
)


class GraphLoader(Protocol):
    def load(self) -> nx.Graph:
        """Returns the NetworkX graph loaded from the source."""
        ...

    def unload(self) -> None:
        """Cleans up any resources used by the loader."""
        ...


class InMemoryGraphLoader:
    _graph: nx.Graph

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def load(self) -> nx.Graph:
        return self._graph

    def unload(self) -> None:
        pass


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


@dataclass
class Feature:
    name: str
    value: Any


class FrameworkGraph:
    id: int
    features: dict[str, Any]
    _loader: GraphLoader

    def __init__(self, id: int, features: dict[str, Any], loader: GraphLoader):
        self.id = id
        self.features = features
        self._loader = loader

    def __enter__(self) -> nx.Graph:
        return self._loader.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._loader.unload()

    @classmethod
    def from_memory(
        cls, graph: nx.Graph, features: dict[str, Any] | None = None
    ) -> "FrameworkGraph":
        if features is None:
            features = {}
        loader = InMemoryGraphLoader(graph)
        return cls(id=-1, features=features, loader=loader)

    @classmethod
    def from_sqlite(
        cls, id: int, conn: sqlite3.Connection, features: dict[str, Any]
    ) -> "FrameworkGraph":
        loader = SQLiteGraphLoader(id, conn)
        return cls(id=id, features=features, loader=loader)


class Writer(Protocol):
    def update(self, graph: FrameworkGraph) -> None:
        """Updates the given FrameworkGraph in the storage."""
        ...

    def __enter__(self) -> "Writer":
        """Enters the context for writing."""
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exits the context for writing."""
        ...


class MemoryWriter:
    def update(self, graph: FrameworkGraph) -> None:
        pass

    def __enter__(self) -> "MemoryWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


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


class Dataset(Protocol):
    def __getitem__(self, index: int) -> FrameworkGraph:
        """Returns the FrameworkGraph at the specified index."""
        ...

    def __len__(self) -> int:
        """Returns the number of FrameworkGraphs in the dataset."""
        ...

    def __iter__(self):
        """Returns an iterator over the FrameworkGraphs in the dataset."""
        ...

    def append(self, graph: FrameworkGraph) -> None:
        """Appends a FrameworkGraph to the dataset."""
        ...

    def extend(self, graphs: list[FrameworkGraph]) -> None:
        """Extends the dataset with a list of FrameworkGraphs."""
        ...

    def writer(self) -> Writer:
        """Returns a Writer for updating graphs in the dataset."""
        ...


class MemoryDataset:
    _graphs: list[FrameworkGraph]

    def __init__(self):
        self._graphs = []

    def __getitem__(self, index: int) -> FrameworkGraph:
        return self._graphs[index]

    def __len__(self) -> int:
        return len(self._graphs)

    def __iter__(self):
        return iter(self._graphs)

    def append(self, graph: FrameworkGraph) -> None:
        self._graphs.append(graph)

    def extend(self, graphs: list[FrameworkGraph]) -> None:
        self._graphs.extend(graphs)

    def writer(self) -> Writer:
        return MemoryWriter()


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
            graph = FrameworkGraph.from_sqlite(id, self.conn, features)
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
        new_graph = FrameworkGraph.from_sqlite(new_graph_id, self.conn, graph.features)
        self.graphs.append(new_graph)

    def extend(self, graphs: list[FrameworkGraph]) -> None:
        for graph in graphs:
            self.append(graph)

    def writer(self) -> Writer:
        return SQLiteWriter(self.conn, self.writer_batch_size)
