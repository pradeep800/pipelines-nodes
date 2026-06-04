"""Typed models for the view catalog returned by the views endpoint.

A :class:`ViewList` is a sequence of :class:`View` objects (so you can iterate
it or index it like a list) that also carries the caller's ``username`` and
``tenant_schema``. Each :class:`View` exposes its :class:`ViewColumn` s, and
each column reports whether it is ``accessible`` to the caller.

Use :meth:`ViewList.to_frame` (or :meth:`View.to_frame`) to flatten the catalog
into a pandas DataFrame for easy browsing and filtering in a notebook.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ViewColumn:
    """A single column of a view.

    Attributes:
        name: Column name.
        type: Logical column type (e.g. ``integer``, ``string``, ``datetime``).
        description: Human-readable description of the column.
        accessible: Whether the caller is permitted to read this column.
    """

    name: str
    type: str
    description: str
    accessible: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewColumn:
        return cls(
            name=data.get("name") or "",
            type=data.get("type") or "",
            description=data.get("description") or "",
            accessible=bool(data.get("accessible", False)),
        )


@dataclass(frozen=True)
class View:
    """A view (table) the caller can see, with its columns.

    Attributes:
        name: View name (e.g. ``person``).
        description: Human-readable description of the view.
        category: Grouping/category for the view (e.g. ``Clinical``).
        columns: The view's columns.
    """

    name: str
    description: str
    category: str
    columns: list[ViewColumn]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> View:
        return cls(
            name=data.get("name") or "",
            description=data.get("description") or "",
            category=data.get("category") or "",
            columns=[ViewColumn.from_dict(c) for c in (data.get("columns") or []) if c is not None],
        )

    @property
    def accessible_columns(self) -> list[ViewColumn]:
        """The subset of columns the caller is permitted to read."""
        return [c for c in self.columns if c.accessible]

    def to_frame(self) -> pd.DataFrame:
        """Return this view's columns as a DataFrame, one row per column."""
        rows = [
            {
                "column": c.name,
                "type": c.type,
                "description": c.description,
                "accessible": c.accessible,
            }
            for c in self.columns
        ]
        return pd.DataFrame(rows, columns=["column", "type", "description", "accessible"])


@dataclass(frozen=True)
class ViewList:
    """The catalog of views available to the caller.

    Behaves like a sequence of :class:`View` (iterate or index it) and also
    carries the caller's ``username`` and ``tenant_schema``.

    Attributes:
        username: The authenticated username the catalog was built for.
        tenant_schema: The caller's tenant schema name.
        views: The available views.
    """

    username: str
    tenant_schema: str
    views: list[View]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewList:
        return cls(
            username=data.get("username") or "",
            tenant_schema=data.get("tenant_schema") or "",
            views=[View.from_dict(v) for v in (data.get("views") or []) if v is not None],
        )

    @property
    def names(self) -> list[str]:
        """The names of all available views."""
        return [v.name for v in self.views]

    def get(self, name: str) -> View | None:
        """Return the view with this name, or ``None`` if there isn't one."""
        for v in self.views:
            if v.name == name:
                return v
        return None

    def to_frame(self) -> pd.DataFrame:
        """Flatten every view's columns into one DataFrame, one row per column.

        Columns: ``view``, ``category``, ``column``, ``type``, ``description``,
        ``accessible``. Handy for browsing in a notebook, e.g. filter the
        readable columns with ``client.list_views().to_frame().query("accessible")``.
        """
        rows = [
            {
                "view": view.name,
                "category": view.category,
                "column": col.name,
                "type": col.type,
                "description": col.description,
                "accessible": col.accessible,
            }
            for view in self.views
            for col in view.columns
        ]
        return pd.DataFrame(
            rows,
            columns=["view", "category", "column", "type", "description", "accessible"],
        )

    def __iter__(self) -> Iterator[View]:
        return iter(self.views)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> View:
        return self.views[index]
