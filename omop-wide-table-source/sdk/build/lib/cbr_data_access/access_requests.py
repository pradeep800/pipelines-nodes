"""Typed models for the access-request catalog returned by the requests endpoint.

Access to data is granted per **access request**: an approved request unlocks a
set of cohorts and concepts, and the server exposes one queryable relation per
(request, table) named ``req_<request_id>.<table>``. A :class:`RequestList` is a
sequence of :class:`AccessRequest` objects (iterate or index it like a list)
that also carries the caller's ``username``. Each request exposes its queryable
:class:`RequestTable` s (empty unless the request is approved) and the
:class:`RequestConcept` grants behind it.

Use :meth:`RequestList.to_frame` (or :meth:`AccessRequest.to_frame`) to flatten
the catalog into a pandas DataFrame for easy browsing in a notebook.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd

#: Status value a request must have to grant data access.
STATUS_APPROVED = "approved"


@dataclass(frozen=True)
class RequestConcept:
    """One concept grant inside an access request.

    Attributes:
        concept_id: The granted (local) OMOP concept id.
        cohort: The cohort the grant applies to — the concept filters rows only
            in that cohort's part of the data.
        subgroup: Test/instrument subgroup (e.g. ``Deep Breathing``).
        source_field_name: The raw source field the concept maps to.
        source_field_description: Human-readable description of that field.
    """

    concept_id: int
    cohort: int
    subgroup: str
    source_field_name: str
    source_field_description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestConcept:
        return cls(
            concept_id=int(data.get("concept_id") or 0),
            cohort=int(data.get("cohort") or 0),
            subgroup=data.get("subgroup") or "",
            source_field_name=data.get("source_field_name") or "",
            source_field_description=data.get("source_field_description") or "",
        )


@dataclass(frozen=True)
class RequestTable:
    """One queryable table of an approved access request.

    Attributes:
        name: Bare OMOP table name (e.g. ``person``).
        relation: Exactly what to put in a FROM clause through the proxy
            (e.g. ``req_dc43d8d9_6f6c_4152_b5f1_0f9668ba01e4.person``).
    """

    name: str
    relation: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestTable:
        return cls(
            name=data.get("name") or "",
            relation=data.get("relation") or "",
        )


@dataclass(frozen=True)
class AccessRequest:
    """A data access request and what it makes queryable.

    Attributes:
        id: The request id (UUID) as issued by the auth server.
        schema: The logical namespace the request's tables live under,
            ``req_<request_id with dashes as underscores>``.
        dataset: Dataset the request was made against (e.g. ``AFT``).
        use_case: The use case the request was made for, as stated by the
            requester (free text; empty if the server didn't supply one).
        status: Request status; only ``approved`` grants access.
        cohorts: Cohort ids granted by the request.
        created_at: Creation timestamp (ISO-8601 string, as sent by the server).
        tables: Queryable tables — empty unless the request is approved.
        concepts: The concept grants behind the request.
    """

    id: str
    schema: str
    dataset: str
    use_case: str
    status: str
    cohorts: list[int]
    created_at: str
    tables: list[RequestTable]
    concepts: list[RequestConcept]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessRequest:
        return cls(
            id=data.get("id") or "",
            schema=data.get("schema") or "",
            dataset=data.get("dataset") or "",
            use_case=data.get("use_case") or "",
            status=data.get("status") or "",
            cohorts=[int(c) for c in (data.get("cohorts") or [])],
            created_at=data.get("created_at") or "",
            tables=[RequestTable.from_dict(t) for t in (data.get("tables") or []) if t is not None],
            concepts=[
                RequestConcept.from_dict(c) for c in (data.get("concepts") or []) if c is not None
            ],
        )

    def __repr__(self) -> str:
        head = f"{self.id[:8]}…" if len(self.id) > 8 else self.id
        return (
            f"AccessRequest(id={head}, dataset={self.dataset}, use_case={self.use_case!r}, "
            f"status={self.status}, cohorts={self.cohorts}, tables={len(self.tables)}, "
            f"concepts={len(self.concepts)})"
        )

    @property
    def is_approved(self) -> bool:
        """Whether this request grants data access."""
        return self.status.strip().lower() == STATUS_APPROVED

    @property
    def table_names(self) -> list[str]:
        """The bare names of the request's queryable tables."""
        return [t.name for t in self.tables]

    def get_table(self, name: str) -> RequestTable | None:
        """Return the table with this name, or ``None`` if there isn't one."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def to_frame(self) -> pd.DataFrame:
        """Return this request's tables as a DataFrame, one row per table."""
        rows = [{"table": t.name, "relation": t.relation} for t in self.tables]
        return pd.DataFrame(rows, columns=["table", "relation"])

    def concepts_frame(self) -> pd.DataFrame:
        """Return this request's concept grants as a DataFrame, one row per concept."""
        rows = [
            {
                "concept_id": c.concept_id,
                "cohort": c.cohort,
                "subgroup": c.subgroup,
                "source_field_name": c.source_field_name,
                "source_field_description": c.source_field_description,
            }
            for c in self.concepts
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "concept_id",
                "cohort",
                "subgroup",
                "source_field_name",
                "source_field_description",
            ],
        )


@dataclass(frozen=True)
class RequestList:
    """The catalog of access requests for the caller.

    Behaves like a sequence of :class:`AccessRequest` (iterate or index it) and
    also carries the caller's ``username``.

    Attributes:
        username: The authenticated username the catalog was built for.
        requests: The access requests (any status).
    """

    username: str
    requests: list[AccessRequest]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestList:
        return cls(
            username=data.get("username") or "",
            requests=[
                AccessRequest.from_dict(r) for r in (data.get("requests") or []) if r is not None
            ],
        )

    @property
    def ids(self) -> list[str]:
        """The ids of all requests, in catalog order."""
        return [r.id for r in self.requests]

    @property
    def approved(self) -> list[AccessRequest]:
        """The subset of requests that grant data access."""
        return [r for r in self.requests if r.is_approved]

    def get(self, request_id: str) -> AccessRequest | None:
        """Return the request with this id (or ``schema`` alias), or ``None``.

        Accepts either the UUID form (``dc43d8d9-6f6c-...``) or the schema form
        (``req_dc43d8d9_6f6c_...``) so anything you copied from a query or the
        catalog works.
        """
        for request in self.requests:
            if request.id == request_id or request.schema == request_id:
                return request
        return None

    def to_frame(self) -> pd.DataFrame:
        """Flatten the catalog into one DataFrame, one row per (request, table).

        Columns: ``request_id``, ``dataset``, ``use_case``, ``status``,
        ``cohorts``, ``table``, ``relation``. Requests with no queryable tables
        (pending/rejected) appear as a single row with an empty
        ``table``/``relation``.
        """
        rows = []
        for request in self.requests:
            base = {
                "request_id": request.id,
                "dataset": request.dataset,
                "use_case": request.use_case,
                "status": request.status,
                "cohorts": ", ".join(str(c) for c in request.cohorts),
            }
            if request.tables:
                for table in request.tables:
                    rows.append({**base, "table": table.name, "relation": table.relation})
            else:
                rows.append({**base, "table": "", "relation": ""})
        return pd.DataFrame(
            rows,
            columns=[
                "request_id",
                "dataset",
                "use_case",
                "status",
                "cohorts",
                "table",
                "relation",
            ],
        )

    def summary(self) -> pd.DataFrame:
        """Return a concise overview, one row per request.

        Columns: ``request_id``, ``dataset``, ``use_case``, ``status``,
        ``cohorts``, and ``tables`` (the count of queryable tables). For the full
        per-table breakdown (with ``relation`` strings) use :meth:`to_frame`
        instead.
        """
        rows = [
            {
                "request_id": r.id,
                "dataset": r.dataset,
                "use_case": r.use_case,
                "status": r.status,
                "cohorts": ", ".join(str(c) for c in r.cohorts),
                "tables": len(r.tables),
            }
            for r in self.requests
        ]
        return pd.DataFrame(
            rows,
            columns=["request_id", "dataset", "use_case", "status", "cohorts", "tables"],
        )

    def __repr__(self) -> str:
        return (
            f"RequestList(username={self.username}, requests={len(self.requests)}, "
            f"approved={len(self.approved)})"
        )

    def _repr_html_(self) -> str | None:
        """Render the concise :meth:`summary` as a table in Jupyter notebooks."""
        try:
            return self.summary()._repr_html_()
        except Exception:
            return None

    def __iter__(self) -> Iterator[AccessRequest]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> AccessRequest:
        return self.requests[index]
