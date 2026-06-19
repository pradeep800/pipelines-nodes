"""SDK-level "search_path": qualify bare table names with a request schema.

The proxy only recognizes relations named ``req_<request_id>.<table>``. After
:meth:`~cbr_data_access.client.DataAccessClient.set_access` pins a request, the
client rewrites each query so unqualified table names resolve against that
request's schema — exactly like PostgreSQL's ``search_path``, but applied
client-side before the SQL is sent. The proxy remains the enforcement point;
this rewrite is purely a convenience.
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp
from sqlglot.dialects.postgres import Postgres
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)


class _SQLAlchemyPostgres(Postgres):
    """Postgres dialect that keeps ``:name`` placeholders.

    The stock Postgres generator renders named placeholders in psycopg's
    ``%(name)s`` style, but the SDK passes statements to SQLAlchemy's
    ``text()``, which expects the ``:name`` form they were written in.
    """

    # sqlglot's Dialect metaclass re-binds Generator as a class variable, which
    # mypy cannot treat as a base class.
    class Generator(Postgres.Generator):  # type: ignore[valid-type, misc]
        def placeholder_sql(self, expression: exp.Placeholder) -> str:
            return f":{expression.name}" if expression.name else "?"


def qualify_statement(sql: str, schema: str) -> str:
    """Qualify every bare table name in ``sql`` with ``schema``.

    Leaves untouched: names already schema-qualified, CTE names, and table
    functions (e.g. ``generate_series(...)``). If the statement cannot be
    parsed, it is returned unchanged (with a warning) — the server still
    validates every query, so nothing unsafe can slip through.
    """
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except ParseError:
        logger.warning(
            "Could not parse SQL to apply the pinned request schema; "
            "sending it unchanged. Qualify table names explicitly if needed."
        )
        return sql

    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.db:  # already schema-qualified
            continue
        if not isinstance(table.this, exp.Identifier):  # table function etc.
            continue
        if table.name in cte_names:
            continue
        table.set("db", exp.to_identifier(schema))
    return tree.sql(dialect=_SQLAlchemyPostgres)
