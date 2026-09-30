"""Postgres helpers. Table, column and WHERE fragments come from operator config and are trusted;
identifiers are still quoted. Values are always passed as parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import psycopg2
from psycopg2 import errors
from psycopg2.extras import RealDictCursor

NEW_UUID_SQL = "md5(random()::text || clock_timestamp()::text)::uuid"


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    has_default: bool = False
    identity: str = ""
    generated: str = ""
    not_null: bool = False


@dataclass(frozen=True)
class Raw:
    sql: str


def connect(url: str):
    return psycopg2.connect(url, cursor_factory=RealDictCursor)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_table(name: str) -> str:
    return ".".join(quote_ident(part) for part in name.split("."))


def table_columns(cur, table: str) -> list[Column]:
    cur.execute(
        """
        SELECT a.attname AS name,
               format_type(a.atttypid, a.atttypmod) AS type,
               d.adbin IS NOT NULL AS has_default,
               a.attidentity AS identity,
               a.attgenerated AS generated,
               a.attnotnull AS not_null
        FROM pg_attribute a
        LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        (table,),
    )
    return [
        Column(
            name=row["name"],
            type=row["type"],
            has_default=bool(row["has_default"]),
            identity=(row["identity"] or "").strip(),
            generated=(row["generated"] or "").strip(),
            not_null=bool(row["not_null"]),
        )
        for row in cur.fetchall()
    ]


STREAM_COLUMN_HINTS = ("url", "uri", "rtsp", "stream")


def scrub_stream_overrides(columns: list[Column]) -> dict[str, Any]:
    """Blank any column that could point a stream reader at the template's real camera."""
    overrides: dict[str, Any] = {}
    for column in columns:
        lowered = column.name.lower()
        if any(hint in lowered for hint in STREAM_COLUMN_HINTS):
            overrides[column.name] = "" if column.not_null else None
    return overrides


def column_names(columns: Iterable[Column]) -> set[str]:
    return {column.name for column in columns}


def new_key_override(columns: list[Column], key_column: str) -> dict[str, Any]:
    """Override for the primary key of a cloned row, or {} when the database generates it."""
    by_name = {column.name: column for column in columns}
    if key_column not in by_name:
        raise ValueError(f"Key column '{key_column}' not found")
    key = by_name[key_column]
    if key.has_default or key.identity:
        return {}
    if key.type == "uuid":
        return {key_column: Raw(NEW_UUID_SQL)}
    raise ValueError(f"Cannot generate a new value for key column '{key_column}' of type {key.type}")


def audit_overrides(columns: list[Column]) -> dict[str, Any]:
    names = column_names(columns)
    overrides: dict[str, Any] = {}
    for name in ("created_at", "modified_at", "updated_at"):
        if name in names:
            overrides[name] = Raw("now()")
    for name in ("deleted_at", "deleted_by"):
        if name in names:
            overrides[name] = None
    return overrides


def build_clone_sql(
    table: str,
    columns: list[Column],
    key_column: str,
    where_column: str,
    where_value: str,
    overrides: dict[str, Any],
) -> tuple[str, list[Any]]:
    """INSERT ... SELECT copying rows WHERE where_column = where_value, with overridden columns."""
    known = column_names(columns)
    unknown = [name for name in overrides if name not in known]
    if unknown:
        raise ValueError(f"Overrides for unknown columns of {table}: {', '.join(unknown)}")

    insert_columns: list[str] = []
    select_exprs: list[str] = []
    params: list[Any] = []
    for column in columns:
        if column.generated:
            continue
        if column.name in overrides:
            value = overrides[column.name]
            insert_columns.append(quote_ident(column.name))
            if isinstance(value, Raw):
                select_exprs.append(value.sql)
            elif value is None:
                select_exprs.append(f"NULL::{column.type}")
            else:
                select_exprs.append(f"%s::{column.type}")
                params.append(value)
        elif column.name == key_column and (column.has_default or column.identity):
            continue
        elif column.identity == "a":
            continue
        else:
            insert_columns.append(quote_ident(column.name))
            select_exprs.append(f"src.{quote_ident(column.name)}")

    table_sql = quote_table(table)
    live_only = " AND src.deleted_at IS NULL" if "deleted_at" in known else ""
    sql = (
        f"INSERT INTO {table_sql} ({', '.join(insert_columns)}) "
        f"SELECT {', '.join(select_exprs)} FROM {table_sql} AS src "
        f"WHERE src.{quote_ident(where_column)}::text = %s{live_only} "
        f"RETURNING {quote_ident(key_column)}::text AS key"
    )
    params.append(str(where_value))
    return sql, params


def build_json_array_edit(
    table: str,
    column: str,
    column_type: str,
    key_column: str,
    key_value: str,
    remove_ids: list[str],
    add_ids: list[str],
) -> tuple[str, list[Any]]:
    """Remove ids from (and optionally append ids to) a JSON/JSONB array column of one row."""
    if column_type not in ("json", "jsonb"):
        raise ValueError(f"{table}.{column} is {column_type}, expected json or jsonb")
    col = quote_ident(column)
    expr = (
        "(SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb) "
        f"FROM jsonb_array_elements(COALESCE({col}::jsonb, '[]'::jsonb)) AS elem "
        "WHERE NOT ((elem #>> '{}') = ANY(%s)))"
    )
    params: list[Any] = [list(remove_ids)]
    if add_ids:
        expr = f"({expr} || %s::jsonb)"
        params.append(json.dumps(list(add_ids)))
    sql = f"UPDATE {quote_table(table)} SET {col} = {expr}::{column_type} WHERE {quote_ident(key_column)}::text = %s"
    params.append(str(key_value))
    return sql, params


def to_text(value: Any) -> Any:
    """Serialise a DB value so it can be cast back with %s::<type>."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (bytes, memoryview)):
        return "\\x" + bytes(value).hex()
    return str(value)


def build_restore_sql(table: str, columns: list[Column], key_column: str, row: dict[str, Any]) -> tuple[str, list[Any]]:
    assignments: list[str] = []
    params: list[Any] = []
    for column in columns:
        if column.name == key_column or column.generated or column.name not in row:
            continue
        assignments.append(f"{quote_ident(column.name)} = %s::{column.type}")
        params.append(row[column.name])
    sql = f"UPDATE {quote_table(table)} SET {', '.join(assignments)} WHERE {quote_ident(key_column)}::text = %s"
    params.append(str(row[key_column]))
    return sql, params


def select_values(cur, table: str, column: str, values: list[str], select_columns: list[str]) -> list[str]:
    """Non-empty values of `select_columns` (those the table has) in rows matching `values`."""
    if not values or not select_columns:
        return []
    existing = column_names(table_columns(cur, table))
    wanted = [name for name in select_columns if name in existing]
    if not wanted:
        return []
    selected = ", ".join(f"{quote_ident(name)}::text AS {quote_ident(name)}" for name in wanted)
    cur.execute(
        f"SELECT {selected} FROM {quote_table(table)} WHERE {quote_ident(column)}::text = ANY(%s)",
        (list(values),),
    )
    return [row[name] for row in cur.fetchall() for name in wanted if row[name]]


def delete_by_values(cur, table: str, column: str, values: list[str]) -> int:
    if not values:
        return 0
    cur.execute(
        f"DELETE FROM {quote_table(table)} WHERE {quote_ident(column)}::text = ANY(%s)",
        (list(values),),
    )
    return cur.rowcount


def delete_or_soft_delete(conn, table: str, key_column: str, keys: list[str]) -> str:
    """Hard delete; if a foreign key blocks it, set deleted_at instead. Commits."""
    if not keys:
        return "nothing"
    with conn.cursor() as cur:
        try:
            delete_by_values(cur, table, key_column, keys)
            conn.commit()
            return "deleted"
        except errors.ForeignKeyViolation:
            conn.rollback()
        if "deleted_at" not in column_names(table_columns(cur, table)):
            raise RuntimeError(f"Foreign key blocks deleting from {table} and it has no deleted_at column")
        cur.execute(
            f"UPDATE {quote_table(table)} SET deleted_at = now() WHERE {quote_ident(key_column)}::text = ANY(%s)",
            (list(keys),),
        )
        conn.commit()
        return "soft_deleted"
