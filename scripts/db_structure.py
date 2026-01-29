#!/usr/bin/env python3
"""
Stampa le tabelle presenti in un database SQLite e la loro struttura.

Uso:
    python scripts/db_structure.py data/semcor_index.sqlite3

Mostra per ogni tabella:
 - elenco colonne (nome, tipo, notnull, default, pk)
 - indici
 - foreign keys

Autore: generato automaticamente
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime


def get_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    return cur.fetchall()


def get_columns(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table_name}')")
    cols = cur.fetchall()
    # pragma table_info: cid,name,type,notnull,dflt_value,pk
    return cols


def get_indexes(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA index_list('{table_name}')")
    idxs = cur.fetchall()
    # seq,name,unique,origin,partial
    details = []
    for idx in idxs:
        name = idx[1]
        cur.execute(f"PRAGMA index_info('{name}')")
        cols = cur.fetchall()  # seqno, cid, name
        cur.execute(f"SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,))
        sql = cur.fetchone()
        details.append((name, idx, cols, sql[0] if sql else None))
    return details


def get_foreign_keys(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA foreign_key_list('{table_name}')")
    fks = cur.fetchall()
    # id,seq,table,from,to,on_update,on_delete,match
    return fks


def get_sample_rows(conn, table_name, limit=20):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM '{table_name}' LIMIT ?", (limit,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, rows
    except sqlite3.OperationalError:
        return [], []


def describe_db(path: Path):
    if not path.exists():
        print(f"File non trovato: {path}")
        return 1
    conn = sqlite3.connect(str(path))
    try:
        tables = get_tables(conn)
        if not tables:
            print("Nessuna tabella trovata.")
            return 0

        out_lines = []
        out_lines.append(f"# Struttura database: {path}\n")
        out_lines.append(f"_Generato: {datetime.utcnow().isoformat()}Z_\n")

        for name, typ in tables:
            out_lines.append(f"## {name} ({typ})\n")

            cols = get_columns(conn, name)
            if cols:
                out_lines.append("### Columns\n")
                out_lines.append("| name | type | notnull | pk | default |")
                out_lines.append("|---|---:|:---:|:---:|---|")
                for c in cols:
                    cid, colname, coltype, notnull, dflt_value, pk = c
                    out_lines.append(f"| {colname} | {coltype or ''} | {bool(notnull)} | {pk} | {dflt_value if dflt_value is not None else ''} |")
            else:
                out_lines.append("(no columns)\n")

            # Sample rows (prime 20) — se possibile
            sample_cols, sample_rows = get_sample_rows(conn, name, limit=20)
            if sample_rows:
                out_lines.append("### Sample rows (first 20)\n")
                # header
                hdr = "| " + " | ".join(sample_cols) + " |"
                sep = "|" + "---|" * len(sample_cols)
                out_lines.append(hdr)
                out_lines.append(sep)
                for r in sample_rows:
                    # convert and sanitize values for markdown table
                    cells = []
                    for v in r:
                        if v is None:
                            s = ""
                        else:
                            s = str(v)
                        s = s.replace('\n', ' ').replace('|', '\\|')
                        cells.append(s)
                    out_lines.append("| " + " | ".join(cells) + " |")
            else:
                out_lines.append("(no sample rows)\n")

            fks = get_foreign_keys(conn, name)
            if fks:
                out_lines.append("### Foreign keys\n")
                for fk in fks:
                    _id, seq, ref_table, frm, to, on_update, on_delete, match = fk
                    out_lines.append(f"- `{frm}` -> `{ref_table}.{to}` (on_update={on_update}, on_delete={on_delete}, match={match})")

            idxs = get_indexes(conn, name)
            if idxs:
                out_lines.append("### Indexes\n")
                for name_idx, meta, cols, sql in idxs:
                    seq, idxname, unique, origin, partial = meta
                    uniq = bool(unique)
                    col_names = [c[2] for c in cols]
                    out_lines.append(f"- `{name_idx}`; unique={uniq}; cols={col_names}; sql={sql}")

            out_lines.append("\n---\n")

        # assicurarsi che la cartella outputs esista
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "db_structure.md"
        out_file.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"File generato: {out_file}")
        return 0
    finally:
        conn.close()


def main(argv=None):
    # Hardcoded input DB as requested
    default_db = Path("data/semcor_index.sqlite3")
    # allow optional override for convenience
    parser = argparse.ArgumentParser(description="Stampa struttura di un DB SQLite (hardcoded default)")
    parser.add_argument("db", nargs="?", default=str(default_db), help="Percorso al file SQLite (opzionale)")
    parser.add_argument("-o", "--out", help="Percorso file output markdown (opzionale)", default=None)
    args = parser.parse_args(argv)
    path = Path(args.db)
    return describe_db(path)


if __name__ == '__main__':
    sys.exit(main())
