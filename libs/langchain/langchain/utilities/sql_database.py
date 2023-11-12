"""SQLAlchemy wrapper around a database."""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union

import sqlalchemy
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.types import NullType


def _format_index(index: sqlalchemy.engine.interfaces.ReflectedIndex) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )


def truncate_word(content: Any, *, length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a certain number of words, based on the max string
    length.
    """

    if not isinstance(content, str) or length <= 0:
        return content

    if len(content) <= length:
        return content

    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        max_string_length: int = 300,
        data_dict: dict = None,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        self._data_dict = data_dict
        self._max_string_length = max_string_length

        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        #TODO: Handle include_tables and ignore_tables        
    

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        
        data_dict_only_table_description_str = []
        for table_name, table_details in self._data_dict.items():
            data_dict_only_table_description_str.append(f"\n{table_name}: {table_details['description']}")
        return data_dict_only_table_description_str


    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.
        """

        tables_decription_list = []

        for table_name in table_names:
            table_decription = self._data_dict[table_name]['metaDataString']
            if len( self._data_dict[table_name]['dict']) > 0:
                table_decription += "\n\nFollowing are short description of columns:\n"
                table_decription += "\n".join(f"{col['name']} : {col['description']}" for col in self._data_dict[table_name]['dict'])
            tables_decription_list.append(table_decription)

        final_str = "\n\n".join(tables_decription_list)
        return final_str


    def _execute(
        self,
        command: str,
        fetch: Union[Literal["all"], Literal["one"]] = "all",
    ) -> Sequence[Dict[str, Any]]:
        """
        Executes SQL command through underlying engine.

        If the statement returns no rows, an empty list is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                if self.dialect == "snowflake":
                    connection.exec_driver_sql(
                        "ALTER SESSION SET search_path = %s", (self._schema,)
                    )
                elif self.dialect == "bigquery":
                    connection.exec_driver_sql("SET @@dataset_id=?", (self._schema,))
                elif self.dialect == "mssql":
                    pass
                    # for table_name in self._data_dict.keys():
                    #     pattern = r'([ \n\[])' + table_name + r'([ \.\n\;\[\]]|$)'
                    #     replacement = r'\1' + self._schema + '.' + table_name + r'\2'
                    #     command = re.sub(pattern, replacement, command)
                elif self.dialect == "trino":
                    connection.exec_driver_sql("USE ?", (self._schema,))
                elif self.dialect == "duckdb":
                    # Unclear which parameterized argument syntax duckdb supports.
                    # The docs for the duckdb client say they support multiple,
                    # but `duckdb_engine` seemed to struggle with all of them:
                    # https://github.com/Mause/duckdb_engine/issues/796
                    connection.exec_driver_sql(f"SET search_path TO {self._schema}")
                else:  # postgresql and other compatible dialects
                    connection.exec_driver_sql("SET search_path TO %s", (self._schema,))
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                if fetch == "all":
                    result = [x._asdict() for x in cursor.fetchall()]
                elif fetch == "one":
                    first_result = cursor.fetchone()
                    result = [] if first_result is None else [first_result._asdict()]
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                return result
        return []

    def run(
        self,
        command: str,
        fetch: Union[Literal["all"], Literal["one"]] = "all",
    ) -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        result = self._execute(command, fetch)
        # Convert columns values to string to avoid issues with sqlalchemy
        # truncating text
        res = [
            list(truncate_word(c, length=self._max_string_length) for c in r.values())
            for r in result
        ]
        if not res:
            return ""
        else:
            res = {"columnHeaders": list(result[0].keys()) , "rows": res}
            return str(res)

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(
        self,
        command: str,
        fetch: Union[Literal["all"], Literal["one"]] = "all",
    ) -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"
