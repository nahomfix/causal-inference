import os

import mysql.connector as mysql
import pandas as pd
from mysql.connector import Error


def db_connect(db_name=None):
    """
    Parameters
    ----------
    db_name :
        Default value = None)
    Returns
    -------
    """
    conn = mysql.connect(
        host="localhost",
        user="causal_inference_user",
        password="causal_inference_user_pass",
        database=db_name,
        buffered=True,
    )

    cur = conn.cursor()
    return conn, cur


def create_db(db_name: str) -> None:
    """
    Parameters
    ----------
    db_name :
        str:
    Returns
    -------
    """
    conn, cur = db_connect()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {db_name};")
    conn.commit()
    cur.close()


def create_tables(db_name: str) -> None:
    """
    Parameters
    ----------
    db_name :
        str:
    Returns
    -------
    """
    conn, cur = db_connect(db_name)
    sql_file = "features.sql"
    fd = open(sql_file, "r")
    read_sql_file = fd.read()
    fd.close()

    sql_commands = read_sql_file.split(";")

    for command in sql_commands:
        try:
            res = cur.execute(command)
        except Exception as ex:
            print("Command skipped: ", command)
            print(ex)
    conn.commit()
    cur.close()

    return


def preprocess_df(df: pd.DataFrame, col_to_drop: list) -> pd.DataFrame:
    """
    Parameters
    ----------
    df :
        pd.DataFrame:
    col_to_drop:
        list
    Returns
    -------
    df :
        pd.DataFrame:
    """
    try:
        df = df.drop(columns=col_to_drop, axis=1)
        df = df.fillna(0)
    except KeyError as e:
        print("Error:", e)

    return df


def insert_to_table(db_name: str, df: pd.DataFrame, table_name: str) -> None:
    """
    Parameters
    ----------
    db_name :
        str:
    df :
        pd.DataFrame:
    table_name :
        str:
    Returns
    -------
    """
    conn, cur = db_connect(db_name)

    df = preprocess_df(df, [])

    for _, row in df.iterrows():
        print(row)
        sql_query = f"""INSERT INTO {table_name} (area_se, area_worst, concavity_mean, radius_se, symmetry_worst, texture_worst)
             VALUES(%s, %s, %s, %s, %s);"""
        data = (row[0], row[1], row[2], row[3], (row[4]))

        try:
            # Execute the SQL command
            cur.execute(sql_query, data)

            # Commit your changes in the database
            conn.commit()
            print("Data Inserted Successfully")
        except Exception as e:
            conn.rollback()
            print("Error: ", e)
    return


def db_execute_fetch(
    *args, many=False, table_name="", rdf=True, **kwargs
) -> pd.DataFrame:
    """
    Parameters
    ----------
    *args :
    many :
         (Default value = False)
    table_name :
         (Default value = '')
    rdf :
         (Default value = True)
    **kwargs :
    Returns
    -------
    """
    connection, cursor1 = db_connect(**kwargs)
    if many:
        cursor1.executemany(*args)
    else:
        cursor1.execute(*args)

    # get column names
    field_names = [i[0] for i in cursor1.description]

    # get column values
    res = cursor1.fetchall()

    # get row count and show info
    nrow = cursor1.rowcount
    if table_name:
        print(f"{nrow} records fetched from {table_name} table")

    cursor1.close()
    connection.close()

    # return result
    if rdf:
        return pd.DataFrame(res, columns=field_names)
    else:
        return res


if __name__ == "__main__":
    create_db(db_name="breast_cancer_diagnostic")
    create_tables(db_name="breast_cancer_diagnostic")

    clean_df = pd.read_csv("../data/data_clean.csv")
    selected_features = [
        "area_se",
        "area_worst",
        "concavity_mean",
        "radius_se",
        "symmetry_worst",
        "texture_worst",
    ]
    processed_df = clean_df.loc[:, selected_features]

    insert_to_table(
        db_name="breast_cancer_diagnostic",
        df=processed_df,
        table_name="BreastCancerDiagnostic",
    )
