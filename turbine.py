import cx_Oracle
import numpy as np
import pandas as pd


def data_conn():
    connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

    cursor = connection.cursor()
    cursor.execute("""
            SELECT first_name, last_name
            FROM employees
            WHERE department_id = :did AND employee_id > :eid""",
            did = 50,
            eid = 190)
    for fname, lname in cursor:
        print("Values:", fname, lname)


if __name__ == "__main__":
    data_conn()
