import cx_Oracle
import numpy as np
import pandas as pd


def data_conn():
    connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

    cursor = connection.cursor()
    query = ("""
        SELECT  TAG_PREFIX
                ,TRUNC(TIME) AS my_date
                ,MAX(CTS_VC)
        FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
        WHERE CTS_VC IS NOT NULL
        GROUP BY TAG_PREFIX, TRUNC(TIME)
        ORDER BY TAG_PREFIX, TRUNC(TIME)
    """)

    cursor.execute(query)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    return df


if __name__ == "__main__":
    df = data_conn()
