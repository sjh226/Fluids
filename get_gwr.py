import time
import pyodbc
import csv
import pandas as pd
import urllib
import sqlalchemy
from datetime import date, timedelta
from joblib import Parallel
from joblib import delayed
from PIthon import *

connect_to_Server('149.179.68.101')


def pullTag(tag, days='-1d', end='t'):
    try:
        i = tag
        prefix = i.split('.')[0]
        tank = i.split('.')[1]
        print('[+] Pulling ', i)

        results = get_tag_interpolate(i, days, end, minutes=15)
        final = []
        for j in results:
            final.append([prefix, tank] + j)
        return final
    except:
        pass

def pullTurbine(tag, days='-1d', end='t'):
    try:
        i = tag
        prefix = i.split('.')[0]
        tank = i.split('.')[1]
        print('[+] Pulling ', i)

        results = get_tag_interpolate(i, days, end, minutes=15)
        final = []
        for j in results:
            final.append([prefix, tank] + j)
        return final
    except:
        pass

def pullQuery(pull):
    print("[+] connecting to the servers")
    server = '10.75.6.160'
    cnxn = pyodbc.connect(driver='{SQL Server}', server=server, database='OperationsDataMart', trusted_connection='yes')
    cursor = cnxn.cursor()
    cursor.execute(pull)
    row = cursor.fetchone()
    results = []
    while row:
        results.append(row)
        row = cursor.fetchone()
    return results

def tag_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=TeamOptimizationEngineering;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
        print("Connection Error")
        sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT  P.TAG
                ,W.API
        FROM [TeamOptimizationEngineering].[Reporting].[PITag_Dict] P
        INNER JOIN [OperationsDataMart].[Dimensions].[Wells] W
            ON W.API = P.API
        --WHERE P.API IN ('4903729563', '4903729534', '4903729531', '4903729560',
        --                '4903729561', '4903729555', '4903729556', '4903729582',
        --                '4903729584', '4903729551', '4900724584', '4903729547',
        --                '4903729468', '4903729548', '4903729519', '4903729514');
	""")

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
        df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
        df.columns = [col.lower() for col in df.columns]
    except:
        df = None
        print('Dataframe is empty')

    return df

def csvToList(input):
    output = []
    with open(input, "r") as f:
        reader_out = csv.reader(f, delimiter=',', lineterminator='\n')
        for line in reader_out:
            output.append(line)
    return output

def listListToCsv(input, output):
    with open(output, "w+") as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(input)

def sql_push(df, table):
    params = urllib.parse.quote_plus('Driver={SQL Server Native Client 11.0};\
									 Server=SQLDW-L48.BP.Com;\
									 Database=TeamOptimizationEngineering;\
     								 UID=ThundercatIO;\
     								 PWD=thund3rc@t10'
                                     )
    engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

    df.to_sql(table, engine, schema='Reporting', if_exists='append', index=False)

def pull_gwr(tags=None, tag_limit=None):
    tags_to_pull = []

    query = '''
        SELECT DISTINCT concat([Tag_Prefix],'.',[Tank])
          FROM [TeamOptimizationEngineering].[Reporting].[GWR_Test]
          WITH (NoLock)
          WHERE cast([DateTime] AS DATE)  = cast(getdate()-1 AS DATE)
    '''

    alreadyHave = pullQuery(query)

    if tags:
        for i in tags[1:]:
            if i[0] + '.' + i[1] not in alreadyHave:
                if i[0] in tag_limit:
                    tags_to_pull.append(i[0] + '.' + i[1])
            else:
                print(i)
    else:
        tags_to_pull = tag_pull()

    gatheredData = [['Tag_Prefix', 'Tank', 'DateTimeStamp', 'Value']]

    alldata = []
    results = Parallel(n_jobs=10)(delayed(pullTag)(i, '-1d') for i in tags_to_pull)

    for i in results:
        try:
            gatheredData += i
        except:
            pass

    listListToCsv(gatheredData, 'data/GWRDump.csv')
    df = pd.read_csv('data/GWRDump.csv')
    df.rename(index=str, columns={'DateTimeStamp': 'DateTime'}, inplace=True)
    df.drop_duplicates(inplace=True)
    df.loc[:, 'Value'] = pd.to_numeric(df.loc[:, 'Value'], errors='coerce')
    # df.to_csv('data/gwr_sql.csv', index=False)
    sql_push(df, 'GWR_Test')

def turbine_pull():
    tag_df = tag_pull()
    tags = tag_df['tag'].values

    CTS_Tags = [i + '.' + 'CTS_VY' for i in tags] + [i + '.' + 'WAT_VY' for i in tags]

    gatheredData = [['Tag_Prefix', 'Tag', 'DateTimeStamp', 'Value']]

    alldata = []
    results = Parallel(n_jobs=10)(delayed(pullTurbine)(i, '-2d') for i in CTS_Tags)

    for i in results:
        try:
            gatheredData += i
        except:
            pass

    listListToCsv(gatheredData, 'data/TurbineDump.csv')
    df = pd.read_csv('data/TurbineDump.csv')
    df.rename(index=str, columns={'DateTimeStamp': 'DateTime'}, inplace=True)
    df.drop_duplicates(inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.date
    df = df.groupby(['Tag_Prefix', 'Tag', 'DateTime'], as_index=False).max()
    df = df.loc[df['DateTime'] == date.today() - timedelta(1), :]
    df.loc[:, 'Value'] = pd.to_numeric(df.loc[:, 'Value'], errors='coerce')
    sql_push(df, 'Turbine_Test')

if __name__ == '__main__':
    tags = csvToList('data/GottenTagNamesGWR.csv')
    tag_limit = ['WAM-CH320C1-160H','WAM-CH533B3_80D','WAM-CL29_150H',\
                 'WAM-CL29_160H','WAM-CL32_45H','WAM-LM8_115H','WAM-ML11_150H',\
                 'WAM-ML11_160D','WAM-ML11_160H']

    # pull_gwr(tags, tag_limit)
    pull_gwr()
    turbine_pull()
