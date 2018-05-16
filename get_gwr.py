import time
import pyodbc
import csv
import pandas as pd
import urllib
import sqlalchemy
from tqdm import tqdm
from joblib import Parallel
from joblib import delayed
from PIthon import *

connect_to_Server('149.179.68.101')


def pullTag(tag, days='-1d', end = 't'):
    try:
        i = tag
        prefix = i.split('.')[0]
        tank = i.split('.')[1]
        print('[+] Pulling ', i)

        # results = get_tag_values(i, days, 't')
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

def csvToSQL(tableLocation, csvFile, hasHeader=True, delTable=False, calcCol=False):
    server = '10.75.6.160'
    cnxn = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                          r'Server=10.75.6.160;'
                          r'Database=TeamOptimizationEngineering;'
						  r'UID=ThundercatIO;'
						  r'PWD=thund3rc@t10'
                          )
    print('[+] Connected to ' + server + '\n   Writing to location ' + tableLocation)
    with open(csvFile, 'r') as f:
        reader = csv.reader(f)
        if hasHeader == True:
            data = next(reader)
        query = 'insert into {}'.format(tableLocation) + ' values ({0})'
        cursor = cnxn.cursor()
        if delTable == True:
            cursor.execute('delete from ' + tableLocation)
        if calcCol == True:
            import datetime
            CalcTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for data in tqdm(reader):
            if calcCol == True:
                data.append(CalcTime)
            # print(data)
            try:
                query = query.format(','.join('?' * len(data)))
                # print(query)
                cursor.execute(query, data)
            except Exception as e:
                print('Failed on ', query, e)
        # try:
        #     cursor.execute(query, data)
        # except:
        #     print(data)
        cursor.commit()


if __name__ == '__main__':
    Tags = csvToList('data/GottenTagNamesGWR.csv')
    tags_to_pull = []

    query = '''

        SELECT Distinct concat([Tag_Prefix],'.',[Tank])
          FROM [TeamOptimizationEngineering].[Reporting].[North_GWR]
          with (NoLock)
          where cast([DateTime] as date)  = cast(getdate()-1 as date)

    '''

    alreadyHave = pullQuery(query)

    Tag_limit = ['WAM-CH320C1-160H','WAM-CH533B3_80D','WAM-CL29_150H',\
                 'WAM-CL29_160H','WAM-CL32_45H','WAM-LM8_115H','WAM-ML11_150H',\
                 'WAM-ML11_160D','WAM-ML11_160H']
    for i in Tags[1:]:
        if i[0] + '.' + i[1] not in alreadyHave:
            if i[0] in Tag_limit:
                tags_to_pull.append(i[0] + '.' + i[1])
        else:
            print(i)

    gatheredData = [['Tag_Prefix', 'Tank', 'DateTimeStamp', 'Value']]

    testCount = len(Tag_limit)
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
    # df.to_csv('data/gwr_sql.csv', index=False)
    sql_push(df, 'GWR_Test')
