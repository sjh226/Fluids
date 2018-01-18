import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import sys
import cx_Oracle


def gwr_pull():
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
        Set NOCOUNT ON;

        DROP TABLE IF EXISTS #TOT_Sites;
        DROP TABLE IF EXISTS #CND_Sites;
        DROP TABLE IF EXISTS #WAT_Sites;

        SELECT [TAG_PREFIX]
        INTO [#TOT_Sites]
        FROM TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        WHERE [CalcDate] =
        (
            SELECT MAX([CalcDate])
            FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
              AND [Tank_Type] = 'TOT';

        SELECT [TAG_PREFIX]
        INTO [#WAT_Sites]
        FROM TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        WHERE [CalcDate] =
        (
            SELECT MAX([CalcDate])
            FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
              AND [Tank_Type] = 'WAT';

        SELECT [TAG_PREFIX]
        INTO [#CND_Sites]
        FROM TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        WHERE [CalcDate] =
        (
            SELECT MAX([CalcDate])
            FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
              AND [Tank_Type] = 'CND';

        DROP TABLE IF EXISTS #FacilityLevels;

        SELECT [FacilityKey],
               [W].[API],
               [PT].[TAG_PREFIX],
               [TANK_TYPE],
               CASE
                   WHEN [TANK_TYPE] = 'Wat'
                   THEN-1
                   ELSE 1
               END AS [Tank_Mult],
               [TANKLVL],
               [TANKCNT],
               [CalcDate]
        INTO [#FacilityLevels]
        FROM TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
             JOIN #TOT_Sites AS TS ON TS.TAg_Prefix = PT.Tag_Prefix
             JOIN #WAT_Sites AS WS ON WS.TAg_Prefix = PT.Tag_Prefix
             INNER JOIN TeamOptimizationEngineering.Reporting.PITag_Dict AS PD ON PD.TAG = PT.Tag_Prefix
                                                                                  AND Confidence = 100
             JOIN OperationsDataMart.Dimensions.Wells AS W ON W.API = PD.Api
        WHERE [PT].[Tag_Prefix] NOT IN
        (
            SELECT *
            FROM [#CND_Sites]
        )
        GROUP BY PT.TAG_PREFIX,
                 FacilityKey,
                 W.API,
                 TANK_TYPE,
                 TANKLVL,
                 TANKCNT,
                 CalcDate;

        DROP TABLE IF EXISTS #Final1;

        SELECT DISTINCT
               [TL].[Facilitykey],
               [F].[FacilityName],
               CASE
                   WHEN SUM([Tank_Mult]) OVER(PARTITION BY [TL].[FacilityKey], [TL].CalcDate) = 0
                   THEN SUM([Tank_Mult] * [TankLVL]) OVER(PARTITION BY [TL].[FacilityKey], [TL].CalcDate)
                   ELSE NULL
               END AS [CND_LVL],
               [WAT_LVL],
               [TOT_LVL],
                 TL.CalcDate
        INTO [#Final1]
        FROM #FacilityLevels AS TL
             JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                                                                   AND F.TankCount - 1 <= TL.TankCNT
             LEFT JOIN
        (
            SELECT DISTINCT
                   [TL].[Facilitykey],
                   [TL].[CalcDate],
                   CASE
                       WHEN [TANK_TYPE] = 'TOT'
                       THEN [TankLVL]
                       ELSE NULL
                   END AS [TOT_LVL]
            FROM #FacilityLevels AS TL
                 JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                                                                       AND F.TankCount - 1 <= TL.TankCNT
                                                                       AND [TANK_TYPE] = 'TOT'
        ) AS Tot ON Tot.Facilitykey = TL.Facilitykey and Tot.CalcDate = TL.CalcDate
             LEFT JOIN
        (
            SELECT DISTINCT
                   [TL].[Facilitykey],
                   [TL].[CalcDate],
                   CASE
                       WHEN [TANK_TYPE] = 'WAT'
                       THEN [TankLVL]
                       ELSE NULL
                   END AS [WAT_LVL]
            FROM #FacilityLevels AS TL
                 JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                                                                       AND F.TankCount - 1 <= TL.TankCNT
                                                                       AND [TANK_TYPE] = 'WAT'
        ) AS WAT ON WAT.Facilitykey = TL.Facilitykey and WAT.CalcDate = TL.CalcDate;

        DROP TABLE IF EXISTS #FacilityLevels2;

        SELECT      [FacilityKey],
                    [W].[API],
                    [PT].[TAG_PREFIX],
                    [TANK_TYPE],
                    CASE
                        WHEN [TANK_TYPE] = 'Wat'
                        THEN-1
                        ELSE 1
                    END AS [Tank_Mult],
                    [TANKLVL],
                    [TANKCNT],
                    [CalcDate]
        INTO [#FacilityLevels2]
        FROM   TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        JOIN #CND_Sites AS TS ON TS.TAg_Prefix = PT.Tag_Prefix
        JOIN #WAT_Sites AS WS ON WS.TAg_Prefix = PT.Tag_Prefix
        JOIN TeamOptimizationEngineering.Reporting.PITag_Dict AS PD
            ON PD.TAG = PT.Tag_Prefix
            AND Confidence = 100
        JOIN OperationsDataMart.Dimensions.Wells AS W
            ON W.API = PD.Api
        WHERE
            [PT].[Tag_Prefix] NOT IN
                (SELECT *
                FROM   [#TOT_Sites])
        GROUP BY PT.TAG_PREFIX, FacilityKey, W.API, TANK_TYPE, TANKLVL, TANKCNT, CalcDate;

        DROP TABLE IF EXISTS #Final2;

        SELECT DISTINCT
                    [TL].[Facilitykey],
                    [F].[FacilityName],
                    [CND_LVL],
                    [WAT_LVL],
                    [CND_LVL] + [WAT_LVL] AS [TOT_LVL],
            		[TL].[CalcDate]
        INTO [#Final2]
        FROM #FacilityLevels2 AS TL
        JOIN OperationsDataMart.Dimensions.Facilities AS F
            ON f.Facilitykey = TL.Facilitykey
        LEFT JOIN
            (SELECT DISTINCT
                    [TL].[Facilitykey],
        			[TL].[CalcDate],
                    SUM([TANKCNT]) OVER(PARTITION BY [TL].[Facilitykey], [TL].[CalcDate]) AS [TANKCNT]
            FROM   #FacilityLevels2 AS TL) AS CNT
            ON F.Facilitykey = CNT.Facilitykey
                AND F.TankCount - 1 <= CNT.TANKCNT
        		AND TL.CalcDate = CNT.CalcDate
        LEFT JOIN
            (SELECT DISTINCT
                [TL].[Facilitykey],
        		[TL].[CalcDate],
                CASE
                        WHEN [TANK_TYPE] = 'CND'
                        THEN [TankLVL]
                        ELSE NULL
                END AS [CND_LVL]
            FROM   #FacilityLevels2 AS TL
            WHERE  [TANK_TYPE] = 'CND') AS Tot
            ON Tot.Facilitykey = TL.Facilitykey
        	AND Tot.CalcDate = TL.CalcDate
        LEFT JOIN
                (SELECT DISTINCT
                        [TL].[Facilitykey],
        				[TL].[CalcDate],
                        CASE
                                WHEN [TANK_TYPE] = 'WAT'
                                THEN [TankLVL]
                                ELSE NULL
                        END AS [WAT_LVL]
                FROM   #FacilityLevels2 AS TL
                WHERE  [TANK_TYPE] = 'WAT') AS WAT
            ON WAT.Facilitykey = TL.Facilitykey
        	AND WAT.CalcDate = TL.CalcDate
        ORDER BY [FacilityName];

        DROP TABLE IF EXISTS #LGRV7;

        SELECT *
        INTO [#LGRV7]
        FROM     #Final1
        UNION ALL
        SELECT *
        FROM   #Final2;

        SELECT  LGR.FacilityKey
                ,LGR.FacilityName
                ,LGR.CalcDate
                ,LGR.CND_LVL
                ,LGR.WAT_LVL
                ,LGR.TOT_LVL
                ,DF.FacilityCapacity
        FROM #LGRV7 AS LGR
        JOIN (SELECT	Facilitykey
        				,MAX(CalcDate) maxtime
        		FROM #LGRV7
        		GROUP BY Facilitykey, DAY(CalcDate), MONTH(CalcDate), YEAR(CalcDate)) AS MD
        	ON	MD.Facilitykey = LGR.Facilitykey
        	AND	MD.maxtime = LGR.CalcDate
        JOIN [TeamOptimizationEngineering].[dbo].[DimensionsFacilities] AS DF
            ON DF.Facilitykey = LGR.Facilitykey
        ORDER BY LGR.Facilitykey, LGR.CalcDate;
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    df['CalcDate'] = pd.DatetimeIndex(df['CalcDate']).normalize()

    df['CND_rate'] = df['CND_LVL'] - df['CND_LVL'].shift(1)
    df['WAT_rate'] = df['WAT_LVL'] - df['WAT_LVL'].shift(1)
    df['TOT_rate'] = df['TOT_LVL'] - df['TOT_LVL'].shift(1)

    return df

def oracle_pull():
    connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

    cursor = connection.cursor()
    query = '''
        SELECT
            TAG_PREFIX,
            TIME,
            Tank_Type,
            TankVol,
            TankCnt

        FROM (
                 SELECT
                     row_number()
                     OVER (
                         PARTITION BY TAG_PREFIX
                         ORDER BY Time DESC ) AS rk,
                     TAG_PREFIX,
                     TIME,
                     (nvl(TNK_1_TOT_LVL, 0) + nvl(TNK_2_TOT_LVL, 0) + nvl(TNK_3_TOT_LVL, 0) + nvl(TNK_4_TOT_LVL, 0) +
                      nvl(TNK_5_TOT_LVL, 0) + nvl(TNK_6_TOT_LVL, 0) +
                      nvl(TNK_7_TOT_LVL, 0) + nvl(TNK_8_TOT_LVL, 0) + nvl(TNK_9_TOT_LVL, 0) + nvl(TNK_10_TOT_LVL, 0)) *
                     20                       AS TankVol,
                     GAS_VC,
                     CASE WHEN (TNK_1_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_2_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_3_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_4_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_5_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_6_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_7_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_8_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_9_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_10_TOT_LVL) >= 0
                         THEN 1
                     ELSE 0 END               AS TankCnt,
                     'TOT'                    AS Tank_Type
                 FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
             --Where TIME >= trunc(sysdate-2)
             ) Vol
        WHERE Vol.TankVol > 0
              --AND TIME >= trunc(sysdate)
              --AND rk = 1

        UNION ALL

        SELECT
            TAG_PREFIX,
            TIME,
            Tank_Type,
            TankVol,
            TankCnt

        FROM (
                 SELECT
                     row_number()
                     OVER (
                     PARTITION BY TAG_PREFIX
                         ORDER BY Time DESC ) AS rk,
                     TAG_PREFIX,
                     TIME,
                     (nvl(TNK_1_WAT_LVL, 0) +
                      nvl(TNK_2_WAT_LVL, 0) +
                      nvl(TNK_3_WAT_LVL, 0) +
                      nvl(TNK_4_WAT_LVL, 0) +
                      nvl(TNK_WAT_1_LVL, 0) +
                      nvl(TNK_WAT_10_LVL, 0) +
                      nvl(TNK_WAT_11_LVL, 0) +
                      nvl(TNK_WAT_2_LVL, 0) +
                      nvl(TNK_WAT_3_LVL, 0) +
                      nvl(TNK_WAT_305A_LVL, 0) +
                      nvl(TNK_WAT_305B_LVL, 0) +
                      nvl(TNK_WAT_305C_LVL, 0) +
                      nvl(TNK_WAT_305D_LVL, 0) +
                      nvl(TNK_WAT_305E_LVL, 0) +
                      nvl(TNK_WAT_310A_LVL, 0) +
                      nvl(TNK_WAT_310B_LVL, 0) +
                      nvl(TNK_WAT_310C_LVL, 0) +
                     nvl(TNK_WAT_310D_LVL, 0) +
                      nvl(TNK_WAT_4_LVL, 0) +
                      nvl(TNK_WAT_6_LVL, 0) +
                      nvl(TNK_WAT_A_LVL, 0) +
                      nvl(TNK_WAT_B_LVL, 0) +
                      nvl(TNK_WAT_C_LVL, 0) +
                      nvl(TNK_WAT_D_LVL, 0)) *
                     20                       AS TankVol,
                     GAS_VC,
                     CASE WHEN (TNK_1_WAT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_2_WAT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_3_WAT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_4_WAT_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_1_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_10_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_11_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_2_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_3_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_305A_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_305B_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_305C_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_305D_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_305E_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_310A_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_310B_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_310C_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_310D_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_4_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_6_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_A_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_B_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_c_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_WAT_d_LVL) >= 0
                         THEN 1
                     ELSE 0 END               AS TankCnt,
                     'WAT'                    AS Tank_Type
                 FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
                 --WHERE TIME >= trunc(sysdate - 2)
             ) Vol
        WHERE Vol.TankVol > 0
              --AND TIME >= trunc(sysdate)
              --AND rk = 1

        UNION ALL

        SELECT
            TAG_PREFIX,
            TIME,
            Tank_Type,
            TankVol,
            TankCnt

        FROM (
                 SELECT
                     row_number()
                     OVER (
                         PARTITION BY TAG_PREFIX
                        ORDER BY Time DESC ) AS rk,
                     TAG_PREFIX,
                     TIME,
                     (nvl(TNK_CND_1_LVL, 0) +
                      nvl(TNK_CND_2_LVL, 0) +
                      nvl(TNK_CND_3_LVL, 0) +
                      nvl(TNK_CND_305A_LVL, 0) +
                      nvl(TNK_CND_305B_LVL, 0) +
                      nvl(TNK_CND_305C_LVL, 0) +
                      nvl(TNK_CND_305D_LVL, 0) +
                      nvl(TNK_CND_305E_LVL, 0) +
                      nvl(TNK_CND_305F_LVL, 0) +
                      nvl(TNK_CND_310A_LVL, 0) +
                      nvl(TNK_CND_310B_LVL, 0) +
                      nvl(TNK_CND_310C_LVL, 0) +
                      nvl(TNK_CND_310D_LVL, 0) +
                      nvl(TNK_CND_311_LVL, 0) +
                      nvl(TNK_CND_4_LVL, 0) +
                      nvl(TNK_CND_5_LVL, 0) +
                      nvl(TNK_CND_6_LVL, 0) +
                      nvl(TNK_CND_7_LVL, 0) +
                      nvl(TNK_CND_8_LVL, 0) +
                      nvl(TNK_CND_A_LVL, 0) +
                      nvl(TNK_CND_B_LVL, 0) +
                      nvl(TNK_CND_C_LVL, 0)) *
                     20                       AS TankVol,
                     GAS_VC,
                     CASE WHEN (TNK_CND_1_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_2_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_3_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_305A_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_305B_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_305C_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_305D_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_305E_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_305F_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_310A_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_310B_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_310C_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_310D_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_311_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_4_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_5_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_6_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_7_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_8_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_A_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_B_LVL) >= 0
                         THEN 1
                     ELSE 0 END +
                     CASE WHEN (TNK_CND_C_LVL) >= 0
                         THEN 1
                     ELSE 0 END               AS TankCnt,
                     'CND'                    AS Tank_Type
                 FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
                  --Where TIME >= trunc(sysdate-2)
        ) Vol
        WHERE Vol.TankVol > 0
              --AND TIME >= trunc(sysdate)
              --AND rk = 1
    '''

    cursor.execute(query)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)

    try:
        df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
        df.columns = [col.lower() for col in df.columns]
    except:
    	df = None
    	print('Dataframe is empty')

    cursor.close()
    connection.close()

    return df

def tank_split(df):
    base_df = df[['tag_prefix', 'time']]
    base_df['water'] = np.nan
    base_df['oil'] = np.nan
    base_df['total'] = np.nan

    base_df['water'] = base_df[(base_df[''])]

def off_by_date(df):
    lim_df = df[(df['TOT_LVL'] > df['FacilityCapacity']) & (df['FacilityCapacity'] != 0)]

    off_vals = []
    for date in lim_df['CalcDate'].unique():
        val = np.mean(df[df['CalcDate'] == date]['TOT_LVL'] - \
                      df[df['CalcDate'] == date]['FacilityCapacity'])
        off_vals.append(val)

    print('Averaging {} bbl per day.'.format(np.mean(off_vals)))

def plot_rate(df):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    facility = df['FacilityKey'].unique()[0]

    ax.plot(df['CalcDate'], df['TOT_rate'])

    plt.title('Liquid Rates for Facility {}'.format(facility))
    plt.xlabel('Date')
    plt.ylabel('bbl/day')
    plt.xticks(rotation='vertical')

    plt.savefig('images/rates/total/tot_rate_{}.png'.format(facility))


if __name__ == '__main__':
    # df = gwr_pull()
    # oracle_df = oracle_pull()
    # oracle_df.to_csv('data/oracle_gwr.csv')
    # df.to_csv('data/full_gwr.csv')

    # df = pd.read_csv('data/full_gwr.csv')
    oracle_df = pd.read_csv('data/oracle_gwr.csv')

    # off_by_date(df)

    # for facility in df['FacilityKey'].unique():
    #     plot_rate(df[df['FacilityKey'] == facility])
