import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import sys


def lgr_pull():
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
        DROP TABLE IF EXISTS #TOT_Sites;
        DROP TABLE IF EXISTS #CND_Sites;
        DROP TABLE IF EXISTS #WAT_Sites;

        SELECT [TAG_PREFIX]
        INTO [#TOT_Sites]
        FROM   TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        WHERE  [CalcDate] =
               (SELECT MAX([CalcDate])
                FROM   [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
            AND [Tank_Type] = 'TOT';
        SELECT [TAG_PREFIX]
        INTO [#WAT_Sites]
        FROM   TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        WHERE  [CalcDate] =
               (
                   SELECT MAX([CalcDate])
                   FROM   [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
                 AND [Tank_Type] = 'WAT';
        SELECT [TAG_PREFIX]
        INTO [#CND_Sites]
        FROM   TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
        WHERE  [CalcDate] =
               (
                   SELECT MAX([CalcDate])
                   FROM   [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
                 AND [Tank_Type] = 'CND';

        DROP TABLE IF EXISTS #FacilityLevels;
        SELECT ROW_NUMBER() OVER(PARTITION BY [facilityKey],
                                                          [Tank_type] ORDER BY [DateKey] DESC) AS [rk],
                 [FacilityKey],
                 [W].[API],
                 [PT].[TAG_PREFIX],
                 [DateKey],
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
        FROM   TeamOptimizationEngineering.Reporting.PI_Tanks AS PT
                 INNER JOIN #TOT_Sites AS TS ON TS.TAg_Prefix = PT.Tag_Prefix
                 INNER JOIN #WAT_Sites AS WS ON WS.TAg_Prefix = PT.Tag_Prefix
                INNER JOIN TeamOptimizationEngineering.Reporting.PITag_Dict AS PD ON PD.TAG = PT.Tag_Prefix
                                                                                                            AND Confidence = 100
                 INNER JOIN OperationsDataMart.Dimensions.Wells AS W ON W.API = PD.Api
        WHERE  [CalcDate] =
               (
                   SELECT MAX([CalcDate])
                   FROM   [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
                 AND [PT].[Tag_Prefix] NOT IN
               (
                   SELECT *
                   FROM   [#CND_Sites]
        );
        DROP TABLE IF EXISTS #Final1;
        SELECT DISTINCT
                 [TL].[Facilitykey],
                 [F].[FacilityName],
                 CASE
                     WHEN SUM([Tank_Mult]) OVER(PARTITION BY [TL].[FacilityKey]) = 0
                     THEN SUM([Tank_Mult] * [TankLVL]) OVER(PARTITION BY [TL].[FacilityKey])
                     ELSE NULL
                 END AS [CND_LVL],
                 [WAT_LVL],
                 [TOT_LVL]
        INTO [#Final1]
        FROM           #FacilityLevels AS TL
                            INNER JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                                                                                        AND F.TankCount -1 <= TL.TankCNT
                            LEFT JOIN
               (
                   SELECT DISTINCT
                            [TL].[Facilitykey],
                            CASE
                                   WHEN [TANK_TYPE] = 'TOT'
                                   THEN [TankLVL]
                                   ELSE NULL
                            END AS [TOT_LVL]
                   FROM   #FacilityLevels AS TL
                            INNER JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                                                                                                            AND F.TankCount -1 <= TL.TankCNT
                   WHERE  [RK] = 1
                            AND [TANK_TYPE] = 'TOT'
        ) AS Tot ON Tot.Facilitykey = TL.Facilitykey
                            LEFT JOIN
               (
                   SELECT DISTINCT
                            [TL].[Facilitykey],
                            CASE
                                   WHEN [TANK_TYPE] = 'WAT'
                                   THEN [TankLVL]
                                   ELSE NULL
                            END AS [WAT_LVL]
                   FROM   #FacilityLevels AS TL
                            INNER JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                                                                                                            AND F.TankCount -1 <= TL.TankCNT
                   WHERE  [RK] = 1
                            AND [TANK_TYPE] = 'WAT'
        ) AS WAT ON WAT.Facilitykey = TL.Facilitykey
        WHERE [RK] = 1;

        DROP TABLE IF EXISTS #FacilityLevels2;

        SELECT ROW_NUMBER() OVER(PARTITION BY [facilityKey],
                                                          [Tank_type] ORDER BY [DateKey] DESC) AS [rk],
                 [FacilityKey],
                 [W].[API],
                 [PT].[TAG_PREFIX],
                 [DateKey],
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
                 INNER JOIN #CND_Sites AS TS ON TS.TAg_Prefix = PT.Tag_Prefix
                 INNER JOIN #WAT_Sites AS WS ON WS.TAg_Prefix = PT.Tag_Prefix
                 INNER JOIN TeamOptimizationEngineering.Reporting.PITag_Dict AS PD ON PD.TAG = PT.Tag_Prefix
                                                                                                            AND Confidence = 100
                 INNER JOIN OperationsDataMart.Dimensions.Wells AS W ON W.API = PD.Api
        WHERE  [CalcDate] =
               (
                   SELECT MAX([CalcDate])
                   FROM   [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        )
                 AND [PT].[Tag_Prefix] NOT IN
               (
                   SELECT *
                   FROM   [#TOT_Sites]
        );

        DROP TABLE IF EXISTS #Final2;
        SELECT DISTINCT
                 [TL].[Facilitykey],
                 [F].[FacilityName],
                 [CND_LVL],
                 [WAT_LVL],
                 [CND_LVL] + [WAT_LVL] AS [TOT_LVL]
        INTO [#Final2]
        FROM           #FacilityLevels2 AS TL
                            INNER JOIN OperationsDataMart.Dimensions.Facilities AS F ON f.Facilitykey = TL.Facilitykey
                            LEFT JOIN
               (
                   SELECT DISTINCT
                            [TL].[Facilitykey],
                            SUM(CASE
                                       WHEN [RK] = 1
                                       THEN [TANKCNT]
                                       ELSE 0
                                   END) OVER(PARTITION BY [TL].[Facilitykey]) AS [TANKCNT]
                   FROM   #FacilityLevels2 AS TL
        ) AS CNT ON F.Facilitykey = CNT.Facilitykey
                       AND F.TankCount - 1 <= CNT.TANKCNT
                            LEFT JOIN
               (
                   SELECT DISTINCT
                            [TL].[Facilitykey],
                            CASE
                                   WHEN [TANK_TYPE] = 'CND'
                                   THEN [TankLVL]
                                   ELSE NULL
                            END AS [CND_LVL]
                   FROM   #FacilityLevels2 AS TL
                   WHERE  [RK] = 1
                            AND [TANK_TYPE] = 'CND'
        ) AS Tot ON Tot.Facilitykey = TL.Facilitykey
                            LEFT JOIN
               (
                   SELECT DISTINCT
                            [TL].[Facilitykey],
                            CASE
                                   WHEN [TANK_TYPE] = 'WAT'
                                   THEN [TankLVL]
                                   ELSE NULL
                            END AS [WAT_LVL]
                   FROM   #FacilityLevels2 AS TL
                   WHERE  [RK] = 1
                            AND [TANK_TYPE] = 'WAT'
        ) AS WAT ON WAT.Facilitykey = TL.Facilitykey
        WHERE [RK] = 1
        ORDER BY [FacilityName];
        DROP TABLE IF EXISTS #LGRV7;

        SELECT *
        INTO [#LGRV7]
        FROM     #Final1
        UNION ALL
        SELECT *
        FROM   #Final2;

        SELECT * FROM #LGRV7
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

    return df.drop_duplicates()


if __name__ == '__main__':
    main()
