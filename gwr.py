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

def tag_dict():
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
		SELECT  PTD.TAG AS tag_prefix
				,PTD.API
				,DF.Facilitykey
				,DF.FacilityCapacity
				,DF.FacilityName
		FROM [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
		JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
			ON PTD.API = DW.API
		JOIN [TeamOptimizationEngineering].[dbo].[DimensionsFacilities] AS DF
			ON DW.Facilitykey = DF.Facilitykey
		GROUP BY PTD.TAG, PTD.API, DF.Facilitykey, DF.FacilityCapacity, DF.FacilityName;
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

def tank_count():
	try:
		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
									r'Server=SQLDW-L48.BP.Com;'
									r'Database=OperationsDataMart;'
									r'trusted_connection=yes'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		SELECT DF.Facilitykey
			   ,DF.FacilityName
			   ,DF.TankCount
		FROM [TeamOptimizationEngineering].[dbo].[DimensionsFacilities] DF
		WHERE DF.BusinessUnit = 'North';
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

def map_tag(vol, tag):
	df = vol.merge(tag, on='tag_prefix', how='left')
	# df = df.drop(['tag_prefix', 'API'], axis=1)
	# df = df.dropna()
	df['oil_rate'] = df['oil'] - df['oil'].shift(1)
	df.loc[df['oil_rate'] < 0, 'oil_rate'] = np.nan
	df['oil_rate']
	df['time'] = pd.to_datetime(df['time'])
	df = df.groupby(['Facilitykey', 'time', 'FacilityCapacity', 'FacilityName', 'tag_prefix', 'tankcnt'], as_index=False).mean()
	df = df.groupby(['Facilitykey', 'time', 'FacilityCapacity', 'FacilityName', 'tag_prefix'], as_index=False).max()
	return df.sort_values(['Facilitykey', 'time'])

def tank_split(df):
	water_df = df[df['tank_type'] == 'WAT'][['tag_prefix', 'time', 'tankvol', 'tankcnt']]
	water_df.columns = ['tag_prefix', 'time', 'water', 'tankcnt']
	oil_df = df[df['tank_type'] == 'CND'][['tag_prefix', 'time', 'tankvol', 'tankcnt']]
	oil_df.columns = ['tag_prefix', 'time', 'oil', 'tankcnt']
	total_df = df[df['tank_type'] == 'TOT'][['tag_prefix', 'time', 'tankvol', 'tankcnt']]
	total_df.columns = ['tag_prefix', 'time', 'total', 'tankcnt']

	base_df = water_df.merge(oil_df, on=['tag_prefix', 'time', 'tankcnt'], how='outer')
	df = base_df.merge(total_df, on=['tag_prefix', 'time', 'tankcnt'], how='outer')

	df.loc[df['oil'].isnull(), 'oil'] = df.loc[df['oil'].isnull(), 'total'] - \
										df.loc[df['oil'].isnull(), 'water']

	return df

def tank_na_fill(df):
	result_df = pd.DataFrame(columns=df.columns)
	for tank in df['tag_prefix'].unique():
		tank_df = df[df['tag_prefix'] == tank].fillna(method='ffill')
		result_df = result_df.append(tank_df)
	return result_df

def off_by_date(df):
	lim_df = df[(df['TOT_LVL'] > df['FacilityCapacity']) & (df['FacilityCapacity'] != 0)]

	off_vals = []
	for date in lim_df['CalcDate'].unique():
		val = np.mean(df[df['CalcDate'] == date]['TOT_LVL'] - \
					  df[df['CalcDate'] == date]['FacilityCapacity'])
		off_vals.append(val)

	print('Averaging {} bbl per day.'.format(np.mean(off_vals)))

def total_plot(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	# df = df[df['time'] >= df['time'].max() - pd.Timedelta('31 days')]

	facility = df['FacilityName'].unique()[0]
	# capacity = df['FacilityCapacity'].unique()[0]

	ax.plot(df['time'], df['oil'])
	# ax.axhline(capacity, linestyle='--', color='#920f25', label='Facility Capacity')

	plt.title('Oil GWR Volumes for Facility {}'.format(facility))
	plt.xlabel('Date')
	plt.ylabel('bbl')

	ymin, ymax = plt.ylim()
	if ymin > 0:
		plt.ylim(ymin=0)

	plt.xticks(rotation='vertical')
	plt.tight_layout()

	plt.savefig('images/gwr/good/oil_{}.png'.format(facility))

def plot_rate(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	facility = df['Facilitykey'].unique()[0]

	ax.plot(df['time'], df['oil_rate'])

	plt.title('Liquid Rates for Facility {}'.format(facility))
	plt.xlabel('Date')
	plt.ylabel('bbl/day')

	cnt = 0
	if len(ax.xaxis.get_ticklabels()) > 12:
		for label in ax.xaxis.get_ticklabels():
			if cnt % 7 == 0:
				label.set_visible(True)
			else:
				label.set_visible(False)
			cnt += 1

	plt.xticks(rotation='vertical')

	plt.savefig('images/rates/total/tot_rate_{}.png'.format(facility))

def tank_merge(pi_df, sql_df):
	pi_df['gwr_tanks'] = pi_df['tankcnt']
	pi_df = pi_df[['Facilitykey', 'tankcnt']]
	df = pi_df.merge(sql_df, on='Facilitykey', how='outer')
	df.fillna(0, inplace=True)
	return df.drop_duplicates()

def best_tanks(best_df):
	df = best_df[['Facilitykey', 'FacilityName', 'time', 'water', 'oil']]
	for facility in df['Facilitykey'].unique():
		total_plot(df[df['Facilitykey'] == facility])

def turbine_gwr_pull():
	oracle_df = oracle_pull()
	df = tank_split(oracle_df)
	df.drop('total', axis=1, inplace=True)
	# df.dropna(inplace=True)
	tag_df = tag_dict()
	gwr_df = map_tag(df, tag_df)
	tank_df = tank_count()
	tank_df = tank_merge(gwr_df, tank_df)
	tag_list = ['WAM-ML11_150H', 'WAM-ML11_160H', 'WAM-BB19', 'WAM-CL29_150H', \
                'WAM-CH320C1', 'WAM-HP13_150H', 'WAM-HP13_150H', \
                'WAM-CL29_160H', 'WAM-LM8_115H']
	match_df = tank_df[tank_df['tankcnt'] == tank_df['TankCount']]
	# gwr_df = df[df['Facilitykey'].isin(match_df['Facilitykey'])]
	return gwr_df, tank_df


if __name__ == '__main__':
	# # df = gwr_pull()
	# oracle_df = oracle_pull()
	# tag_df = tag_dict()
	# # oracle_df.to_csv('data/oracle_gwr.csv')
	# # df.to_csv('data/full_gwr.csv')

	this, that = turbine_gwr_pull()

	# # df = pd.read_csv('data/full_gwr.csv')
	# oracle_df = pd.read_csv('data/oracle_gwr.csv')
	#
	# df = tank_split(oracle_df)
	# df.drop('total', axis=1, inplace=True)
	# df.dropna(inplace=True)
	# # df = tank_na_fill(df)
	# # df.to_csv('data/nan_vol_df.csv')
	#
	# # vol_df = pd.read_csv('data/nan_vol_df.csv')
	# tank_df = tank_count()
	# # vol_df = vol_df.dropna()
	# # vol_df = vol_df.dropna(subset=['oil'])
	# df = map_tag(df, tag_df)
	#
	# tank_df = tank_merge(df, tank_df)
	# match_df = tank_df[tank_df['tankcnt'] == tank_df['TankCount']]
	#
	# gwr_df = df[df['Facilitykey'].isin(match_df['Facilitykey'])]
	# best_tanks(gwr_df)
	#
	# # off_by_date(df)
	#
	# # for facility in match_df['Facilitykey'].unique():
	# #     total_plot(df[df['Facilitykey'] == facility])
	#     # break
