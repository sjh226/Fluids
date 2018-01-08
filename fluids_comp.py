import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt


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
        SELECT DW.API
        	  ,DW.WellName
              ,LGR.TotalOilOnSite AS LGROil
              ,LGR.TotalWaterOnSite AS LGRWater
              ,LGR.FacilityCapacity
              ,LGR.CalcDate
        FROM [TeamOptimizationEngineering].[dbo].[InventoryAll] AS LGR
        JOIN (SELECT	FacilityKey
        				,MAX(CalcDate) maxtime
        		FROM [TeamOptimizationEngineering].[dbo].[InventoryAll]
        		GROUP BY FacilityKey, DAY(CalcDate), MONTH(CalcDate), YEAR(CalcDate)) AS MD
        	ON	MD.FacilityKey = LGR.FacilityKey
        	AND	MD.maxtime = LGR.CalcDate
        JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
        	ON LGR.FacilityKey = DW.FacilityKey
        JOIN [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
        	ON PTD.API = DW.API;
    """)

    # WHERE	LGR.FacilityKey IN (
	# 	SELECT FacilityKey
	# 	FROM [TeamOptimizationEngineering].[dbo].[InventoryAll]
	# 	GROUP BY FacilityKey
	# 	HAVING	SUM(TotalOilOnSite) > 0
	# 		AND	SUM(TotalWaterOnSite) > 0
	# 		AND CAST(MAX(CalcDate) AS DATE) = CAST(GETDATE() AS DATE)
	# 		AND COUNT(*) >= 31)

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
        SELECT	PIT.TAG_PREFIX
        		,TANK_TYPE
        		,TANKLVL
        		,CalcDate
        FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks] AS PIT
        JOIN (SELECT	TAG_PREFIX
        				,MAX(CalcDate) maxtime
        	FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        	GROUP BY TAG_PREFIX, DAY(CalcDate), MONTH(CalcDate), YEAR(CalcDate)) AS MD
        ON	MD.TAG_PREFIX = PIT.TAG_PREFIX
        AND	MD.maxtime = PIT.CalcDate;
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

    new_df = df[['TAG_PREFIX', 'CalcDate', 'TANKLVL']]
    new_df = new_df.groupby(['TAG_PREFIX', 'CalcDate'], as_index=False).sum()


    def water(row):
        wat = df[(df['TAG_PREFIX'] == row['TAG_PREFIX']) & (df['CalcDate'] == row['CalcDate'])\
                 & (df['TANK_TYPE'] == 'WAT')]['TANKLVL'].values
        try:
            val = wat[0]
        except:
            val = np.nan
        return val

    def cond(row):
        cond = df[(df['TAG_PREFIX'] == row['TAG_PREFIX']) & (df['CalcDate'] == row['CalcDate'])\
                 & (df['TANK_TYPE'] == 'CND')]['TANKLVL'].values
        try:
            val = cond[0]
        except:
            val = np.nan
        return val

    def total(row):
        tot = df[(df['TAG_PREFIX'] == row['TAG_PREFIX']) & (df['CalcDate'] == row['CalcDate'])\
                 & (df['TANK_TYPE'] == 'TOT')]['TANKLVL'].values
        try:
            val = tot[0]
        except:
            val = np.nan
        return val

    new_df['water'] = new_df.apply(water, axis=1)
    new_df['oil'] = new_df.apply(cond, axis=1)
    new_df['total'] = new_df.apply(total, axis=1)

    return df.drop_duplicates()

def plot_neg(df):
    plt.close()


if __name__ == '__main__':
    # df = lgr_pull()
    df = gwr_pull()
