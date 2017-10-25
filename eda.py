import pandas as pd
import numpy as np
import pyodbc
import sys


def enbase_fetch():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=EDW;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT GDP._id
              ,GDP.totalRunTicketWater
              ,GDP.newWaterInventory AS GDPnewWater
              ,GDP.newOilInventory AS GDPnewOil
        	  ,GDD.tankCode
        	  ,GDD.equipmentId
        	  ,GDD.liquidGaugeFeet
        	  ,GDD.liquidGaugeInches
        	  ,GDD.liquidGaugeQuarter
        	  ,GDD.waterGaugeFeet
        	  ,GDD.waterGaugeInches
        	  ,GDD.waterGaugeQuarter
        	  ,GDD.newWaterInventory AS GDDnewWater
        	  ,GDD.newOilInventory AS GDDnewOil
              ,GDD.createdDate
              ,Tk.MaxVol
              ,Tk.Height
              ,Tk.TankType
              ,RT.assetId
              ,RT.runTicketStartDate
              ,RT.ticketType
              ,RT.grossVolume
              ,RT.openWaterVolume
              ,RT.dispositionType
              ,RT.transferTankCode
        FROM [EDW].[Enbase].[GaugeDataParent] AS GDP
        JOIN [EDW].[Enbase].[GaugeDataDetailed] AS GDD
            ON GDP._id = GDD.parentId
        JOIN [OperationsDataMart].[Dimensions].[Tanks] AS Tk
            ON GDD.tankCode = Tk.tankCode
        JOIN [EDW].[Enbase].[RunTicket] AS RT
            ON GDD.jobId = RT.jobId
        ORDER BY GDD.tankName, RT.runTicketStartDate;
    """)
        # ,GDP.totalBarrelsWater
        # ,GDP.totalBarrelsOil
        # ,GDP.comments
        # ,RT.openTemperature
        # ,RT.colorCutFeet
        # ,RT.colorCutInches
        # ,RT.colorCutFract
        # ,RT.closeMeasFeet
        # ,RT.closeMeasInches
        # ,RT.closeMeasFract
        # ,RT.closeTemperature
        # ,RT.colorCutCloseFeet
        # ,RT.colorCutCloseInches
        # ,RT.colorCutCloseFract
        # ,RT.avgDipVal
        # ,RT.openMeasFeet
        # ,RT.openMeasInches
        # ,RT.openMeasFract
        # ,RT.tankAllWater
        # ,GDP.oldWaterInventory
        # ,GDP.rateWater
        # ,GDP.oldOilInventory
        # ,GDP.totalRunTicketOil
        # ,GDP.rateOil
        # ,GDP.oldRateWater
        # ,GDP.oldRateOil

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

def ticket_fetch():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=EDW;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT   assetId,
                 assetName,
                 runTicketStartDate,
                 ticketType,
                 tankNameSelect,
                 runTicketTankName,
                 tankCode,
                 equipmentId,
                 avgDipVal,
                 openMeasFeet,
                 openMeasInches,
                 openMeasFract,
                 tankAllWater,
                 openTemperature,
                 colorCutFeet,
                 colorCutInches,
                 colorCutFract,
                 closeMeasFeet,
                 closeMeasInches,
                 closeMeasFract,
                 closeTemperature,
                 colorCutCloseFeet,
                 colorCutCloseInches,
                 colorCutCloseFract,
                 grossVolume,
                 openWaterVolume,
                 dispositionType,
                 tankHeight,
                 transferTankName,
                 transferTankCode,
                 runTicketAction
        FROM Enbase.RunTicket AS RT;
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

    return df

def rtr_fetch():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=EDW;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT DDH.Well1_WellFlac, DDH.DateTime, DDH.Meter1_DiffPress,
               DDH.Meter1_DiffPressPDayAvg, DDH.Meter1_VolumeCDay,
               DDH.Meter1_VolumePDay, DDH.Meter1_StaticPress,
               DDH.Meter1_StaticPressPDayAvg, DDH.Meter1_FlowRate,
               DDH.Meter1_Temperature
        FROM RTR.DataDailyHistory AS DDH;
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

    return df

def rtr_clean(df, single=False):
    df_out = df[df['Well1_WellFlac'].notnull()]
    max_rows = 0
    max_well = None
    for well in df_out['Well1_WellFlac'].unique():
        row_count = df_out[df_out['Well1_WellFlac'] == well].shape[0]
        if row_count > max_rows:
            max_rows = row_count
            max_well = well
    if single == True:
        df_out = df_out[df_out['Well1_WellFlac'] == max_well]
    return df_out

def clean_data(df, single=False):
    df_out = df
    max_rows = 0
    max_tank = None
    for tank in df_out['tankCode'].unique():
        row_count = df_out[df_out['tankCode'] == tank].shape[0]
        if row_count > max_rows:
            max_rows = row_count
            max_tank = tank
    if single == True:
        df_out = df_out[df_out['tankCode'] == max_tank]
    df_out = df_out.drop_duplicates()
    return df_out

if __name__ == '__main__':
    df_meas = enbase_fetch()
    df_max = clean_data(df_meas, single=True)
    # df_ticket = ticket_fetch()

    # df_meas.to_csv('enb.csv')
    # df_ticket.to_csv('ticket.csv')
