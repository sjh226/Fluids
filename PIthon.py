import sys

sys.path.append('C:\\Program Files (x86)\\PIPC\\AF\\PublicAssemblies\\4.0\\')
import clr

clr.AddReference('OSIsoft.AFSDK')
clr.AddReference('System.Net')
import numpy as np
from OSIsoft.AF.PI import *
from OSIsoft.AF.Search import *
from OSIsoft.AF.Asset import *
from OSIsoft.AF.Data import *
from OSIsoft.AF.Time import *
from System.Net import NetworkCredential
from System import TimeSpan


def connect_to_Server(serverName='149.179.68.101', verbose=False):
    piServers = PIServers()
    global piServer
    piServer = piServers[serverName]

    cred = NetworkCredential('piuser', '')

    piServer.Connect(cred)

    if verbose:
        tag = PIPoint.FindPIPoint(piServer, "sinusoid")
        lastData = tag.Snapshot()

        print('Snapshot Value:\nTimestamp: {0} Value: {1}\n\n'.format(lastData.Timestamp, lastData.Value))

        startTime = AFTime.Now
        count = 10
        forward = False
        boundary = AFBoundaryType.Inside
        filter = ""
        includeFiltered = True

        archiveData = tag.RecordedValuesByCount(startTime, count, forward, boundary, filter, includeFiltered)

        print('Last 10 Archive Values:\n')
        for item in archiveData:
            print('Timestamp: {0} Value: {1}'.format(item.Timestamp, item.Value))


def get_tag_snapshot(tagname):
    tag = PIPoint.FindPIPoint(piServer, tagname)
    lastData = tag.Snapshot()
    return lastData.Value, lastData.Timestamp

def print_tagnames_for_one_well():
    '''Prints an example of tag names, hard coded but useful as a
    jumping off point, star is the tag prefix, for example 'WAM_USANL20-60'.
    A full tagname is the combination '<well_name>.<tag>' for air temp it is
    'WAM_USANL20-60.AIR_TMP'

    Arguments:
    ----------------------------------
    None -> Just prints

    Returns:
    ----------------------------------
    my_list (list) -> A list of possible tagnames

    '''
    my_list = ['*.AIR_TMP', '*.API_NUM', '*.BATT_12VDC', '*.BATT_5VDC', '*.CAS_PSI',
               '*.CAS_PSI_24H_AVG', '*.CATHODIC', '*.CCS_FLG', '*.CCS_RTC', '*.CCS_RTY',
               '*.CCS_ST', '*.CCS_ST_ACCUM_TM', '*.CCS_VC', '*.CCS_VY', '*.CND_OS_ST', '*.CND_PSI',
               '*.CND_VF_ST', '*.CND_VLV_ST', '*.CND_VLV_ST_ACCUM_TM', '*.COLDBOOT', '*.COM_STATUS',
               '*.COM_STATUS_DTC', '*.COM_STATUS_DTY', '*.COM_STATUS_RTC', '*.COM_STATUS_RTY',
               '*.CONCAT_CCS_BR', '*.CRTL_PARM_CTRL_GAIN', '*.CRTL_VAL_FAIL', '*.CTRL_ARM',
               '*.CTRL_O_C_FLG', '*.CTRL_PRM_CHK_TRAVL_TM', '*.CTRL_PRM_CONSC_OCR_DP_SP_ADJ',
               '*.CTRL_PRM_CTRL_INTVL_SEC',
               '*.CTRL_PRM_DLY_AFT_SI_MET', '*.CTRL_PRM_DP_ADJ_MODE', '*.CTRL_PRM_DP_ADJ_STEP',
               '*.CTRL_PRM_DP_MAX_CHG_SP',
               '*.CTRL_PRM_DP_RANGE', '*.CTRL_PRM_DP_SI_SP', '*.CTRL_PRM_DROP_RT_MULT', '*.CTRL_PRM_FLW_RAT_SP',
               '*.CTRL_PRM_HI_DP_LIM', '*.CTRL_PRM_LOW_DP_LIM', '*.CTRL_PRM_LOW_DP_SP', '*.CTRL_PRM_MAX_OPN_PSI',
               '*.CTRL_PRM_MAX_OPN_PSI_SP', '*.CTRL_PRM_MIN_MIN_UP_ADJ', '*.CTRL_PRM_MIN_OPN_PSI',
               '*.CTRL_PRM_MIN_OPN_TM_AFT_PLUN_ARR',
               '*.CTRL_PRM_MIN_SI_TM', '*.CTRL_PRM_OPEN_CRIT_TYP', '*.CTRL_PRM_PLN_SPED_SP',
               '*.CTRL_PRM_PLUN_FAIL_ADJ_PCNT',
               '*.CTRL_PRM_PSI_ADJ_FCTR', '*.CTRL_PRM_SI_CAS_PSI_BLD_UP', '*.CTRL_PRM_START_OPN_PSI',
               '*.CTRL_PRM_SWAB_MODE',
               '*.CTRL_PRM_TUB_STRN_DEPTH', '*.CTS_RTC', '*.CTS_RTY', '*.CTS_VC',
               '*.CTS_VY', '*.CYC_ALRM', '*.CYC_CTRL_ST', '*.CYC_GAS_FR_SP', '*.CYC_LIN_PSI', '*.CYC_LO_CAS_PSI',
               '*.CYC_PSI_SP', '*.CYC_RUN_TM', '*.CYC_SHUT_TM', '*.CYC_VELOCITY',
               '*.CYC_VOL', '*.DEHY_DP', '*.DEHY_TMP', '*.E_FLOW', '*.EFM_PLATE', '*.GAS_DT',
               '*.GAS_DTC', '*.GAS_DTY', '*.GAS_FR', '*.GAS_MIDNIGHT_VY', '*.GAS_RTC', '*.GAS_RTY',
               '*.GAS_VC', '*.GAS_VY', '*.GLY_P1_TMP', '*.GLY_P2_TMP', '*.JACCODE', '*.LIN_HTR_TMP',
               '*.LST_ARR_TM', '*.LST_PLN_SPED', '*.LST_PLN_TM', '*.MODBUS_FAIL',
               '*.MODBUS_FAIL_CUR', '*.OFF_REPORT', '*.OFFLINE', '*.PLN_CYC_CNT_CUR',
               '*.PLN_FLG', '*.PLN_NON_ARVL_CNT', '*.PROD_CTRL_VAL_PV', '*.RTU_IP',
               '*.SALES_LINE_PSI', '*.SI_FLG', '*.TNK_1_TMP', '*.TNK_1_TOT_LVL',
               '*.TNK_1_WAT_LVL', '*.TNK_2_TMP', '*.TNK_2_TOT_LVL', '*.TNK_2_WAT_LVL',
               '*.TNK_HI_ST', '*.TNK_ST', '*.TUB_PSI', '*.TUB_PSI_24H_AVG',
               '*.UNT_MAINT_CNT', '*.UNT_PSI', '*.UNT_TMP', '*.WAT_VC', '*.WAT_VY', '*.Z:SCC_FLG']
    return my_list

def get_tag_values(tagname, timestart, timeend):
    tag = PIPoint.FindPIPoint(piServer, tagname)
    timeRange = AFTimeRange(timestart, timeend)
    boundary = AFBoundaryType.Inside
    data = tag.RecordedValues(timeRange, boundary, '', False, 0)
    dataList = []
    for i, sample in enumerate(data):
        dataList.append([str(sample.Timestamp), str(sample.Value)])
    return dataList


def get_tag_interpolate(tagname, timestart, timeend, seconds=0, minutes=0, hours=0):
    '''Get interpolated data. This query fetches interpolated data based on the
    tagname in the timestart and timeend range. The time interval is passed
    through the second, minute and hours arguments

    List of what I know:
        't'  -> Beginning of today, end of last night, about midnight
        's'  -> seconds
        'm'  -> minutes
        'h'  -> hour
        'd'  -> day
        'y'  -> year
        '8m' -> 8 minutes
        '-8m'-> 8 minutes ago
        '+8m'-> 8 minutes from now
        '04/01/2018' -> 12:00:00 AM on April 1st, 2018


    Arguments:
    ----------------------------------
    tagname (str) -> A well name and the tag you are looking for
    timestart (str) -> The start time, don't pass datetime objects in here
    timeend (str) -> The end time, don't pass datetime objects in here
    seconds (int) -> Seconds between intervals (careful, this will generate a
                    lot of data)
    minutes (int) -> Minutes between intervals
    hours (int) -> hours between intervals
    Returns:
    ----------------------------------
    data (list of lists) -> A list of lists, where each element contains a list
        consisting of a timestamp (str) and values(str).

    Ex.
    ----------------------------------
    data = get_tag_interpolate('WAM-CH222D5.TUB_PSI','-8d','t', seconds=0,
                 minutes=15, hours=0)
    data

    <<< [['4/23/2018 12:00:00 AM', '394.8289'],
            ['4/23/2018 12:15:00 AM', '402.7233'],
            ['4/23/2018 12:30:00 AM', '212.2917']...]
    '''
    tag = PIPoint.FindPIPoint(piServer, tagname)
    timeRange = AFTimeRange(timestart, timeend)
    full_timespan = TimeSpan(hours, minutes, seconds)
    timeSpan = AFTimeSpan(full_timespan)
    data = tag.InterpolatedValues(timeRange, timeSpan, '', False)
    dataList = []
    for i, sample in enumerate(data):
        dataList.append([str(sample.Timestamp), str(sample.Value)])
    return dataList


if __name__ == '__main__':
	PIthon.connect_to_Server('149.179.68.101')
	value, timestamp = PIthon.get_tag_snapshot('sinusoid')
	print('Timestamp: {0} Value: {1}'.format(timestamp, value))
