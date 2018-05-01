import sys
sys.path.append('C:\\Program Files (x86)\\PIPC\\AF\\PublicAssemblies\\4.0\\')
import clr
clr.AddReference('OSIsoft.AFSDK')
clr.AddReference('System.Net')

from OSIsoft.AF.PI import *
from OSIsoft.AF.Search import *
from OSIsoft.AF.Asset import *
from OSIsoft.AF.Data import *
from OSIsoft.AF.Time import *
from System.Net import NetworkCredential

def connect_to_Server(serverName):
	piServers = PIServers()
	global piServer
	piServer = piServers[serverName]

	cred = NetworkCredential('piuser','')

	piServer.Connect(cred)

	tag = PIPoint.FindPIPoint(piServer, "sinusoid")
	lastData = tag.Snapshot()

	print('Snapshot Value:\nTimestamp: {0} Value: {1}\n\n'.format(lastData.Timestamp, lastData.Value))

	startTime = AFTime.Now
	count = 10
	forward = False;
	boundary = AFBoundaryType.Inside
	filter = ""
	includeFiltered = True

	archiveData = tag.RecordedValuesByCount(startTime, count, forward, boundary, filter, includeFiltered)

	print('Last 10 Archive Values:\n')
	for item in archiveData :
		print('Timestamp: {0} Value: {1}'.format(item.Timestamp, item.Value))

def get_tag_values(tagname,timestart,timeend):
    tag = PIPoint.FindPIPoint(piServer, tagname)
    timeRange = AFTimeRange(timestart,timeend)
    boundary = AFBoundaryType.Inside
    data = tag.RecordedValues(timeRange,boundary,'',False,0)
    dataList = list(data)
    print (len(dataList))
    results = np.zeros((len(dataList), 2), dtype='object')
    for i, sample in enumerate(data):
        results[i, :] = np.array([str(sample.Timestamp), str(sample.Value)])
    return results

def get_tag_snapshot(tagname):
	tag = PIPoint.FindPIPoint(piServer, tagname)
	lastData = tag.Snapshot()
	return lastData.Value, lastData.Timestamp


if __name__ == '__main__':
	PIthon.connect_to_Server('149.179.68.101')
	value, timestamp = PIthon.get_tag_snapshot('sinusoid')
	print('Timestamp: {0} Value: {1}'.format(timestamp, value))
