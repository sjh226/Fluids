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

def get_tag_snapshot(tagname):
    tag = PIPoint.FindPIPoint(piServer, tagname)
    lastData = tag.Snapshot()
    return lastData.Value, lastData.Timestamp
