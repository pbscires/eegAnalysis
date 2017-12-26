'''

'''
from datetime import datetime, timedelta
import json
import re

jsonStr1 = '{"NumSeizures":1,"EndTime":"14:23:36","FileName":"chb03_01.edf","StartTime":"13:23:36","SeizureEndTimes":[414],"SeizureStartTimes":[362]}'
jsonStr2 = '{"NumSeizures":1,"EndTime":"12:41:33","FileName":"chb04_05.edf","StartTime":"10:02:37","SeizureEndTimes":[7853],"SeizureStartTimes":[7804]}'
jsonStr = '{"NumSeizures":1,"EndTime":"24:02:26","FileName":"chb10_27.edf","StartTime":"22:02:08","SeizureEndTimes":[2447],"SeizureStartTimes":[2382]}'


def getDateTimeObj(timeStr):
    dateStr = "2010-01-01"
    m = re.match(r'(\d+):(\d+):(\d+)', timeStr)
    if (m != None):
        (hh, mm, ss) = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        print ("hh mm ss = ", hh, mm, ss)
        if (hh >= 24):
            hh -= 24
            dateStr = "2010-01-02"
        print ("hh mm ss = ", hh, mm, ss)
        
        dateTimeStr = dateStr + " " + ':'.join([str(hh), str(mm), str(ss)])
        datetimeObj = datetime.strptime(dateTimeStr, "%Y-%m-%d %H:%M:%S")
        return datetimeObj
    else:
        return None

def getDateTimeObjFromSeconds(startTime, n_seconds):
    dateTimeObj = startTime + timedelta(seconds=n_seconds)
    return dateTimeObj

if __name__ == '__main__':
    jsonRoot = json.loads(jsonStr)
    seizureElem = jsonRoot
    fileName = seizureElem['FileName']
    fileStartTime = seizureElem['StartTime']
    fileEndTime = seizureElem['EndTime']
    seizureStartTimes = seizureElem['SeizureStartTimes']
    seizureEndTimes = seizureElem['SeizureEndTimes']
    
    print ("fileName = ", fileName)
    print ("fileStartTime = ", fileStartTime)
    print ("fileEndTime = ", fileEndTime)
    print ("seizureStartTimes = ", seizureStartTimes)
    print ("seizureEndTimes = ", seizureEndTimes)
    
    fileStartTime = getDateTimeObj(fileStartTime)
    
    fileEndTime = getDateTimeObj(fileEndTime)
    print ("fileStartTime = ", fileStartTime, ", fileEndTime = ", fileEndTime)

#     startTime = datetime.strptime(fileStartTime, "%Y-%m-%d %H:%M:%S")
#     endTime = datetime.strptime(fileEndTime, "%Y-%m-%d %H:%M:%S")
    diffTime = fileEndTime - fileStartTime
    print ("startTime = ", fileStartTime, ", endTime = ", fileEndTime, ", diffTime = ", diffTime.seconds)
    
    for timeInSeconds in seizureStartTimes:
        datetimeObj = getDateTimeObjFromSeconds(fileStartTime, int(timeInSeconds))
        print ("Seizure start time = ", datetimeObj)
    for timeInSeconds in seizureEndTimes:
        datetimeObj = getDateTimeObjFromSeconds(fileStartTime, int(timeInSeconds))
        print ("Seizure end time = ", datetimeObj)
    
#     print ("Testing time addition: ", fileStartTime + timedelta(seconds=7200))