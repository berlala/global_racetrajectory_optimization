# GPS Tracker Data Process

#TODO: 引入OSMNx进行地图投影

import numpy as np
import csv
import math 
import matplotlib.pyplot as plt

def wgs84_to_webMercator(lng,lat):
    # Convert WGS84 to WebMercator in Meters
    x = lng*20037508.342789 / 180.0
    y = math.log(math.tan((90+lat)*math.pi/360))/(math.pi / 180.0) # 注意括号
    y = y *20037508.342789 / 180.0
    return x,y


# Date(GMT),Date(Local),Time(sec),Latitude,Longitude,Horizontal Accuracy(m),
# Altitude(m),Vertical Accuracy(m),Distance(m),Speed(m/s),Average Speed(m/s),Course(deg),
# True Heading(deg),Magnetic Heading(deg),Heading Accuracy(deg),Glide Ratio,Heart Rate (bpm)
with open('FridayNightTrack.csv', mode='r',encoding='utf-8',newline='') as f:
    reader = csv.reader(f)
    t = []
    Lat = []
    Long = []
    Alt  = []
    Dis = []
    Spd = []
    Heading = []
    x_m = []
    y_m = []
    index = 0
    for row in reader:
        if index > 0:
            t.append(row[2])
            Lat.append(row[3])
            Long.append(row[4])
            Alt.append(row[6])
            Dis.append(row[8])
            Spd.append(row[9])
            Heading.append(row[12])
            x,y = wgs84_to_webMercator(float(row[4]), float(row[3]))
            x_m.append(x)
            y_m.append(y)
        index = index+1

data = np.vstack((t,Lat,Long,Alt,Dis))

x_m = np.asarray(x_m) - x_m[0]
y_m = np.asarray(y_m) - y_m[0]
left_m = 4*np.ones(len(x_m))
right_m = 4*np.ones(len(x_m))

data_2 = np.vstack((x_m,y_m,left_m,right_m))
result = np.transpose(data_2)

np.savetxt('fridaytrack.csv',result,delimiter=',')

plt.plot(x_m,y_m)
plt.title('Map')
plt.axis('equal')
plt.show()
