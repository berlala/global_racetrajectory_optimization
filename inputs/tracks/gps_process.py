# GPS Tracker Data Process
# 使用folium库进行地图投影

import numpy as np
import csv
import math 
import matplotlib.pyplot as plt
import folium
import webbrowser

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
    coordinate=[]
    index = 0
    for row in reader:
        if index > 0:
            t.append(row[2])
            Lat.append(row[3]) #纬度坐标
            Long.append(row[4]) #经度坐标
            coordinate.append([float(row[3]),float(row[4])])
            Alt.append(row[6])
            Dis.append(row[8])
            Spd.append(float(row[9]))
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

# plt.plot(x_m,y_m)
# plt.title('Map')
# plt.axis('equal')
# plt.show()


# Plot on the Map
back_map = folium.Map(location=[39.907366,116.397400],zoom_start=13) #使用OpenStreetMap默认底图,中心点坐标，天安门，腾讯地图坐标拾取器获取
#back_map = folium.Map(location=[39.907366,116.397400],zoom_start=13,tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',attr='default')  # 注意使用高德地图需要对坐标进行转换
#folium.PolyLine(coordinate, dash_array=10,color='blue',opacity=0.8).add_to(back_map) #dash_array虚线画线， weight实现画线
folium.ColorLine(positions = coordinate, colors= Spd[:-1], colormap = ['r','g','b'], weight = 5).add_to(back_map) # 实现画图，给线赋速度信息
back_map.save('map.html') #显示地图
webbrowser.open('map.html')


# Tips:
# # 更换不同的底图
# tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}', # 高德街道图
# tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', # 高德卫星图