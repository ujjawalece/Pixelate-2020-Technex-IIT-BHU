import numpy as np
import cv2 as cv
import time
import imutils
import array
import cv2.aruco as aruco
import math
import serial
n=9

cap = cv.VideoCapture(1)


ret, frame = cap.read()
roi = cv.selectROI(frame)
imcrop = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
cv.imshow('Cropped Image', imcrop)
#frame = cv.resize(frame, (600, 600), interpolation=cv.INTER_AREA)
print(roi)


    #cv.imshow('frame', frame)




cv.waitKey(1000)


a=20
dis=10
tk=10
dk=10



areay=30
areab=1500
areaw=200
arear=100
areag=100
x_1=int(roi[0])
x_2=int(roi[0]+roi[2])
y_1=int(roi[1])
y_2=int(roi[1]+roi[3])
dim=540
bluedis=30#distance of two blue
gd=25#green crop
wd=25

"""x_1=186
x_2=296+186
y_1=96
y_2=96+296"""

font = cv.FONT_HERSHEY_COMPLEX
lr=np.array([29,21,123])
ur=np.array([140,104,255])
lg=np.array([145,131,57])
ug=np.array([191,164,101])
lb=np.array([207,152,44])
ub=np.array([255,213,134])
ly=np.array([0,169,235])
uy=np.array([225,255,255])
lw=np.array([204,234,199])
uw=np.array([255,255,255])



"""lg = np.array([160, 133, 76])
ug = np.array([181, 141, 101])"""


ser = serial.Serial('COM9',9600)
def stop():
    ser.write(b's')
def left():
    ser.write(b'l')
def right():
    ser.write(b'r')
def left2():
    ser.write(b'a')
def right2():
    ser.write(b'z')
def forward():
    ser.write(b'f')
def servoup():
    ser.write(b'u')
def servodown():
    ser.write(b'd')
def stop():
    ser.write(b's')
def delay():
    ser.write(b'd')
def right1():
    ser.write(b'r')
    time.sleep(2)
def servoup1():
    ser.write(b'k')
def servodown1():
    ser.write(b'j')
def end():
    ser.write(b'e')

array = [int("99")] * (n*n)
class Graph:
    def minDistance(self, dist, queue):
        minimum = float("Inf")
        min_index = -1
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index
    def printPath(self, parent, j,z):
        if parent[j] == -1:
            #print(j)
            return
        self.printPath(parent, parent[j],z-1)
        #print(j)
        array[z]=j
    def printSolution(self, dist, parent,des):

        #print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, des, dist[des])),
        self.printPath(parent, des,n*n-1)
    def dijkstra(self, graph, src,des):
        #array = [int("99")] * (n * n)
        row = len(graph)
        col = len(graph[0])
        dist = [float("Inf")] * row
        parent = [-1] * row
        dist[src] = 0
        queue = []
        for i in range(row):
            queue.append(i)
        while queue:
            u = self.minDistance(dist, queue)
            queue.remove(u)
            for i in range(col):
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u
        self.printSolution(dist, parent,des)#
#cap=cv.VideoCapture(1)
#cap = cv.VideoCapture('http://172.17.18.192:4747/mjpegfeed?640x480')
while True:
    ret, im = cap.read()
    cropped = im[y_1:y_2,x_1:x_2]
    img = cv.resize(cropped, (dim , dim), interpolation=cv.INTER_AREA)
    cv.imwrite("hjk.png",img)
    #img = cv.resize(im, (540,540), interpolation = cv.INTER_AREA)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)





    mr1 = cv.inRange(img, ly, uy)
    contour1, hierarchy = cv.findContours(mr1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mu_1 = [None] * (len(contour1))
    mc_1 = [None] * (len(contour1))
    county = 0
    for i in range(len(contour1)):
        area = cv.contourArea(contour1[i])
        if area > areay:
            county = county + 1
        else:
            contour1[i][0][0][0] = 10000
    for i in range(len(contour1)):
        if contour1[i][0][0][0] != 10000:
            mu_1[i] = cv.moments(contour1[i])
            # mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "y")
    mu1 = [None] * (county)
    mc1 = [None] * (county)
    county = 0
    for i in range(len(contour1)):
        if contour1[i][0][0][0] != 10000:
            area = cv.contourArea(contour1[i])
            if area > areay:

                approx = cv.approxPolyDP(contour1[i], 0.015 * cv.arcLength(contour1[i], True), True)
                if len(approx) < 8:
                    mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "ys")
                    mc1[county] = mc_1[i]
                    county = county + 1
                else:
                    mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "yc")
                    mc1[county] = mc_1[i]
                    county = county + 1
    for i in range(county):
        cv.putText(img, '.', (int(mc1[i][0]), int(mc1[i][1])), font, 0.5, (0, 0, 0))
    print(county)





    mr2 = cv.inRange(img, lb, ub)
    contour2, hierarchy = cv.findContours(mr2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mu_2 = [None] * (len(contour2))
    mc_2 = [None] * (len(contour2))
    countb = 0
    for i in range(len(contour2)):
        area = cv.contourArea(contour2[i])
        if area > areab:
            countb = countb + 1
        else:
            contour2[i][0][0][0] = 10000
    for i in range(len(contour2)):
        if contour2[i][0][0][0] != 10000:
            mu_2[i] = cv.moments(contour2[i])
            mc_2[i] = (mu_2[i]['m10'] / (mu_2[i]['m00'] + 1e-5), mu_2[i]['m01'] / (mu_2[i]['m00'] + 1e-5), "b")
    mu2 = [None] * (countb)
    mc2 = [None] * (countb+2)
    countb = 0
    for i in range(len(contour2)):
        if contour2[i][0][0][0] != 10000:
            mc2[countb] = mc_2[i]
            countb = countb + 1
    mc2[countb] =(mc2[0][0]+bluedis, mc2[0][1], "b")
    print(mc2[countb])
    mc2[countb+1] =(mc2[0][0]-bluedis, mc2[0][1], "b")
    print(mc2[countb+1])
    countb=countb+2
    for i in range(countb):
        cv.putText(img, '.', (int(mc2[i][0]), int(mc2[i][1])), font, 0.5, (255, 255, 255))
    print(countb)




    mr3 = cv.inRange(img, lw, uw)
    contour3, hierarchy = cv.findContours(mr3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mu_3 = [None] * (len(contour3))
    mc_3 = [None] * (len(contour3))
    count = 0
    for i in range(len(contour3)):
        area = cv.contourArea(contour3[i])
        if area > areaw:
            count = count + 1
        else:
            contour3[i][0][0][0] = 10000
    for i in range(len(contour3)):
        if contour3[i][0][0][0] != 10000:
            mu_3[i] = cv.moments(contour3[i])
            mc_3[i] = (mu_3[i]['m10'] / (mu_3[i]['m00'] + 1e-5), mu_3[i]['m01'] / (mu_3[i]['m00'] + 1e-5), "w")
    mu3 = [None] * (count)
    mc3 = [None] * (count)
    count = 0
    for i in range(len(contour3)):
        if contour3[i][0][0][0] != 10000:
            mc3[count] = mc_3[i]
            count = count + 1
    for i in range(count):
        cv.putText(img, '.', (int(mc3[i][0]), int(mc3[i][1])), font, 0.5, (0, 0, 0))
        # cv.drawContours(img, contour3, -1, (0, 255, 0), 2)
    print(count)





    mr4 = cv.inRange(img, lr, ur)
    contour4, hierarchy = cv.findContours(mr4, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mu_4 = [None] * (len(contour4))
    mc_4 = [None] * (len(contour4))
    countr = 0
    for i in range(len(contour4)):
        area = cv.contourArea(contour4[i])
        if area > arear:
            countr = countr + 1
        else:
            contour4[i][0][0][0] = 10000
    for i in range(len(contour4)):
        if contour4[i][0][0][0] != 10000:
            mu_4[i] = cv.moments(contour4[i])
            # mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "y")
    mu4 = [None] * (countr)
    mc4 = [None] * (countr)
    countr = 0
    for i in range(len(contour4)):
        if contour4[i][0][0][0] != 10000:
            area = cv.contourArea(contour4[i])
            if area > arear:

                approx = cv.approxPolyDP(contour4[i], 0.02 * cv.arcLength(contour4[i], True), True)
                if len(approx) < 8:
                    mc_4[i] = (mu_4[i]['m10'] / (mu_4[i]['m00'] + 1e-5), mu_4[i]['m01'] / (mu_4[i]['m00'] + 1e-5), "rs")
                    mc4[countr] = mc_4[i]
                    countr= countr + 1
                else:
                    mc_4[i] = (mu_4[i]['m10'] / (mu_4[i]['m00'] + 1e-5), mu_4[i]['m01'] / (mu_4[i]['m00'] + 1e-5), "rc")
                    mc4[countr] = mc_4[i]
                    countr = countr + 1
    for i in range(countr):
        cv.putText(img, '.', (int(mc4[i][0]), int(mc4[i][1])), font, 0.5, (0, 0, 0))
    print(countr)



    for i in range(count):

        x1 = int(mc3[i][0] - gd)
        x2 = int(mc3[i][0] + gd)
        y1 = int(mc3[i][1] - gd)
        y2 = int(mc3[i][1] + gd)
        print(x1, x2, y1, y2)

        cimg = img[y1:y2, x1:x2]

        hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV)

        mr7 = cv.inRange(cimg, lg, ug)

        contour7, hierarchy = cv.findContours(mr7, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contour7) > 0:
            #print(mc3[i][2])
            l1 = list(mc3[i])
            l1[2] = "wh"
            mc3[i] = tuple(l1)
            print(mc3[i][2])

        else:
            #print(mc3[i][2])
            l1 = list(mc3[i])
            l1[2] = "ww"
            mc3[i] = tuple(l1)
            print(mc3[i][2])
    print(mc3)


    i=0
    mc = [None]*(county+countb+count+countr)
    while(i< county):
        mc[i]=mc1[i]
        i=i+1
    k=i
    while(i< (countb+county)) :
        mc[i]=mc2[i-k]
        i=i+1
    k=i
    while(i< (county+countb+count)):
        mc[i]=mc3[i-k]
        i=i+1
    k=i
    while(i< (county+countb+count+countr)) :
        mc[i]=mc4[i-k]
        i=i+1
    #print(mc)
    u=(county+countb+count+countr)


    for l in range(u):
        for j in range(u):
            if int(mc[l][1])<int(mc[j][1]):
                t=(mc[l])
                (mc[l]) = (mc[j])
                (mc[j]) = t
    j=0
    i=0
    while(i<u and j<=u):
        while (j<u and int(mc[i][1])-25<int (mc[j][1])<  int(mc[i][1])+25):
            j=j+1
        j=j-1
        k=i
        while(k<=j):
            if int(mc[i][0])>int(mc[k][0]):
                t=(mc[i])
                (mc[i]) = (mc[k])
                (mc[k]) = t

            k=k+1
        i=i+1
        j = j + 1

    """for i in range(u):
        print(int(mc[i][0]), int(mc[i][1]),mc[i][2])"""
    print (u)
    #print(mc1)
    ax = [[mc[i][0] for i in range(u)] for j in range(1)]
    ay = [[mc[i][1] for i in range(u)] for j in range(1)]
    ac = [[mc[i][2]for i in range(u)] for j in range(1)]
    horcruxes = [-1,-1,-1,-1]
    weapon = [-1,-1 ,-1,-1]
    jail = [-1, -1,-1,-1]
    weap=0
    horc=0
    ja=0
    for i in range(n*n):
        if ac[0][i]=="ww":
            weapon[weap]=i
            weap=weap+1
        if ac[0][i]=="wh":
            horcruxes[horc]=i
            horc=horc+1
        if ac[0][i] == "b":
            if 200 < ax[0][i] < 400 and 200 < ay[0][i] < 400:
                jail[len(jail) - 1] = i
            else:
                jail[ja] = i
                ja = ja + 1


    print(ac)
    print(mc)
    print(horcruxes)
    print(weapon)
    print(jail)

    g = Graph()

    count=0
    chcarr2 = [int("99")] * (n * n)  # yellow square
    for i in range(n * n):
        if ac[0][i] == "b":
            chcarr2[i] = i
    for i in range(n * n):
        if chcarr2[i] != 99:
            count = count + 1
    print(count)
    updarr2 = [int("99")] * (count)
    k = 0
    for i in range(n * n):
        if chcarr2[i] != 99:
            updarr2[k] = chcarr2[i]
            k = k + 1


    count=0
    chcarr3 = [int("99")] * (n * n)  # yellow square
    for i in range(n * n):
        if ac[0][i] == "ww" or ac[0][i] == "wh":
            chcarr3[i] = i
    for i in range(n * n):
        if chcarr3[i] != 99:
            count = count + 1
    # print(count)
    updarr3 = [int("99")] * (count)
    k = 0
    for i in range(n * n):
        if chcarr3[i] != 99:
            updarr3[k] = chcarr3[i]
            k = k + 1


    print(updarr2)
    print(updarr3)
    arr = [[0 for i in range(n*n)] for j in range(n*n)]
    for i in range(n*n):
        for j in range(n*n):
            if j==i+1 or j==i+n or j==i-1 or j==i-n:
                arr[i][j]=9
            else:
                arr[i][j]=0
            for k in range(len(updarr2)):
                if i == updarr2[k]:
                    if j == i + 1 or j == i + n or j == i - 1 or j == i - n:
                        arr[i][j] = 15
            for k in range(len(updarr3)):
                if i == updarr3[k]:
                    if j == i + 1 or j == i + n or j == i - 1 or j == i - n:
                        arr[i][j] = 15
            if i%n==n-1 and i+1<n*n:
                arr[i][i+1]=0
            if i%n==0:
                arr[i][i-1]=0

    graph =arr
    print(graph)
    cv.imshow('first_frame', img)
    cv.imwrite('img.png',img)
    cv.waitKey(100)
    break


detected=0

path = [int("99")] * (2)

for raj in range(2*(len(jail))):
    print("raj changed")

    if raj<(2*(len(jail))):
        if raj%2==0:
            if raj==0:
                while True:
                    #cap = cv.VideoCapture(1)

                    ret, im = cap.read()
                    cropped = im[y_1:y_2,x_1:x_2]
                    img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
                    parameters = aruco.DetectorParameters_create()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    #print(size(corners))
                    #cap.release()
                    try:
                        print(corners)
                        c1x = (corners[0][0][0][0] + corners[0][0][1][0]) / 2
                        c2x = (corners[0][0][2][0] + corners[0][0][3][0]) / 2
                        c1y = (corners[0][0][0][1] + corners[0][0][1][1]) / 2
                        c2y = (corners[0][0][2][1] + corners[0][0][3][1]) / 2
                        cenx = (c1x + c2x) / 2
                        ceny = (c1y + c2y) / 2
                        print(cenx,ceny)
                        distblue=[-1,-1,-1,-1]
                        for i in range(len(distblue)):
                            distblue[i] = math.sqrt(((ax[0][jail[i]] - cenx) ** 2) + ((ay[0][jail[i]] - ceny) ** 2))
                        print(distblue)
                        minpos = distblue.index(min(distblue))
                        print(minpos)
                        source=jail[minpos]
                        break
                    except:
                        stop()
                        print('s')
                        continue

            else:
                source = path[len(path) - 2]
            for i in range(n * n):
                if array[i] != 99:
                    array[i] = 99

            des= horcruxes[int(raj/2)]
            g.dijkstra(graph, source, des)
            count = 0
            for i in range(n * n):
                if array[i] != 99:
                    count = count + 1
            path = [int("99")] * (count + 1)
            path[0] = source
            for i in range(1, count + 1):
                path[i] = array[n * n - count + i - 1]
            print(path)
            motion = [int("99")] * (2)
            for i in range(len(path) - 1):

                while True:
                    #cap = cv.VideoCapture(1)
                    print(i,raj)
                    ret, im = cap.read()
                    cropped = im[y_1:y_2,x_1:x_2]
                    img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
                    parameters = aruco.DetectorParameters_create()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    #print(size(corners))
                    #cap.release()
                    try:
                        #print(corners)
                        c1x = (corners[0][0][0][0] + corners[0][0][1][0]) / 2
                        c2x = (corners[0][0][2][0] + corners[0][0][3][0]) / 2
                        c1y = (corners[0][0][0][1] + corners[0][0][1][1]) / 2
                        c2y = (corners[0][0][2][1] + corners[0][0][3][1]) / 2
                        cenx = (c1x + c2x) / 2
                        ceny = (c1y + c2y) / 2
                        #print(cenx, ceny)
                        motion[0] = path[i]
                        motion[1] = path[i + 1]
                        p0 = [ax[0][path[i]], ay[0][path[i]]]
                        p1 = [ax[0][path[i + 1]], ay[0][path[i + 1]]]
                        pc = [cenx, ceny]
                        p2 = [c1x, c1y]
                        p3 = [c2y, c2y]

                        vsd = complex(p1[0] - pc[0], p1[1] - pc[1])
                        varu = complex(p2[0] - pc[0], p2[1] - pc[1])
                        angle = np.angle(varu / vsd, deg=True)
                        print(angle)
                        distance = math.sqrt(((p1[0] - pc[0]) ** 2) + ((p1[1] - pc[1]) ** 2))
                        print(distance)

                        if (angle) < -a:
                            print('r')
                            right()
                            ang = np.degrees(angle)

                            if distance < dis:
                                break
                            #time.sleep(t)
                        if (angle) > a :
                            print('l')
                            left()
                            ang = np.degrees(angle)

                            if distance < dis:
                                break
                            #time.sleep(t)
                        if -a< (angle) < a:
                            if i == len(path) - 2:
                                if angle<-5:
                                    right2()
                                if angle>5:
                                    left2()
                                if -5<angle<5:
                                    print('f')
                                    forward()

                                    # time.sleep(t)
                                    servodown()
                                    print("servodown")
                                    break
                            else:
                                if distance>dis:
                                    print('f')
                                    forward()
                                if distance<dis:
                                    break
                        cv.imshow('0', img)
                        cv.waitKey(200)

                    except:
                        stop()
                        print('s')
                        continue
            print("raj to be changed")

        else:
            print("raj changed")
            for i in range(n * n):
                if array[i] != 99:
                    array[i]=99
            source = path[len(path) - 2]
            des = jail[int((raj-1)/2)]
            g.dijkstra(graph, source, des)
            count = 0
                # print(array)
                #print(array)
            for i in range(n * n):
                if array[i] != 99:
                    #print(array[i])
                    count = count + 1
            #print(count)
            path = [int("99")] * (count + 1)
            #print(path1)
            path[0] = source
            for i in range(1, count + 1):
                path[i] = array[n * n - count + i - 1]
            print(path)
            motion = [int("99")] * (2)
            for i in range(len(path) - 1):
                while True:
                    # cap = cv.VideoCapture(1)
                    print(i,raj)
                    ret, im = cap.read()
                    cropped = im[y_1:y_2,x_1:x_2]
                    img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
                    parameters = aruco.DetectorParameters_create()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    # print(size(corners))
                    # cap.release()
                    try:
                        # print(corners)
                        c1x = (corners[0][0][0][0] + corners[0][0][1][0]) / 2
                        c2x = (corners[0][0][2][0] + corners[0][0][3][0]) / 2
                        c1y = (corners[0][0][0][1] + corners[0][0][1][1]) / 2
                        c2y = (corners[0][0][2][1] + corners[0][0][3][1]) / 2
                        cenx = (c1x + c2x) / 2
                        ceny = (c1y + c2y) / 2
                        # print(cenx, ceny)
                        motion[0] = path[i]
                        motion[1] = path[i + 1]
                        p0 = [ax[0][path[i]], ay[0][path[i]]]
                        p1 = [ax[0][path[i + 1]], ay[0][path[i + 1]]]
                        pc = [cenx, ceny]
                        p2 = [c1x, c1y]
                        p3 = [c2y, c2y]

                        vsd = complex(p1[0] - pc[0], p1[1] - pc[1])
                        varu = complex(p2[0] - pc[0], p2[1] - pc[1])
                        angle = np.angle(varu / vsd, deg=True)
                        print(angle)
                        distance = math.sqrt(((p1[0] - pc[0]) ** 2) + ((p1[1] - pc[1]) ** 2))
                        print(distance)

                        if (angle) < -a:
                            print('r')
                            right()
                            ang = np.degrees(angle)

                            if distance < dis:
                                break
                            # time.sleep(t)
                        if (angle) > a:
                            print('l')
                            left()
                            ang = np.degrees(angle)

                            if distance < dis:
                                break
                            # time.sleep(t)
                        if -a < (angle) < a:
                            if i == len(path) - 2:
                                print('f')
                                #forward()

                                # time.sleep(t)
                                servoup()
                                print("servoup")
                                break
                            else:
                                if distance > dis:
                                    print('f')
                                    forward()

                                    # time.sleep(t)
                                if distance < dis:
                                    break
                        cv.imshow('0', img)
                        cv.waitKey(200)

                    except:
                        stop()
                        print('s')
                        continue
detected1=['rs','rs','rs','rs']
for check in range(len(horcruxes)):
    print(check)
    d = horcruxes[check]
    x = ax[0][d]
    y = ay[0][d]
    x1 = int(ax[0][d] - wd)
    x2 = int(ax[0][d] + wd)
    y1 = int(ay[0][d] - wd)
    y2 = int(ay[0][d] + wd)
    print(x1, x2, y1, y2)

    ret, im = cap.read()
    cropped = im[y_1:y_2, x_1:x_2]

    img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
    cimg = img[y1:y2, x1:x2]

    cv.imshow("cropped", cimg)
    cv.imwrite("crp.png", cimg)
    cv.waitKey(1000)

    hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV)

    mr5 = cv.inRange(cimg, ly, uy)
    cv.imshow('d', mr5)
    cv.waitKey(1000)

    contour5, hierarchy = cv.findContours(mr5, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contour5))

    mu_5 = [None] * (len(contour5))
    mc_5 = [None] * (len(contour5))
    county = 0
    for i in range(len(contour5)):
        area = cv.contourArea(contour5[i])
        if area > areay:
            county = county + 1
        else:
            contour5[i][0][0][0] = 10000
    for i in range(len(contour5)):
        if contour5[i][0][0][0] != 10000:
            mu_5[i] = cv.moments(contour5[i])
            # mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "y")
    mu5 = [None] * (county)
    mc5 = [None] * (county)
    county = 0
    for i in range(len(contour5)):
        if contour5[i][0][0][0] != 10000:
            area = cv.contourArea(contour5[i])
            if area > areay:

                approx = cv.approxPolyDP(contour5[i], 0.02 * cv.arcLength(contour5[i], True), True)
                if len(approx) < 8:
                    mc_5[i] = (mu_5[i]['m10'] / (mu_5[i]['m00'] + 1e-5), mu_5[i]['m01'] / (mu_5[i]['m00'] + 1e-5), "ys")
                    mc5[county] = mc_5[i]
                    detected1[check] = mc5[county][2]
                    county = county + 1

                    print(county)
                else:
                    mc_5[i] = (mu_5[i]['m10'] / (mu_5[i]['m00'] + 1e-5), mu_5[i]['m01'] / (mu_5[i]['m00'] + 1e-5), "yc")
                    mc5[county] = mc_5[i]
                    detected1[check] = mc5[county][2]
                    county = county + 1
                    print(county)

    for i in range(county):
        cv.putText(cimg, '.', (int(mc5[i][0]), int(mc5[i][1])), font, 0.5, (0, 0, 0))

    mr6 = cv.inRange(cimg, lr, ur)
    cv.imshow('d', mr6)
    cv.waitKey(1000)

    contour6, hierarchy = cv.findContours(mr6, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contour6))

    mu_6 = [None] * (len(contour6))
    mc_6 = [None] * (len(contour6))
    countr = 0
    for i in range(len(contour6)):
        area = cv.contourArea(contour6[i])
        if area > arear:
            countr = countr + 1
        else:
            contour6[i][0][0][0] = 10000
    for i in range(len(contour6)):
        if contour6[i][0][0][0] != 10000:
            mu_6[i] = cv.moments(contour6[i])
            # mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "y")
    mu6 = [None] * (countr)
    mc6 = [None] * (countr)
    countr = 0
    for i in range(len(contour6)):
        if contour6[i][0][0][0] != 10000:
            area = cv.contourArea(contour6[i])
            if area > arear:

                approx = cv.approxPolyDP(contour6[i], 0.02 * cv.arcLength(contour6[i], True), True)
                if len(approx) < 8:
                    mc_6[i] = (mu_6[i]['m10'] / (mu_6[i]['m00'] + 1e-5), mu_6[i]['m01'] / (mu_6[i]['m00'] + 1e-5), "rs")
                    mc6[countr] = mc_6[i]
                    detected1[check] = mc6[countr][2]
                    countr = countr + 1

                    print(countr)
                else:
                    mc_6[i] = (mu_6[i]['m10'] / (mu_6[i]['m00'] + 1e-5), mu_6[i]['m01'] / (mu_6[i]['m00'] + 1e-5), "rc")
                    mc6[countr] = mc_6[i]
                    detected1[check] = mc6[countr][2]
                    countr = countr + 1
                    print(countr)

    for i in range(countr):
        cv.putText(cimg, '.', (int(mc6[i][0]), int(mc6[i][1])), font, 0.5, (0, 0, 0))

    print(detected1[check])
    cv.imshow("cropped", cimg)
    cv.imwrite("green.png",cimg)
    cv.waitKey(100)
print("khatam horcruxes")
print(detected1)




for raj in range(2*len(jail)):
    if raj%2==0:
        for i in range(n * n):
            if array[i] != 99:
                array[i]=99
        source = path[len(path) - 2]
        #source=14
        des = weapon[int(raj/2)]
        g.dijkstra(graph, source, des)
        count = 0
        for i in range(n * n):
            if array[i] != 99:
                count = count + 1
        path = [int("99")] * (count + 1)
        path[0] = source
        for i in range(1, count + 1):
            path[i] = array[n * n - count + i - 1]
        print(path)
        motion = [int("99")] * (2)
        for i in range(len(path) - 1):

            while True:
                # cap = cv.VideoCapture(1)
                print(i)
                ret, im = cap.read()
                cropped = im[y_1:y_2, x_1:x_2]
                img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
                aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
                parameters = aruco.DetectorParameters_create()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                # print(size(corners))
                # cap.release()
                try:
                    # print(corners)
                    c1x = (corners[0][0][0][0] + corners[0][0][1][0]) / 2
                    c2x = (corners[0][0][2][0] + corners[0][0][3][0]) / 2
                    c1y = (corners[0][0][0][1] + corners[0][0][1][1]) / 2
                    c2y = (corners[0][0][2][1] + corners[0][0][3][1]) / 2
                    cenx = (c1x + c2x) / 2
                    ceny = (c1y + c2y) / 2
                    # print(cenx, ceny)
                    motion[0] = path[i]
                    motion[1] = path[i + 1]
                    p0 = [ax[0][path[i]], ay[0][path[i]]]
                    p1 = [ax[0][path[i + 1]], ay[0][path[i + 1]]]
                    pc = [cenx, ceny]
                    p2 = [c1x, c1y]
                    p3 = [c2y, c2y]

                    vsd = complex(p1[0] - pc[0], p1[1] - pc[1])
                    varu = complex(p2[0] - pc[0], p2[1] - pc[1])
                    angle = np.angle(varu / vsd, deg=True)
                    print(angle)
                    distance = math.sqrt(((p1[0] - pc[0]) ** 2) + ((p1[1] - pc[1]) ** 2))
                    print(distance)

                    if (angle) < -a:
                        print('r')
                        right()
                        ang = np.degrees(angle)

                        if distance < dis:
                            break
                        # time.sleep(t)
                    if (angle) > a:
                        print('l')
                        left()
                        ang = np.degrees(angle)

                        if distance < dis:
                            break
                        # time.sleep(t)
                    if -a < (angle) < a:
                        if i == len(path) - 2:
                            if angle < -5:
                                right2()
                            if angle > 5:
                                left2()
                            if -5 < angle < 5:
                                print('f')
                                forward()

                                # time.sleep(t)
                                servodown1()
                                print("servodown")
                                break
                        else:
                            if distance > dis:
                                print('f')
                                forward()
                            if distance < dis:
                                break
                    cv.imshow('0', img)
                    cv.waitKey(200)
                except:
                    stop()
                    print('s')
                    continue
        detected="non"
        while True:

            right1()
            print("r")
            time.sleep(1)
            right1()
            print("r")
            stop()

            # time.sleep(t)

            # k=22
            d = weapon[int(raj/2)]
            print("ghum gaya")

            x=ax[0][d]
            y=ay[0][d]

            x1 = int(ax[0][d] - wd)
            x2 = int(ax[0][d] + wd)
            y1 = int(ay[0][d] - wd)
            y2 = int(ay[0][d] + wd)
            print(x1,x2,y1,y2)

            ret, im = cap.read()
            cropped = im[y_1:y_2,x_1:x_2]

            img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
            cimg = img[y1:y2,x1:x2]

            cv.imshow("cropped",cimg)
            cv.imwrite("crp.png",cimg)
            cv.waitKey(1000)

            hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV)

            mr5 = cv.inRange(cimg, ly, uy)
            cv.imshow('d', mr5)
            cv.waitKey(1000)

            contour5, hierarchy = cv.findContours(mr5, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            print(len(contour5))

            mu_5 = [None] * (len(contour5))
            mc_5 = [None] * (len(contour5))
            county = 0
            for i in range(len(contour5)):
                area = cv.contourArea(contour5[i])
                if area > areay:
                    county= county + 1
                else:
                    contour5[i][0][0][0] = 10000
            for i in range(len(contour5)):
                if contour5[i][0][0][0] != 10000:
                    mu_5[i] = cv.moments(contour5[i])
                    # mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "y")
            mu5 = [None] * (county)
            mc5 = [None] * (county)
            county = 0
            for i in range(len(contour5)):
                if contour5[i][0][0][0] != 10000:
                    area = cv.contourArea(contour5[i])
                    if area > areay:

                        approx = cv.approxPolyDP(contour5[i], 0.02 * cv.arcLength(contour5[i], True), True)
                        if len(approx) < 8:
                            mc_5[i] = (mu_5[i]['m10'] / (mu_5[i]['m00'] + 1e-5), mu_5[i]['m01'] / (mu_5[i]['m00'] + 1e-5), "ys")
                            mc5[county] = mc_5[i]
                            detected = mc5[county][2]
                            county = county + 1

                            print(county)
                        else:
                            mc_5[i] = (mu_5[i]['m10'] / (mu_5[i]['m00'] + 1e-5), mu_5[i]['m01'] / (mu_5[i]['m00'] + 1e-5), "yc")
                            mc5[county] = mc_5[i]
                            detected = mc5[county][2]
                            county = county + 1
                            print(county)

            for i in range(county):
                cv.putText(cimg, '.', (int(mc5[i][0]), int(mc5[i][1])), font, 0.5, (0, 0, 0))





            mr6 = cv.inRange(cimg, lr, ur)
            cv.imshow('d', mr6)
            cv.waitKey(1000)

            contour6, hierarchy = cv.findContours(mr6, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            print(len(contour6))

            mu_6 = [None] * (len(contour6))
            mc_6 = [None] * (len(contour6))
            countr = 0
            for i in range(len(contour6)):
                area = cv.contourArea(contour6[i])
                if area > arear:
                    countr = countr + 1
                else:
                    contour6[i][0][0][0] = 10000
            for i in range(len(contour6)):
                if contour6[i][0][0][0] != 10000:
                    mu_6[i] = cv.moments(contour6[i])
                    # mc_1[i] = (mu_1[i]['m10'] / (mu_1[i]['m00'] + 1e-5), mu_1[i]['m01'] / (mu_1[i]['m00'] + 1e-5), "y")
            mu6 = [None] * (countr)
            mc6 = [None] * (countr)
            countr = 0
            for i in range(len(contour6)):
                if contour6[i][0][0][0] != 10000:
                    area = cv.contourArea(contour6[i])
                    if area > arear:

                        approx = cv.approxPolyDP(contour6[i], 0.02 * cv.arcLength(contour6[i], True), True)
                        if len(approx) < 8:
                            mc_6[i] = (mu_6[i]['m10'] / (mu_6[i]['m00'] + 1e-5), mu_6[i]['m01'] / (mu_6[i]['m00'] + 1e-5), "rs")
                            mc6[countr] = mc_6[i]
                            detected = mc6[countr][2]
                            countr = countr + 1

                            print(countr)
                        else:
                            mc_6[i] = (
                            mu_6[i]['m10'] / (mu_6[i]['m00'] + 1e-5), mu_6[i]['m01'] / (mu_6[i]['m00'] + 1e-5), "rc")
                            mc6[countr] = mc_6[i]
                            detected = mc6[countr][2]
                            countr = countr + 1
                            print(countr)

            for i in range(countr):
                cv.putText(cimg, '.', (int(mc6[i][0]), int(mc6[i][1])), font, 0.5, (0, 0, 0))
            if detected=="yc" or detected=="ys" or detected=="rc" or detected=="rs":
                break



        print(detected)
        cv.imshow("cropped", cimg)
        cv.waitKey(100)
        print("khatam")
        for i in range(len(detected1)):
            if detected1[i]==detected:
                number=i
                break



    else:

        print("raj last mein")

        for i in range(n * n):
            if array[i] != 99:
                array[i]=99
        count = 0
        chcarr1 = [int("99")] * (n * n)  # yellow square
        for i in range(n * n):
            if ac[0][i] == detected:
                chcarr1[i] = i
        for i in range(n * n):
            if chcarr1[i] != 99:
                count = count + 1
        # print(count)
        updarr1 = [int("99")] * (count)
        k = 0
        for i in range(n * n):
            if chcarr1[i] != 99:
                updarr1[k] = chcarr1[i]
                k = k + 1
        print(updarr1)
        arr2 = [[0 for i in range(n * n)] for j in range(n * n)]
        for i in range(n * n):
            for j in range(n * n):
                if j == i + 1 or j == i + n or j == i - 1 or j == i - n:
                    arr2[i][j] = 9
                else:
                    arr2[i][j] = 0
                for k in range(len(updarr1)):
                    if i == updarr1[k]:
                        if j == i + 1 or j == i + n or j == i - 1 or j == i - n:
                            arr2[i][j] = 1
                for k in range(len(updarr2)):
                    if i == updarr2[k]:
                        if j == i + 1 or j == i + n or j == i - 1 or j == i - n:
                            arr2[i][j] = 15
                for k in range(len(updarr3)):
                    if i == updarr3[k]:
                        if j == i + 1 or j == i + n or j == i - 1 or j == i - n:
                            arr2[i][j] = 15
                if i % n == n - 1 and i + 1 < n * n:
                    arr2[i][i + 1] = 0
                if i % n == 0:
                    arr2[i][i - 1] = 0

        graph = arr2

        source = path[len(path) - 2]
        des = horcruxes[number] #ys
        g.dijkstra(graph, source, des)
        count =0
        for i in range(n * n):
            if array[i] != 99:
                count = count + 1
        path = [int("99")] * (count + 1)
        path[0] = source
        for i in range(1, count + 1):
            path[i] = array[n * n - count + i - 1]
        print(path)
        motion = [int("99")] * (2)
        for i in range(len(path) - 1):
            while True:
                # cap = cv.VideoCapture(1)
                print(i)
                ret, im = cap.read()
                cropped = im[y_1:y_2, x_1:x_2]
                img = cv.resize(cropped, (dim, dim), interpolation=cv.INTER_AREA)
                aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
                parameters = aruco.DetectorParameters_create()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                # print(size(corners))
                # cap.release()
                try:
                    # print(corners)
                    c1x = (corners[0][0][0][0] + corners[0][0][1][0]) / 2
                    c2x = (corners[0][0][2][0] + corners[0][0][3][0]) / 2
                    c1y = (corners[0][0][0][1] + corners[0][0][1][1]) / 2
                    c2y = (corners[0][0][2][1] + corners[0][0][3][1]) / 2
                    cenx = (c1x + c2x) / 2
                    ceny = (c1y + c2y) / 2
                    # print(cenx, ceny)
                    motion[0] = path[i]
                    motion[1] = path[i + 1]
                    p0 = [ax[0][path[i]], ay[0][path[i]]]
                    p1 = [ax[0][path[i + 1]], ay[0][path[i + 1]]]
                    pc = [cenx, ceny]
                    p2 = [c1x, c1y]
                    p3 = [c2y, c2y]

                    vsd = complex(p1[0] - pc[0], p1[1] - pc[1])
                    varu = complex(p2[0] - pc[0], p2[1] - pc[1])
                    angle = np.angle(varu / vsd, deg=True)
                    print(angle)
                    distance = math.sqrt(((p1[0] - pc[0]) ** 2) + ((p1[1] - pc[1]) ** 2))
                    print(distance)

                    if (angle) < -a:
                        print('r')
                        right()
                        ang = np.degrees(angle)

                        if distance < dis:
                            break
                        # time.sleep(t)
                    if (angle) > a:
                        print('l')
                        left()
                        ang = np.degrees(angle)

                        if distance < dis:
                            break
                        # time.sleep(t)
                    if -a < (angle) < a:
                        if i == len(path) - 2:
                            print('f')
                            #forward()

                            # time.sleep(t)
                            servoup1()
                            print("servoup")

                            break
                        else:
                            if distance > dis:
                                print('f')
                                forward()

                                # time.sleep(t)
                            if distance < dis:
                                break
                    cv.imshow('0', img)
                    cv.waitKey(200)

                except:
                    stop()
                    print('s')
                    continue
stop()
end()