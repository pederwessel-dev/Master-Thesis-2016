 # -*- coding: utf-8 -*-

import numpy as np
import datetime
print datetime.datetime.today()

def dataset(filename, criterion,startdate="2009-02-23",enddate="2010-02-22"):
	#setting __doc__
	"""
	Criterion is the number of elements.
	The dates must be set on the form YYYY-MM-DD as a string.
	"""
#	import pprint
#	pp = pprint.PrettyPrinter(indent=4)
	infile = open(filename, 'r')		#open file for reading
	unstrippedlines = infile.readlines()
	lines = []
	for line in unstrippedlines:
		lines.append(line.strip('\n'))
	
	#fetching the number of lines in the header
	headernumb = 0
	indexcount = 0
	while headernumb < len(lines):
		l = lines[indexcount]
		if len(l.split(","))>criterion:
			break
		headernumb +=1
		indexcount +=1
	firstline = lines[headernumb]
	lastline = lines[-1]

	#deleting the header
	headerlines = lines[:headernumb]	
	UTM_coor,heights = [],[]
	for l in headerlines:
		if l.find("Elevation")>= 0:
			h = l.split('-')
			del h[0]
			heights.append(h)
		elif l.find("Latitude") >= 0:
			g = l.split('-')
			del g[0]
			UTM_coor.append(g)
		elif l.find("Longitude") >= 0:
			f = l.split('-')
			del f[0]
			UTM_coor.append(f)
	for i in range(headernumb):					
		del lines[0]
	
	#setting number of total amount of measurements
	comments = []
	comments.append("The total amount line numbers in the header is %d." %headernumb)
	comments.append("The total number of measurements before filtering is %d." % len(lines))

	#For crosschecking the first and the last measurement in the dataseries
	if firstline != lines[0] or lastline != lines[-1]: 
		comments.append("The data has been read wrongly.")
	else:
		comments.append("The data has been read correctly.")
	
	#neglecting the time series without a wind speed value
	newlines = []
	counter = 0
	for i in range(len(lines)): 		 
		if lines[i].find('NaN') < 0:	
			newlines.append(lines[i])
			counter +=1	
	comments.append("The number of measurements after deleting"+\
	" wind speed measurements with NaN is %d." % counter)
	
	#categorizing data into columns 
	import numpy as np
	rows = []
	for line in newlines:
		row = line.split(',')
		rows.append(tuple(row)) 

	#Fetching the field names in the header
	col_names = headerlines[headernumb-len(rows[0])-1:-1]

	#Storing the data in a structured array	
	dtype_list = ['i4', 'a10', 'a8']
	for i in range(3, len(rows[0])): 
		dtype_list.append('<f8')
	dt = np.dtype(zip(col_names,dtype_list))
	cols = np.array(rows,dt) 
	
	#Selecting the dataset period. The periods must be on the form YYYY-MM-DD on a string.

	indexlist = []
	for i in range(len(cols['Date Field'])):
	    date = stringtodate(cols['Date Field'][i])
	    if stringtodate(startdate) <= date <= stringtodate(enddate):
	        indexlist.append(i)
	
	start, stop = indexlist[0], indexlist[-1]
	del indexlist
    
	#Writing the field names in this file
	comments.append("The fields in this file are:")
	for i in range(len(dt.names)):
		comments.append([i+1, dt.names[i]])
	
	#Counting number of days of measurements	
	for name in dt.names:
		if name == 'Date Field':
			datefield = cols[name]
	counter = 0
	daycounter = 0
	while counter < len(datefield):
		day1 = int(datefield[counter-1][-2] + datefield[counter-1][-1])
		day = int(datefield[counter][-2] + datefield[counter][-1])
		if day1 != day:
			daycounter +=1
		counter+=1
	comments.append("The measurement period has extended for %d days." %daycounter)
	infile.close()									
	#defining the output
	return cols[start:stop],comments, heights, UTM_coor
	
def RH_ar(filename,startdate = "2009-02-22", enddate = "2010-02-23"):
	"""
	This function returns moisture data into structured arrays.
	"""
	import datetime
	infile = open(filename, 'r')
	linelist = infile.readlines()
	lines = []
	for line in linelist:
		lines.append(line.split(';'))
    #Fetching the field names
	fieldnames = lines[9][0:3]
	
	#Deleting the header
	del lines[0:10]
	dates , times, rhs = [],[],[]
	for row in lines:
		dates.append(row[0])
		times.append(row[1])
		rhs.append(row[2])	
			
	#Categorizing the header
	dt = np.dtype([(fieldnames[0], 'S10'), (fieldnames[1], 'S8'), (fieldnames[2], float)])
	dataarray = np.asarray(zip(dates,times,rhs), dt)
	
	#Filtering the data by date
	indexlist = []
	for i in range(len(dataarray[fieldnames[0]])):
	    date = stringtodate(dataarray[fieldnames[0]][i])
	    if stringtodate(startdate) <= date <= stringtodate(enddate):
	        indexlist.append(i)
	start, stop = indexlist[0], indexlist[-1]
	del indexlist
	return dataarray[start:stop]
	
def stringtodate(stringdate):
    #Setting __doc__
    """
    This function return a string type on the form YYYY-MM-DD to a date type.
    """
    import re, datetime
    match = re.search(r'\d{4}-\d{2}-\d{2}', stringdate)
    return datetime.datetime.strptime(match.group(),'%Y-%m-%d').date()

def stringtotime(stringtime):
    #Setting __doc__
    """
    This function return a string type on the form HH:MM:SS to a time type.
    """
    import re, datetime
    match = re.search(r'\d{2}:\d{2}:\d{2}', stringtime)
    return datetime.datetime.strptime(match.group(),'%H:%M:%S').time()


def tablewriter(list_, nameoffile, path, dirs, criterion, comments = "no"):
    """
    This function writes data on a csv-file for an overview over the dateset.
    """ 
    with open(nameoffile,"wb") as outfile:
        csv_writer = csv.writer(outfile,delimiter = ' ')
        for filename in dirs:
			csv_writer.writerow(["The following file is running " + filename])
			csv_writer.writerow(["			"])
			cols = list_[0]
			comments = list_[1]
			for name in cols.dtype.names:
				if type(cols[name][0]) == np.float64:
					csv_writer.writerow([np.average(cols[name])])			
			if comments == "yes":
				for name in cols.dtype.names:
					if type(cols[name][0]) == np.float64:
						csv_writer.writerow([name])
						csv_writer.writerow(['	','		'])
				for i in range(len(comments)):
					csv_writer.writerow([comments[i]])
					csv_writer.writerow(['	', '	'])
	return "The data has now been written."

def heightwriter(filename):
    """
    This function writes the heigts and UTM-coordinates of the height above ground at the
    geographical position of meteorological masts as a csv-file. 
    """  
    with open(filename,"wb") as outfile:
        csv_writer = csv.writer(outfile,delimiter = ' ')
        for filename in dirs:
        	csv_writer.writerow(dataset(path + filename, criterion)[2])
        for filename in dirs:	
        	csv_writer.writerow(dataset(path + filename, criterion)[3])
        for filename in dirs:	
        	csv_writer.writerow(["The following file is running " + filename])
        	csv_writer.writerow(["			"])
			
def heights(filename):
    """
    This function reads height above ground values of the geographical position of the
    meteorological masts.
    """
    infile = open(filename, 'r')		#open file for reading
    infile.readline()					#put every line as list elements
    station_no, elevation, UTM_la, UTM_lo = [],[],[],[]
    for line in infile:
        station_no.append(eval(line.strip('\n').split(' ')[0]))
        elevation.append(eval(line.strip('\n').split(' ')[1]))
        UTM_la.append(eval(line.strip('\n').split(' ')[2]))
        UTM_lo.append(eval(line.strip('\n').split(' ')[3]))
	heights = zip(station_no, elevation, UTM_la, UTM_lo)
	dt = np.dtype([('Station Number', 'i'),('Elevation', 'float'),('Latitude','float'),\
	 ('Longitude', 'float')])
	return np.array(heights, dtype = dt)

def combinatoric(filename):
    """
    This function finds the combination between two meteorological masts.
    """
    latitude = heights(filename)['Latitude']
    longitude = heights(filename)['Longitude']
    #Making the UTM coordinates into a vector.
    COOR = np.array(zip(latitude ,longitude))
    Station_no = heights(filename)['Station Number']
    Elevation = heights(filename)['Elevation']
    counter = 0
    comblist1, comblist2, distancelist, El_diff, columnlist= [],[],[],[],[]
    for i in range(len(COOR)):
    	for j in range(len(COOR)):
			if i!=j:
				B = COOR[i]-COOR[j]
				El_diff.append(eval("%.2f" % (Elevation[i]-Elevation[j])))
				comblist1.append(Station_no[i])
				comblist2.append(Station_no[j])
				distancelist.append(eval("%.2f" % (np.linalg.norm(B)/1000.)))
				counter +=1
	print "There are %d combinations between 2 selections and %d possibilities." \
		% (counter, len(COOR))
	dt = [('Met Mast 1', 'int'),('Met Mast 2', 'int'),('Distance','float'),\
	('Elevation difference (1-2)','float')]
	array_ = np.asarray(zip(comblist1, comblist2,distancelist,El_diff),dtype = dt)
	sort_arr = np.sort(array_,order = ('Distance','Elevation difference (1-2)'))
	#Returning two lists for cross checking
	return sort_arr[0::2] , np.delete(sort_arr, np.s_[0::2],0) 

def csv_writer(list_,csv_name):
    """
    This function writes list elements to a csv-file.
    """
    with open(csv_name,"wb") as outfile:
        csv_writer = csv.writer(outfile)
        for element in list_:
            csv_writer.writerow(element)

def statistics_table():
    """
    This function writes the statistics of the selected data.
    """ 
    import os,csv,shutil
    path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    dirs = os.listdir(path)
    criterion = 9
    foldername = "Statistics-tables"
    if os.path.isdir(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    samefiles = []
    for i in range(len(dirs)-1):
        if dirs[i+1][0:4] == dirs[i][0:4]:    
            samefiles.append([dirs[i][0:4],[dirs[i],dirs[i+1]]])
    for elements in samefiles:
		station_no, files = elements 
		with open(foldername + "/" + "%s-statistics.csv"%station_no,"wb") as outfile:
		    csv_writer = csv.writer(outfile,delimiter = ' ')
		    for filename in files:
		        cols = dataset(path + filename,criterion)[0]
		        csv_writer.writerow(["The current file is being read:", filename])
		        csv_writer.writerow(["              "])
		        for name in cols.dtype.names:
		            if type(cols[name][0]) == np.float64:
				        csv_writer.writerow(["Column name: ", name])
				        csv_writer.writerow(["Number of elements: ", len(cols[name])])
				        csv_writer.writerow(["Average value: ", np.average(cols[name])])
				        csv_writer.writerow(["Variance: ", np.var(cols[name])])
				        csv_writer.writerow(["Standard deviation: ", np.std(cols[name])])
				        csv_writer.writerow(["Median value: ", np.median(cols[name])])
				        csv_writer.writerow(["Minimum value: ", np.min(cols[name])])
				        csv_writer.writerow(["Maximum value: ", np.max(cols[name])])
				        csv_writer.writerow(["Range: ", np.max(cols[name]) - np.min(cols[name])])
				        csv_writer.writerow(["              "])
    return ""

def statistics_timeplots():
    """
    This function writes the statistics of the selected data.
    """ 
    import os,csv,shutil, datetime
    import matplotlib.pyplot as plt
    path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    dirs = os.listdir(path)
    criterion = 9
    foldername = "Statistics-plots"
    if os.path.isdir(foldername):
        shutil.rmtree(foldername)
    samefiles = []
    for i in range(len(dirs)-1):
        if dirs[i+1][0:4]==dirs[i][0:4]:
            samefiles.append([dirs[i][0:4],[dirs[i],dirs[i+1]]])
	os.mkdir(foldername)	
	for elements in samefiles:
	    station_no, files = elements
	    subfolder = foldername + "/" + station_no 
	    os.mkdir(subfolder)
	    for filename in files:
	        cols = dataset(path + filename,criterion)[0]
		    #Finding the x-labels 
	        loclist, dates_list, indexlist = [],[],[]
	        indexlist.append(0)
	        dates_list.append(stringtodate(cols["Date Field"][0]))
	        loclist.append(0)
	        for i in range(1,len(cols['Date Field'])):
	            if cols["Date Field"][i][5:7] != cols["Date Field"][i-1][5:7]:
	                dates_list.append(stringtodate(cols["Date Field"][i]))
	                loclist.append(i)
	            #Finding the indexes for the days to calculate the daily averages
	            elif cols["Date Field"][i][8:10] != cols["Date Field"][i-1][8:10]:
	                indexlist.append(i)
	        dates_list.append(stringtodate(cols["Date Field"][-1]))
	        loclist.append(-1)
	        for name in cols.dtype.names:
	            if name != "Time Field" and name != "Station no" and name!= "Date Field":
	                titlefont = {'fontname': 'Arial', 'size': '15', 'weight':'normal'}
	                axisfont = {'fontname': 'Arial', 'size': '14'}
	                fig, ax = plt.subplots(1)
	                ax.plot(cols[name], 'k', linewidth = 1.0)
	                ax.set_title("Time plot of the field %s" % name, **titlefont)
	                plt.xlim(loclist[0], loclist[-1])
	                plt.xticks(loclist, dates_list, rotation=70)
	                ax.set_xlabel("Time")
	                if name[0:3] == "Ano":
	                    ax.set_ylabel("Wind speed (m/s)", **axisfont)
	                elif name[0:3] == "Dir":
	                    ax.set_ylabel(r"Wind direction ($^\circ$)", **axisfont)
	                elif name[0:4] == "Temp":
	                    ax.set_ylabel(r"Temperature ($^\circ C$)", **axisfont)
	                plt.ylim(0,np.max(cols[name])+2)
	                fig.savefig(subfolder + "/" + "%s-%s-timeplot.png" % (station_no, name))
	                plt.clf()
    return ""
    
def dailyav(indexlist,cols):
    """
    This function returns an array with the daily averages.
    """
    av_ar = np.zeros((len(cols),float))
    daily_av = []
    for i in range(1,len(indexlist)):
        daily_av.append(np.average(cols[name][i-1:i]))
    return len(indexlist),len(daily_av)
                             
def pixel(file,x,y):
    """
    This function returns the closest pixel-values of a coordinates in a os.geo-instance.
    """ 
    px = file.GetGeoTransform()[0]
    py = file.GetGeoTransform()[3]
    rx = file.GetGeoTransform()[1]
    ry = file.GetGeoTransform()[5]
    rasterx = int((x - px) / rx)
    rastery = int((y-py) / ry)
    return rasterx,rastery

def linestringsreader(filename):
    """
    This function returns a list of line coordinates by reading the file containing the coordinates.
    """
    infile = open(filename)
    infile.readline()
    LSlist = []
    x1list,y1list,x2list,y2list = [],[],[],[]
    for line in infile:
    	row = line.strip('\n').split(',')
    	LSlist.append([eval(row[0]),eval(row[1]),eval(row[2]),eval(row[3])])
	return LSlist

def elevraster(linestring, gridresolution, file,data):
    """
    This function returns the interpolated values along a line on a raster. 
    """
    from shapely.geometry import LineString
    line = LineString(linestring)
    length = line.length
    diagonal = np.sqrt(gridresolution[0]**2+gridresolution[1]**2)
    resolution = int(round(float(length)/diagonal))
    x, y, z,color,dist = [],[],[],[],[]
    arlist = np.linspace(0,length,resolution)
    for distance in arlist:
        point = line.interpolate(distance)
        xp,yp = float(point.x), float(point.y)
        rasterx,rastery = pixel(file,xp,yp)
        x.append(xp)
        y.append(yp)
        z.append(data[rasterx,rastery])
        color.append(data[rasterx,rastery])
        dist.append(distance/1000.)
	return x,y,z,color,dist,round(diagonal,2)

def transformer(filename):
    """
    This function reads the raster file.
    """
    import osgeo.gdal as gdal #from gdalconst import *
    dataset = gdal.Open(filename, GA_ReadOnly)
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    bandtype = gdal.GetDataTypeName(band.DataType)
    return gt[1], gt[5], dataset, band.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize).astype(np.float) 

def plotter(path, funcxyz, angle, twolist,counter):
    """
    This function plots the elevtion profile between two meteorological masts.
    """ 
    import matplotlib.pyplot as plt
    titlefont = {'fontname': 'Arial', 'size': '15', 'weight':'normal'}
    axisfont = {'fontname': 'Arial', 'size': '14'}
    x,y,z,color,dist,diagonal = funcxyz
    fig = plt.figure(counter)
    plt.plot(dist,z,linewidth = 2, color = 'black')
    plt.title('Elevation profile for the combination %d-%d (dir:%d' \
    %(twolist[0],twolist[1],angle)+ r'$^\circ$)', **titlefont) 
    plt.xlabel('Distance (km)',**axisfont)
    plt.ylim(250,600)
    plt.ylabel('Elevation (m)',**axisfont)
    plt.xlim(dist[0],dist[-1])
    fig.savefig(path + '%d-Combination_%d_%d_res%d.png'%(counter,twolist[0],twolist[1],diagonal))
    return ""

def dir(LSelement):
    """
    This function returns the orientation of a vector.
    """ 
    ar1, ar2 = LSelement
    A = np.asarray(ar1,float) - np.asarray(ar2,float)
    return int(round(180 + np.arctan2(A[0],A[1])*(180/np.pi)))

def dirarray(angle):
    """
    This function finds the unit vector from a northward oriented vector.
    """
    x,y = np.array((0,1))
    angle = (angle + 90)*(np.pi/180)
    xp = x*np.cos(angle) - y*np.sin(angle)
    yp = y*np.sin(angle) + y*np.cos(angle)
    return np.array((xp,yp))/np.linalg.norm(np.array(xp,yp))

def standard_dev(xlist,ylist):
    """
    This evaluates the values of a regression analysis of two arrays.
    """
    xav = np.average(xlist)
    yav = np.average(ylist)
    N = len(ylist)
    Sx = np.sqrt(sum((xlist - xav)**2)/float(N))
    Sy = np.sqrt(sum((ylist - yav)**2)/float(N))
    b = sum((xlist - xav)*(ylist - yav))/sum((xlist-xav)**2)
    r = b*(Sx/Sy)
    sYX = Sy*np.sqrt(((N-1)/(N-2))*(1-r**2))
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(xlist,ylist)
    return abs(round(np.arctan(b)*180/np.pi,2)),round(r,2), round(sYX,2)

def calculate_initial_compass_bearing(pointA, pointB):
    """
    This function finds the orientation of a line between two points.
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
#	import math 

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)\
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180deg to + 180deg which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def isfloat(value):
    """
    This function tests a value if it is a float.
    """ 
    try:
        float(value)
        return True

    except ValueError:
        return False

def speedheight(namelist):
    """
    This function finds a speed columns from a list of strings.
    """
    height, strlist, speedcolname = [], [], []
    for name in namelist:
        if name.find('Ano')	>= 0 and name.find('Avg') >= 0:
        	speedcolname.append(name)
        	for element in name.split(' '):
        		strlist.append(element)
        		for nextelement in element.split('_'):
        		    strlist.append(nextelement)	
					
	for element in strlist:
			if isfloat(element) == True:
				height.append(element)
	return np.asarray(height,float), speedcolname

def dirheight(namelist):
    """
    This function finds the data from the wind wave in a list of strings.
    """
    height, dircolname,strlist = [], [], []
    for name in namelist:
    	if name.find('Dir') >= 0:
    		dircolname.append(name)
    		for element in name.split(' '):
    		    strlist.append(element)
    		    for nextelement in element.split('_'):
    		        strlist.append(nextelement)	
	for element in strlist:
		if isfloat(element) == True:
			height.append(element)
	return np.asarray(height,float), dircolname
		
def logprofiledata(foldername, station_no, files, plottype, startdate, enddate, Nsectors = 16):
    """
    This function plots vertical wind profiles.
    """
    import os
    subfolder = foldername + "/" + str(station_no)
    os.mkdir(subfolder)
    cols1 = dataset(path+files[0],criterion,startdate, enddate)[0]
    cols2 = dataset(path+files[1],criterion,startdate, enddate)[0]
    dircolname1, dataarray1 = xyNsectors(cols1, Nsectors)
    dircolname2, dataarray2 = xyNsectors(cols2, Nsectors)
    tolerance = round(360./(2*Nsectors),2)
    for name1 in dircolname1:
        for name2 in dircolname2:
            dirfoldername = subfolder + "/" + name1 + "-" +  name2
            os.mkdir(dirfoldername)	
            x1N, x2N, y1N, y2N = [], [], [], []
            for i in range(np.size(dataarray1)):
                if name1 == dataarray1['Wind Vane'][i]:
                    x1N.append(dataarray1['Speed'][i])
                    y1N.append(dataarray1['Height'][i])
            for i in range(np.size(dataarray2)):
                if name2 == dataarray2['Wind Vane'][i]:
                    x2N.append(dataarray2['Speed'][i])
                    y2N.append(dataarray2['Height'][i])
            threshold = plottype
            if threshold == "pointplot":
                for i in range(len(x1N)):
                    pointplotter(dirfoldername, i+1, 
                                name1, name2, 
                                dataarray1['Direction'][i],             
                                dataarray2['Direction'][i], 
                                dataarray1['Anemometer'][i].T, 
                                dataarray2['Anemometer'][i].T,
                                station_no, tolerance, 
                                x1N[i].T, y1N[i].T, x2N[i].T,y2N[i].T)
            if threshold == "logplot":
			    for i in range(len(x1N)):
			        uarray, zarray, alphas = logdata(x1N[i].T, y1N[i].T, x2N[i].T, y2N[i].T, 100)
			        logplotter(dirfoldername, i+1 , 
			                    name1, name2,
			                    dataarray1['Direction'][i],
			                    station_no, tolerance, 
			                    uarray, zarray, alphas)
	return "" 

def xyNsectors(cols, Nsectors):
    """
    This function categorizes data into sectors.
    """
    namelist = cols.dtype.names
    speed_height, speedcolname = speedheight(namelist)
    dir_height, dircolname = dirheight(namelist)
    bin = 360./Nsectors
    x_all, y_all, dircollist, speedcollist, sectorlist, directionlist = [],[],[],[],[],[]
    for k in range(len(dircolname)):
        counter = 1
        direction  = 0
        lower,upper = direction - bin/2., direction + bin/2.
        #Categorizing into sectors within the directions
        directionlist = []
        while upper <= 360.:
            x_av , y_av, x_dir, y_dir,speedlist = [] , [] ,[] , [], []
            #Filtering for every speed column by direction
            for j in range(len(speedcolname)): #Searching through all rows in the dataset
                for i in range(len(cols[dircolname[k]])): 
                #Categorizing the speed values in bins
                    if lower <= cols[dircolname[k]][i] <= upper:
                        y_dir.append(cols[dircolname[k]][i])
                        x_dir.append(cols[speedcolname[j]][i])
                #Assembling the average speed in every sector
                x_av.append(np.average(np.asarray(x_dir,float)))
                y_av.append(speed_height[j])
            speedlist.append(speedcolname[j])
            speedcollist.append(np.asarray(speedlist,'S50'))
            x_all.append(np.asarray(x_av,float))
            y_all.append(np.asarray(y_av,float))
            dircollist.append(dircolname[k])
            sectorlist.append(counter)
            directionlist.append(direction)				
            direction += bin
            lower,upper = direction - bin/2., direction + bin/2.
            counter += 1
    dt = np.dtype([('Wind Vane', 'S50'), ('Sector','i'), ('Direction', 'f8'), ('Anemometer', 'S50', (1,len(speedcollist[0]))), ('Speed', float, (1,len(x_all[0]))), ('Height', float,  (1,len(y_all[0])))])
    dataarray = np.array(zip(dircollist, sectorlist, directionlist, speedcollist,x_all,y_all), dtype = dt)
    return dircolname, dataarray

def logdata(x1array_int, y1array_int, x2array_int, y2array_int, N):
    """
    This function returns arrays of combinations of u,z with different wind shear coefficients.
    """
    u1s, z1s = simanalyzer(x1array_int, y1array_int)
    u2s, z2s = simanalyzer(x2array_int, y2array_int)
    u = np.concatenate((u1s, u2s))
    z = np.concatenate((z1s, z2s))
    alphas = windshear(u, z)
    zarray = np.zeros((N,int(round(len(alphas['alpha'])**2))),float)
    uarray = np.zeros((N,int(round(len(alphas['alpha'])**2))),float)
    counter = 0
    alphalist = []
    for k, alpha in enumerate(alphas['alpha']):
        for j in range(len(alphas['ui'])):
            zarray[:,counter] = np.linspace(0,max(z),N)
            uarray[:,counter] = alphas['ui'][j]*(zarray[:,counter]/alphas['zi'][j])**alpha
            alphalist.append(alphas['alpha'][k])
            counter += 1
 #           print counter, alphas['ui'][j],alphas['uj'][j],alphas['zi'][j],alphas['zj'][j],alphas['alpha'][k]
    return uarray, zarray, np.asarray(alphalist,float)
    

def windshear(u,z):
    """
    This function returns an array of wind shear coefficients from the power law relation u(z)=u(z1)(z/z1)^alpha.
    """
    ui, uj, zi, zj, alpha = [], [], [], [], []
    for i in range(len(u)):
        A = range(len(u))
        del A[i]
        for j in A:
            ui.append(u[i])
            uj.append(u[j])
            zi.append(z[i])
            zj.append(z[j])                   
            alpha.append(np.log(u[i]/u[j])/np.log(z[i]/z[j]))
    dt = [('ui',float), ('uj',float), ('zi', float), ('zj', float), ('alpha', float)]
    alpha_ar = np.asarray(zip(ui,uj,zi,zj, alpha), dt)
    return np.sort(alpha_ar, order='alpha')[0::2]
    
def simanalyzer(xarray_int, yarray_int):
    """
    This function returns a new array with the average speed value for speed values at the     same height.  
    """
    xarray = np.array(xarray_int,float)
    yarray = np.array(yarray_int,float)
    x,y,xindex = [],[], []
    for i in range(len(yarray)):
        A = range(len(yarray))
        del A[i]
        for j in A:
            if yarray[i] == yarray[j]:
                x.append(xarray[i])
                xindex.append(i)
                y.append(yarray[i])
    x_av = np.average(np.asarray(x,float))
    x_temp = np.delete(xarray,xindex)
    x_new = np.insert(x_temp,0,np.array((x_av)))
    y_temp = np.delete(yarray, xindex)
    y_new = np.insert(y_temp,0,np.array((y[0])))
    return x_new , y_new
         
def logplotter(folder, sectorcounter, name1,name2, direction1, station_no, tolerance, uarray, zarray, alphas):
    """
    This function generates plots of logarithmic vertical wind profiles.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines 
    plt.hold('on')
    titlefont = {'fontname': 'Arial', 'size': '15', 'weight':'normal'}
    axisfont = {'fontname': 'Arial', 'size': '14'}
    fig, ax = plt.subplots(1)
    linestyle = matplotlib.lines.lineStyles.keys()[3:7]* np.shape(uarray)[1]
    degrees = 0
    for j in range(np.shape(uarray)[1]):
        ax.plot(uarray[:,j], zarray[:,j], label = (r"$\alpha = $ %.2f") % alphas[j],ls = linestyle[j])
        ax.annotate(j+1, (uarray[int(round(np.shape(uarray)[1]*0.9)), j],         
	                zarray[int(round(np.shape(uarray)[1]*0.9)), j]), 
	                xytext = (24*np.cos(degrees*(np.pi/180))-12*np.sin(degrees*(np.pi/180)), 24*np.sin(degrees*(np.pi/180))+12*np.cos(degrees*(np.pi/180))), 
	                textcoords = 'offset points',
                    arrowprops = dict(arrowstyle = '-', 
                    connectionstyle = 'arc3,rad=0'),
                    rotation = 0)
        degrees += (-30)
	textlist = [[str(name1 + "-" + name2)]] 
	ax.set_title(r"Vertical Logarithmic profile (dir: %.1f$^\circ$,+/-%.2f) Sector %d" % (direction1, tolerance,sectorcounter),**titlefont)
	ax.legend(loc='best',fontsize = 10)
	ax.set_xlabel("Speed (m/s)", **axisfont)
	ax.set_ylabel("Height (m)", **axisfont)
	plt.xlim(0,8)
	fig.savefig(folder + "/" + "Sector_%d-%s-%s.png" % (sectorcounter, station_no, direction1)) 
	plt.hold('off')
	return " "

def pointplotter(folder, counter, name1, name2, direction1, ano1, ano2, station_no, tolerance, x1 , y1, x2, y2):
    """
    This function generates plots of the average values of the points.
    """
    import matplotlib.pyplot as plt
    plt.hold('on')
    titlefont = {'fontname': 'Arial', 'size': '15', 'weight':'normal'}
    axisfont = {'fontname': 'Arial', 'size': '14'}
    fig, ax = plt.subplots(1)
    plot1, = ax.plot(x1 , y1, "r+", linestyle = "None", markersize=6) 
    plot2, = ax.plot(x2, y2, "k+", linestyle = "None", markersize=6)
    textlist = [[str(name1 + "-" + name2)]] 
    degrees = 0
    for i,txt in enumerate(ano1):
	    ax.annotate(i+1, 
	                (x1[i],y1[i]), 
	                xytext = (24*np.cos(degrees*(np.pi/180))-12*np.sin(degrees*(np.pi/180)), 24*np.sin(degrees*(np.pi/180))+12*np.cos(degrees*(np.pi/180))), 
	                textcoords = 'offset points',
                    arrowprops = dict(arrowstyle = '-', 
                    connectionstyle = 'arc3,rad=0'),
                    rotation = 0)
            textlist.append([i+1, txt])
            degrees += (-30)
    for j,txt in enumerate(ano2):
        ax.annotate(j+i+2, (x2[j],y2[j]), xytext = (24*np.cos(degrees*(np.pi/180))-12*np.sin(degrees*(np.pi/180)), 24*np.sin(degrees*(np.pi/180))+12*np.cos(degrees*(np.pi/180))), textcoords = 'offset points', arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'), rotation =0)
        textlist.append([j+i+2, txt])
        degrees+= (-30)

	import csv
	with open(folder + "/" + "%s-Referencelist.txt"% station_no,"wb") as outfile:
	    csv_writer = csv.writer(outfile)
            for element in textlist:
                csv_writer.writerow(element)
	ax.set_title(r"Vertical Profile (dir: %.1f$^\circ$,+/-%.2f) Sector %d" % (direction2, tolerance,counter),**titlefont)
	ax.legend([plot1, plot2],["Ice filtered set", "Manually corrected set"], loc=0 )
	ax.set_xlabel("Speed (m/s)", **axisfont)
	ax.set_ylabel("Height (m)", **axisfont)
#	ax.set_ylim(0, 65)
#	ax.set_xlim(0, 8)
	fig.savefig(folder + "/" + "Sector_%d-%s-%s.png" % (counter, station_no, direction1)) 
	plt.hold('off')
	return " "
 
def windrosecontour(ws,wd):
    """
    This function generates wind roses with contour with a given speed list and associated wind direction.
    """
    from windrose import WindroseAxes
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    ax = WindroseAxes.from_ax()
    ax.contourf(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot)
    ax.contour(wd, ws, bins=np.arange(0, 8, 1), colors='black')
    ax.set_legend()
    plt.show()

def windrosebar(ws,wd):
    """
    This function generates wind roses from a list of speed values and associated wind direction.
    """
    from windrose import WindroseAxes
    from matplotlib import pyplot as plt
    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.9, edgecolor='white')
    ax.set_legend()
    plt.show()
    bins = ax._info['bins']
    direction = ax._info['dir']
    table = ax._info['table']
    print bins
	
def weibull(Nsector):
	return ""		
		
def tiflist(path):
    """
    This function returns a list of filenames that ends with .tif in a folder.
    """
    import os
    os.path.dirname(path)
    dirs = os.listdir(path)
    tiflist = []
    for file in dirs:
        if file.endswith('tif'):
            tiflist.append(file)

def to_percent(y, position):
    import matplotlib
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    # The percent symbol needs escaping in latex
 
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def RHhist(): 
    """
    This function makes histogram of data of relative humidity. 
    The code is inspired from this website: 
    http://stackoverflow.com/questions/5328556/histogram-matplotlib
    """  
    import os,sys
    import matplotlib.pyplot as plt
    import matplotlib.axes as mx
    from matplotlib.ticker import FuncFormatter
    
    path = "Moisture-data"
    dirs = []
    for dir in os.listdir(path):   
        if dir.endswith(".csv"):
            dirs.append(dir.decode("utf-8"))        
    fig,ax = plt.subplots (3,1)
    num_bins = 10
    titlefont = {'fontname': 'Arial', 'size': '8', 'weight':'semibold'}
    subtitlefont = {'fontname': 'Arial', 'size': '6', 'weight':'normal'}
    axisfont = {'fontname': 'Arial', 'size': '6'}
    subfont = {'fontname': 'Arial', 'size': '7', 'weight': 'semibold'}
    for i,filename in enumerate(dirs):
        file = filename
        dataset = RH_ar(path + "/" + filename)
        RH_name = dataset.dtype.names[2]
        n, bin, patches = ax[i].hist(dataset[RH_name], range(0,101,num_bins), facecolor='green', alpha=0.8)
        ax[i].grid(True)
        stepx = float(bin[1]-bin[0])*0.25
        stepy = 250
        for b,el in zip(bin,n):
            percent = 100*el/float(len(dataset[RH_name]))           
            ax[i].text(b+stepx,el+stepy, "%.1f" % percent + r"$\%$" , **subtitlefont)
        ax[i].set_xlabel(r"Relative humidity($\%$)",**axisfont)
        ax[i].set_ylabel("Number of counts", **axisfont)
        ax[i].set_ylim([0,max(n)+1000])
        ax[i].set_title(u"%s" % file.replace(".csv",""), **subfont)
        ax[i].tick_params(labelsize=6)
        fig.set_size_inches(4, 4)
    fig.suptitle("The distribution of relative \n humidity between 22/02/2009 and 23/02/2010", **titlefont)
    fig.subplots_adjust(hspace = 0.7, top = 0.85)
    filename = path + "/" + "RH-histograms.png"
    if os.path.isfile(filename):
        os.remove(filename)
    fig.savefig(filename,dpi=800)
    return ""

def cpplot():
    import matplotlib.pyplot as plt
    from scipy import stats
    import matplotlib

    matplotlib.rc('text', usetex = True)
    matplotlib.rc('font', **{'family' : "sans-serif"})
    params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
    plt.rcParams.update(params)
    
    fig, ax = plt.subplots(1)
    
    T = np.array((240, 260, 280, 290, 298.15, 300, 320),float)
    h = np.array((240267, 260323,280390, 290430, 298615, 300473, 320576),float)

    #Performing the linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(T, h)
    print "Slope:", slope
    print "Intercept:", intercept
    print "r_value: ", r_value
    print "p_value: ", p_value 
    print "std_err: ", std_err
    
    t = np.linspace(T[0],T[-1],10)
    h_reg = slope*t + intercept
    
    titlefont = {'fontname': 'Arial', 'size': '8', 'weight':'bold'}    
    axisfont = {'fontname': 'Arial', 'size': '6'}

    line2 = ax.plot(t,h_reg/1000, label = "Regression analysis of enthalpy", color = "yellow", lw=0.5) 
    line1 = ax.plot(T, h/1000, linestyle = 'None', color = "black", marker = 'o', label = "Data points", markersize = 3)
    ax.set_xlim([230,330])
    ax.set_ylim([200,400])
    ax.grid(True) 
    ax.set_title("A visualization of the definition of heat capacity", **titlefont)
    ax.set_xlabel("Temperature (K)", **axisfont)
    ax.set_ylabel("Enthalpy (kJ/kg)", **axisfont)
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax.legend(lns , labs,  fontsize = 5, loc = 'best')
    fig.set_size_inches(4,3)
    filename = "../Figures/" + "Cp-plot.png"
    import os
    if os.path.isfile(filename):
        os.remove(filename)
    fontProperties = {'family':'Arial', 'weight': 'normal', 'size': 5}
    ax.set_xticklabels(ax.get_xticks(), fontProperties)
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    fig.tight_layout()
    fig.savefig(filename, dpi = 1000)
    return ""

def sat_vap_p(func_name,T_C):
    """
    This function returns the saturation vapor pressure (in Pa) for solid and liquid water as a function of temperature in Kelvin units. The equations are proposed by Sonntag (1990). 
    """ 
    T = T_C + 273.15 #Converting temperature from the Celsius scale to Kelvin.
    if func_name == "ice":
        if 173.15 <= T <= 273.16:
            return 100*np.exp(24.7219 - (6024.5282/T) + (1.0613868*10**-2)*T - (1.3198825*10**-5)*T**2 - 0.49382577*np.log(T))
        else:
            return "The function of saturation vapor pressure for ice is not valid at this temperature."
    if func_name == "liq":   
        if 173.15 <= T <= 373.15:
            return 100*(np.exp(16.635764 - (6096.9385/T) - (2.711193*10**-2)*T + (1.673952*10**-5)*T**2 + 2.433502*np.log(T)))
        else: 
            return "The function of saturation vapor pressure for liquid is not valid at this temperature."
    if func_name == "Buck":
        return 100*(6.1121*np.exp((18.678 - T_C/234.5)*(T_C/(257.14+T_C))))

def sat_vap_p_ar(func_name, t_ar):
    """
    This functions has an array of temperature value as input and returns an array of pressure values.
    """
    p = np.zeros((len(t_ar)),float)
    for i in range(len(t_ar)):
        p[i] = sat_vap_p(func_name, t_ar[i])
    return p 

def sat_vap_plot():
    import matplotlib.pyplot as plt

    import matplotlib

    matplotlib.rc('text', usetex = True)
    matplotlib.rc('font', **{'family' : "sans-serif"})
    params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
    plt.rcParams.update(params)

    t_ice = np.linspace(-30, 0.01,30)
    t_liq = np.linspace(-30,30,40)
    
    titlefont = {'fontname': 'Arial', 'size': '8', 'weight':'semibold', 'y': '0.95'}    
    axisfont = {'fontname': 'Arial', 'size': '7'}
    
    fig,ax = plt.subplots(2,1)
    
    line1 = ax[0].plot(t_liq,sat_vap_p_ar("liq",t_liq), ls = 'None', marker = "o", color = 'yellow', label = r'$e_{s,liquid}^\ast$ (S)', alpha = 0.5, markersize = 4)
    line2 = ax[0].plot(t_ice,sat_vap_p_ar("ice",t_ice), ls = 'None', marker = 'o', markersize = 2, color = 'blue', label = r'$e_{s,solid}^\ast$ (S)', alpha = 0.7)
    line3 = ax[0].plot(t_liq, sat_vap_p_ar("B", t_liq), ls = '-', color = 'red', label = r'$e_{s,solid+liquid}^\ast$ (Buck)', lw = 0.3)

    line4 = ax[1].plot(t_ice,sat_vap_p_ar("ice",t_ice) - sat_vap_p_ar("liq",t_ice), label = r"$e_{s,s}^\ast$(S) - $e_{s,l}^\ast$(S)", lw = 0.8)
    line5 = ax[1].plot(t_liq, sat_vap_p_ar("Buck", t_liq) - sat_vap_p_ar("liq", t_liq), label = r"$e_{s,s+l}^\ast$(B) - $e_{s,l}^\ast$(S)", lw = 0.8)
    line6 = ax[1].plot(t_ice, sat_vap_p_ar("Buck", t_ice) - sat_vap_p_ar("ice", t_ice), label = r"$e_{s,s+l}^\ast$(B) - $e_{s,s}^\ast$(S)", lw = 0.8)

    ax[0].set_xlabel(r"Temperature (\textcelsius)", **axisfont)
    ax[0].set_ylabel("Pressure (Pa)", **axisfont)
    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    leg = ax[0].legend(lns, labs, loc = 'best', fontsize = 5)
    leg.get_frame().set_alpha(0.6)
    leg.get_frame().set_linewidth(0.2)


    ax[1].set_xlabel(r"Temperature (\textcelsius)", **axisfont)
    ax[1].set_ylabel(r"Pressure (Pa)", axisfont)
    lns = line4 + line5 + line6
    labs = [l.get_label() for l in lns]
    leg = ax[1].legend(lns, labs, loc = 'best', fontsize = 5)
    leg.get_frame().set_alpha(0.6)
    leg.get_frame().set_linewidth(0.2)

    fontProperties = {'family':'Arial', 'weight': 'normal', 'size': 6}
    ax[0].set_xticklabels(ax[0].get_xticks(), fontProperties)
    ax[0].set_yticklabels(ax[0].get_yticks(), fontProperties)
    ax[1].set_xticklabels(ax[1].get_xticks(), fontProperties)
    ax[1].set_yticklabels(ax[1].get_yticks(), fontProperties)
    
    fig.subplots_adjust(hspace = 0.4)
    filename = "../Figures/psatfuncs.png"
    import os
    if os.path.isfile(filename):
        os.remove(filename)
    fig.set_size_inches(4,4)
    title = fig.suptitle("Saturation vapor pressure", **titlefont)
    fig.savefig(filename, dpi = 1000)
    return ""
    
def RH_correlation():
    """
    This function finds the correlation between the datasets of relative humidity around the planned area.
    """
    import os,csv,datetime
    path = "Moisture-data"
    dirs = []
    for dir in os.listdir(path):   
        if dir.endswith(".csv"):
            dirs.append(dir.decode("utf-8"))        
    cor_av = []
    index = 0
    counter = 0
    filename = "Moisture-comb.csv"
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'wb') as outfile:   
        csv_writer = csv.writer(outfile, delimiter = ' ')
        for filename1 in dirs:
            for filename2 in dirs:
                if filename1 != filename2:
                    lloop = 1
                    kloop = 1
                    cor = []
                    date1, time1, RH1 = Set_content(RH_ar(path + "/" + filename1))
                    date2, time2, RH2 = Set_content(RH_ar(path + "/" + filename2))
                    for k in range(len(date1)):
                        for l in range(len(date2)):
                            if date1[k] == date2[l] and time1[k] == time2[l]:
                                a = (1. - abs(float((RH2[l] - RH1[k]))/RH1[k]))
                                cor.append(a)
                                index += 1
                                break
                            lloop += 1
                        kloop += 1
                    csv_writer.writerow(["loop: ", counter, "file 1:", filename1[0:2], "file 2: ", filename2[0:2]])
                    csv_writer.writerow(["Number of outer loops: ", kloop])
                    csv_writer.writerow(["Number of inner loops: ", lloop])
                    csv_writer.writerow(["Length correlation list ", len(cor)])
                    csv_writer.writerow(["Average value of correlation: ", np.average(np.asarray(cor,float))])
                counter += 1
    return ""

def RH_matrix(date_m,set_content):
    import datetime
    date, time, RH = set_content
    RH_m = np.zeros([len(date_m)],float)
    for i in np.arange(len(date_m)):
        for j in np.arange(len(date)):
            if date_m[i] == date[j] + " " + time[j]:
                RH_m[i] = RH[j]
                break
    return RH_m

def Correlation_Fischer2D(date_m):
    """
    This function finds the correlation between the datasets of relative humidity around the planned area.
    """
    import os
    path = "Moisture-data"
    dirs = []
    for dir in os.listdir(path):   
        if dir.endswith(".csv"):
            dirs.append(dir.decode("utf-8"))
    for filename1 in dirs:
        for filename2 in dirs:
            if filename1 != filename2:
                RH1_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + filename1)))
                RH2_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + filename2)))
                indexlist = []
                for i in np.arange(len(RH1_m)):
                    if RH1_m[i] == 0.0 or RH2_m[i] == 0.0:
                        indexlist.append(i)
                RH1 = np.delete(RH1_m, indexlist)
                RH2 = np.delete(RH2_m, indexlist)
                N = len(RH1)
                av = sum(RH1+RH2)/(2.*N)
                s2 = (sum((RH1 - av)**2)+sum((RH2 - av)**2))/(2.*N)
                r = sum((RH1 - av)*(RH2 - av))/(2*N*s2)  
                print "The standard deviation is: ", np.sqrt(s2)
                print "The average value is: ", av
                print "The array length beforing deleting zero values was: ", len(RH1_m)
                print "The resulting array length after deleting zero values is: ", N              
                print "With RH1: ", filename1, "  and RH2: ", filename2
                print "The intraclass of coefficient is: ", r
    return ""

def Correlation_Fischer3D_ar(date_m):
    """
    This function finds the correlation between the datasets of relative humidity around the planned area.
    """
    import os
    path = "Moisture-data"
    dirs = []
    for dir in os.listdir(path):   
        if dir.endswith(".csv"):
            dirs.append(dir.decode("utf-8"))
    RH1_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[0])))
    RH2_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[1])))
    RH3_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[2])))
    indexlist = []
    for i in np.arange(len(RH1_m)):
        if RH1_m[i] == 0.0 or RH2_m[i] == 0.0 or RH3_m[i] == 0.0:
            indexlist.append(i)
    RH1 = np.delete(RH1_m, indexlist)
    RH2 = np.delete(RH2_m, indexlist)
    RH3 = np.delete(RH3_m, indexlist)
    N = len(RH1)
    av = sum(RH1+RH2+RH3)/(3.*N)
    s2 = (sum((RH1 - av)**2)+sum((RH2 - av)**2)+sum((RH3 - av)**2))/(3.*N)
    r = sum((RH1 - av)*(RH2 - av)+(RH1 - av)*(RH3 - av) + (RH2 - av)*(RH3 - av))/(3*N*s2)
    print "The standard deviation is: ", np.sqrt(s2)
    print "The average value is: ", av
    print "The array length beforing deleting zero values was: ", len(RH1_m)
    print "The resulting array length after deleting zero values is: ", N
    return r
    
def Correlation_Fischer3D(date_m):
    """
    This function finds the correlation between the datasets of relative humidity around the planned area.
    """
    import os
    path = "Moisture-data"
    dirs = []
    for dir in os.listdir(path):   
        if dir.endswith(".csv"):
            dirs.append(dir.decode("utf-8"))
    RH1 = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[0])))
    RH2 = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[1])))
    RH3 = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[2])))
    av = (np.average(RH1) + np.average(RH2) + np.average(RH3))/3.
    s2 = (np.std(RH1)**2 + np.std(RH2)**2 + np.std(RH3)**2)/3.
    N = len(RH1)
    r,i = 0,0
    while i < N:
        sum = ((RH1[i]-av)*(RH2[i]-av))+((RH1[i]-av)*(RH3[i]-av)) + ((RH2[i] - av)*(RH3[i] - av))
        r += sum
        i += 1
    return r/(3*N*s2)

def Set_content(dataset):
    """
    This function compresses the code for retrieving data from the datasets provided by SMHI.
    """
    date = dataset[dataset.dtype.names[0]]
    time = dataset[dataset.dtype.names[1]]
    var = dataset[dataset.dtype.names[2]]
    return date,time, var
      
answer = "no"
if answer == "yes":
	path = "Elevation/"		
	data = transformer(path+tiflist(path)[2])
	from Stringwriter import *
	filename = ('Combinations.txt')
	filename_heights = "Dataheights-Ostersund_verified.txt"
	comb_arr = combreader(filename)
	heights = heightsreader(filename_heights)
	LSlist = comb_UTM_array(comb_arr,heights)
	counter = 1
	path = 'ElevationProfiles/'
	for i in range(len(LSlist)):
		linestring = [(LSlist['lo1'][i],LSlist['la1'][i]),(LSlist['lo2'][i],LSlist['la2'][i])]
		funcxyz = elevraster(linestring,[data[0],data[1]], data[2],data[3])
		plotter(path, funcxyz, dir(linestring), [LSlist['M1'][i],LSlist['M2'][i]],counter)
		counter+=1

plot_1 = "no"
	
if plot_1 == "yes":
	path = "Elevation/"		
	data = transformer(path+tiflist(path)[2])
	from Stringwriter import *
	filename = ('Combinations.txt')
	filename_heights = "Dataheights-Ostersund_verified.txt"
	comb_arr = combreader(filename)
	heights = heightsreader(filename_heights)
	LSlist = comb_UTM_array(comb_arr,heights)
	counter = 1
	path = 'ElevationProfiles/'
	linestring = [(LSlist['lo1'][0],LSlist['la1'][0]),(LSlist['lo2'][0],LSlist['la2'][0])]
	funcxyz = elevraster(linestring,[data[0],data[1]], data[2],data[3])
	plotter(path, funcxyz, dir(linestring), [LSlist['M1'][0],LSlist['M2'][0]],counter)

looping = "no"

if looping == "yes":
	path = "Elevation/"		
	data = transformer(path+tiflist(path)[2])
	from Stringwriter import *
	filename = ('Combinations.txt')
	filename_heights = "Dataheights-Ostersund_verified.txt"
	comb_arr = combreader(filename)
	heights = heightsreader(filename_heights)
	LSlist = comb_UTM_array(comb_arr,heights)
	counter = 1
	a = np.zeros((len(LSlist),),dtype=[('index','i'), \
	('absSlope','f'),('R','f'),('Std','f'),('M1','i'),('M2','i')])

	for i in range(len(LSlist)):
		linestring = [(LSlist['lo1'][i],LSlist['la1'][i]),\
		(LSlist['lo2'][i],LSlist['la2'][i])]

		x,y,z,color,dist, diag = elevraster(linestring,[data[0],data[1]], data[2],data[3])
		a['absSlope'][i], a['R'][i], a['Std'][i] = \
		standard_dev(np.asarray(dist,float)*1000,np.asarray(z,float))

		a['M1'][i] = LSlist['M1'][i]
		a['M2'][i] = LSlist['M2'][i]
		a['index'][i] = counter
		counter+=1

		writecsv = "no"

	if writecsv == "yes":
		import csv
		with open('slopetable.txt','wb') as outfile:
			csv_writer = csv.writer(outfile, delimiter = ' ')
			for row in np.sort(a,order='absSlope'):
				csv_writer.writerow(row)

windroseplot = "no"

if windroseplot == "yes":
	path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
	import os
	dirs = os.listdir(path)
	import numpy as np
	cols = dataset(path+dirs[0],9)[0]
	speed_height, speedcolname = speedheight(cols.dtype.names)
	dir_height, dircolname = dirheight(cols.dtype.names)
	ws = cols[speedcolname[0]]
	wd = cols[dircolname[0]]
	windrosebar(ws,wd)
	
pointplot = "no"

if pointplot == "yes":
	import os,shutil
	path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
	dirs = os.listdir(path)
	criterion = 9
	samefiles = []
	plot_ = "pointplot"
	for i in range(len(dirs)-1):	
	    if dirs[i+1][0:4]==dirs[i][0:4]:
	        samefiles.append([dirs[i][0:4],[dirs[i],dirs[i+1]]])
	foldername = "Vertical-wind-profiles"
	if os.path.isdir(foldername):
		shutil.rmtree(foldername)
	os.mkdir(foldername)
	for elements in samefiles:
		station_no, files = elements
		logprofiledata(foldername, station_no, files,plot_)

logprof = "no"

if logprof == "yes":
	import os,shutil
	path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
	dirs = os.listdir(path)
	criterion = 9
	samefiles = []
	plottype = "logplot"
	startdate, enddate = "2009-02-23", "2010-02-22"
	
	for i in range(len(dirs)-1):	
	    if dirs[i+1][0:4]==dirs[i][0:4]:
			samefiles.append([dirs[i][0:4],[dirs[i],dirs[i+1]]])
	foldername = "Vertical-wind-profiles-log"
	if os.path.isdir(foldername):
		shutil.rmtree(foldername)
	os.mkdir(foldername)
	for elements in samefiles:
		station_no, files = elements
		logprofiledata(foldername, station_no, files, plottype, startdate, enddate)

moisture = "no"
if moisture == "yes":
    import os
    path = "Moisture-data"
    for filename in os.listdir(path):
	    if filename.endswith('.csv'):
	        print filename

cor = "yes"

if cor == "yes":
    #Making the RH_matrices in the same dimension.
    import datetime
    startdate = datetime.date(2009,02,23)
    starttime = datetime.time(0,0,0)
    start = datetime.datetime.combine(startdate,starttime)
    enddate = datetime.date(2010,02,22)
    endtime = datetime.time(23,59,59)
    end = datetime.datetime.combine(enddate,endtime)
    
    datefield_list = []
    while start <= end:
        datefield_list.append(str(start))
        start = start + datetime.timedelta(seconds=3600) 
    date_m = np.asarray(datefield_list)
    print "The intraclass correlation coefficient is: ", Correlation_Fischer3D_ar(date_m)
#    Correlation_Fischer2D(date_m)

#cpplot()


"""
for i in range(len(datearray)):
    if  period1 == datearray[i]: 
        counterlist.append(i)
    elif period2 == datearray[i]:
#        counterlist.append(i)
#print counterlist
#print dataset(path+dirs[0], criterion)[0]['Date Field'][counterlist[0]], dataset(path+dirs[0], criterion)[0]['Date Field'][counterlist[-1]]
#	logprofile(cols.dtype.names, cols, 270)
#	print logprofile(path)

#	import pprint
#	pp = pprint.PrettyPrinter(indent = 4)
#	pp.pprint(dataarray1)

	textstr = ""
	for i in range(len(textlist)):
	    texting = "%d: %s \n" % (textlist[i][0], textlist[i][1][0])
	    textstr+= texting
	props = dict(boxstyle = "square", facecolor = "white", alpha = None, ec = 'k')
	ax.text(0.95,0.05, textstr, 
	        verticalalignment = "bottom", 
	        horizontalalignment = "right", 
            transform = ax.transAxes,
            color = "k",
            fontsize = 9,
            bbox = props)
"""