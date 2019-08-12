 # -*- coding: utf-8 -*-

import numpy as np
import datetime,os,sys
print datetime.datetime.today()

def dataset(st_no, type, criterion,startdate="2009-02-23",enddate="2010-02-22"):
	#setting __doc__
	"""
	Criterion is the number of elements.
	The dates must be set on the form YYYY-MM-DD as a string.
	"""
	path_n1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave-Data/"
	path_n = path_n1 + "Skogsvalidering WindSim SCA/wind data/TIL SSVAB"
	for file in os.listdir(path_n):
	    if file.find(type) >= 0 and file.find("%s"%str(st_no)) >= 0:
	        filename = os.path.abspath(path_n + "/" + file)
	    if file.find(type) >= 0 and file.find("%s"%str(st_no)) >= 0:
	        filename = os.path.abspath(path_n + "/" + file)          
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

def RH_ar(filename,startdate = "2009-02-23", enddate = "2010-02-22"):
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

def heightsreader(filename):
	infile = open(filename, "r")
	infile.readline()
	list_ = []
	station_no, elevation, UTM_la, UTM_lo = [],[],[],[]
	for line in infile:
		station_no.append(eval(line.split(' ')[0]))
		elevation.append(eval(line.split(' ')[1]))
		UTM_la.append(eval(line.strip(' ').split(' ')[2]))
		UTM_lo.append(eval(line.strip('\n').split(' ')[3]))
	heights = zip(station_no, elevation, UTM_la, UTM_lo)
	dt = np.dtype([('Station Number', 'i'),('Elevation', 'float'),('Latitude','float'),\
	 ('Longitude', 'float')])
	return np.array(heights, dtype = dt)
	
def combreader(filename):
	infile = open(filename,'r')
	infile.readline()
	metmast1,metmast2 = [],[]
	for line in infile:
		metmast1.append(eval(line.split('	')[0]))
		metmast2.append(eval(line.split('	')[1]))
	dt = [('Met Mast 1','i'), ('Met Mast 2','i')]
	return np.array(zip(metmast1,metmast2),dtype = dt)

def new_matrix_reader(varname, st_no):
    import os
    if varname == "RH":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_n/rhmatrices"
        for file in os.listdir(foldername2):
            if file.find("%s"%str(st_no)) >= 0:
                infile = open(os.path.abspath(foldername2 + "/" + file),'r')
        lines = infile.readlines()
        date,time,var = [],[],[]
        for line in lines:
            row = line.strip('\r\n').split(',')
            date.append(row[0])
            time.append(row[1])
            var.append(row[2])
        dt = [('Date', 'S10'),('Time', 'S8'),('RH',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    if varname == "P":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_n/pmatrices"
        for file in os.listdir(foldername2):
            if file.find("%s"%str(st_no)) >= 0:
                infile = open(os.path.abspath(foldername2 + "/" + file),'r')
        lines = infile.readlines()
        date,time,var = [],[],[]
        for line in lines:
            row = line.strip('\r\n').split(',')
            date.append(row[0])
            time.append(row[1])
            var.append(row[2])
        dt = [('Date', 'S10'),('Time', 'S8'),('P',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    else: 
        return "Please choose between RH, P and specify the met mast."

def comb_UTM_array(comb_arr, heights):
	dt = np.dtype([('M1','i'), ('la1','float'),('lo1','float'),('M2','i'),('la2','float'),('lo2','float')])
	comb_UTM_arr = np.zeros((len(comb_arr),),dtype = dt)
	for i in range(len(comb_arr)):
		for j in range(len(heights)):
			if comb_arr['Met Mast 1'][i] == heights['Station Number'][j]:
				comb_UTM_arr['M1'][i] = comb_arr['Met Mast 1'][i]
				comb_UTM_arr['la1'][i] = heights['Latitude'][j]
				comb_UTM_arr['lo1'][i] = heights['Longitude'][j]
			elif comb_arr['Met Mast 2'][i] == heights['Station Number'][j]:
				comb_UTM_arr['M2'][i] = comb_arr['Met Mast 2'][i]
				comb_UTM_arr['la2'][i] = heights['Latitude'][j]
				comb_UTM_arr['lo2'][i] = heights['Longitude'][j]
 	return comb_UTM_arr

def multilinestring(list_):
	endstring = ""
	for row in list_:
		line = "(%d %d,%d %d)," % (row[2],row[1],row[5],row[4]) 
		endstring+=line 
	return "MULTILINESTRING("+ endstring + ")"

def linestrings(list_):
	with open("linestrings.txt","wb") as outfile:
		csv_writer = csv.writer(outfile)
		counter = 1
		for row in list_:
			csv_writer.writerow(["%.5f, %.5f,%.5f, %.5f"% \
			(row[2], row[1],row[5], row[4])])
			counter +=1
def linestringlist(list_):
	linestringlist = []
	for row in list_:
		linestringlist.append(["LINESTRING(%.5f %.5f,%.5f %.5f)"% \
			(row[2], row[1],row[5], row[4])])
	return linestringlist

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
    This function returns the interpolated values along a line on a rasterized image. 
    """
    from shapely.geometry import LineString
    resx,resy = gridresolution
    line = LineString(linestring)
    length = line.length
    diagonal = np.sqrt(resx**2+resy**2)
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
    import osgeo.gdal as gdal 
    from gdalconst import GA_ReadOnly
    dataset = gdal.Open(filename, GA_ReadOnly)
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    bandtype = gdal.GetDataTypeName(band.DataType)
    return gt[1], gt[5], dataset, band.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize).astype(np.float) 

def elevation_plotter(path, x,z, angle, st_no1,st_no2,diagonal, counter):
    """
    This function plots the elevtion profile between two meteorological masts.
    """ 
    import matplotlib.pyplot as plt
    titlefont = {'fontname': 'Arial', 'size': '15', 'weight':'normal'}
    axisfont = {'fontname': 'Arial', 'size': '14'}
    fig = plt.figure(counter)
    plt.plot(x,z,linewidth = 2, color = 'black')
    plt.title('Elevation profile for the combination %d-%d (dir:%d' \
    %(st_no1,st_no2,angle)+ r'$^\circ$)', **titlefont) 
    plt.xlabel('Distance (km)',**axisfont)
    plt.ylim(0,600)
    plt.ylabel('Elevation (m)',**axisfont)
    plt.xlim(x[0],x[-1])
    fig.savefig(path + '%d-Combination_%d_%d_res%d.png'%(counter,st_no1,st_no2,diagonal),dpi=1000)
    return ""
    
def flatness_par(x,y):
    """
    This function calculates a flatteness parameter with the secant method and returns the average value.
    """
    der_y = np.zeros((len(x)))
    for i in range(1,len(x)):
        der_y[i] = abs((y[i] - y[i-1])/(x[i] - x[i-1]))
    return sum(der_y)/len(der_y)

def flattness_ex_main():
    var = [np.linspace(1,10,100), np.linspace(1,20,100), np.linspace(1,40,100),np.linspace(1,80,100), np.linspace(1,160,100)]
    y1, y2, y3, y4, y5, y6,y7 = [],[],[],[],[],[],[]
    for x in var:
        y1.append(flatteness_par(x,2*x))
        y2.append(flatteness_par(x,np.exp(-x)))
        y3.append(flatteness_par(x,x**2))
        y4.append(flatteness_par(x,x**3))
        y5.append(flatteness_par(x,np.sin(x*180/np.pi)))
        y = np.ones(len(x))*10
        y6.append(flatteness_par(x,y))
        y7.append(flatteness_par(x,(x + 5)/(x**2 - 10)))
    
    m = np.ones(len(var))*10
    for i in range(1,len(var)):
        m[i] = m[i-1]*2
    
    for i in range(len(var)):
        print m[i], y1[i],y2[i], y3[i], y4[i], y5[i], y6[i], y7[i]

def elevation_comb_loop(path):
    """
    This function loops a function with the combination of meteoroligical masts.
    """
    tif_filename = "Elevation/Elev_merged_5x5_30iter.tif"		
    resx,resy,set, dataset_ = transformer(tif_filename)
    diagonal = np.sqrt(resx**2+resy**2)
    filename = ('Combinations.txt')
    filename_heights = "Dataheights-Ostersund_verified.txt"
    comb_arr = combreader(filename)
    heights = heightsreader(filename_heights)
    LSlist = comb_UTM_array(comb_arr,heights)
    for i in range(len(LSlist)):
		ar1 = np.array((LSlist['lo1'][i],LSlist['la1'][i]))
		ar2 = np.array((LSlist['lo2'][i],LSlist['la2'][i]))
		x,y,z,color,dist, diagonal = elevraster([ar1,ar2],[resx,resy],set,dataset_)
		dist_ar = np.asarray(dist,float)
		z_ar = np.asarray(z,float)
		st_no1 = LSlist['M1'][i]
		st_no2 = LSlist['M2'][i]		
		elevation_stats_writer(path,i,dist_ar,z_ar,st_no1,st_no2,diagonal)
		#elevation_plotter(path, dist,z, dir(ar1,ar2), st_no1,st_no2,diagonal,i+1)
def elevation_stats_writer(path,counter, dist,z,st_no1,st_no2,diagonal):
    import os, shutil,datetime,csv
    date = datetime.datetime.today()
    filename = path + "/" + "%d-%d-%d-stats_res%.1f.txt" % (counter+1,st_no1,st_no2,diagonal)
    with open(filename, 'wb') as outfile:
        csv_writer = csv.writer(outfile,delimiter = ',')
        csv_writer.writerow(["Date & Time", date])
        csv_writer.writerow(["Combination: ", "%d-%d" %(st_no1, st_no2)])
        csv_writer.writerow(["Standard deviation (height): ", np.std(z)])
        csv_writer.writerow(["Average value (height): ", np.average(z)])
        csv_writer.writerow(["Median (height): ", np.median(z)])
        csv_writer.writerow(["Range (height): ", max(z) - min(z)])
        csv_writer.writerow(["Distance: ", abs(dist[-1] - dist[0])])
        csv_writer.writerow(["Minimum value(height): ", min(z)])
        csv_writer.writerow(["Maximum value(height): ", max(z)])
        csv_writer.writerow(["Number of elements: ", len(z)])
        csv_writer.writerow(["Var(height): ", np.var(z)])
        csv_writer.writerow(["Flatness parameter: ", round(flatness_par(dist,z),3)])
        csv_writer.writerow(["Standard median deviation (z)", sum(np.sqrt((z - np.median(z))**2))/len(z)])
    return ""

def elevation_stats_array(order1='dist'):
    """
    The contents of this file:
    index    Parameter
    0 Date & Time
    1 Met mast 1
    2 Met mast 2
    3 Standard deviation (height)
    4 Average value (height)
    5 Median (height)
    6 Range (height)
    7 Distance
    8 Minimum value(height)
    9 Maximum value(height)
    10 Number of elements: 
    11 Var(height)
    12 Flatness parameter 
    13 Standard Median deviation
    """
    import os
    foldername = "Elevation-statistics"
    date, comb, m1, m2, std, av = [],[],[],[],[],[]
    med, range, dist,min,max,no, var, flat, std_m = [],[],[],[],[],[],[],[],[]    
    for i, file in enumerate(os.listdir(foldername)):
        infile = open(foldername + "/" + file,'r')
        lines = infile.readlines()
        row = []
        for i,line in enumerate(lines):
            row.append(line.strip('\r\n').split(','))
        date.append(row[0][1])
        m1.append(row[1][1][0:4])
        m2.append(row[1][1][5:])
        std.append(row[2][1])
        av.append(row[3][1])
        med.append(row[4][1])
        range.append(row[5][1])
        dist.append(row[6][1])
        min.append(row[7][1])
        max.append(row[8][1])
        no.append(row[9][1])
        var.append(row[10][1])
        flat.append(row[11][1])
        std_m.append(row[12][1])
    dt = [('Date & Time', 'S10'), ('M1', 'i'), ('M2', 'i'), ('std', float), ('av',float),('med',float),('range',float),('dist', float), ('min',float), ('max', float), ('No_el', int), ('var',float), ('flat',float), ('std_m',float)]
    a = np.array(zip(date, m1, m2, std, av,med, range, dist,min,max,no, var, flat, std_m),dt)
    return np.sort(a, order = order1)

def dir(ar1,ar2):
    """
    This function returns the orientation of a vector.
    """ 
    x,y = ar1 - ar2
    return int(round(180 + np.arctan2(x,y)*(180/np.pi)))

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

def isfloat(value):
    """
    This function tests a value if it is a float and returns the value in floating point if true.
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
    date, time, var = set_content
    var_m = [None]*len(date_m)
    ilist = np.zeros([len(date_m)],int)
    jlist = np.zeros([len(date_m)],int)
    for i in np.arange(len(date)):
        for j in np.arange(len(date_m)):
            if date[i] + " " + time[i]== date_m[j] :
                var_m[j] = var[i]
                ilist[j] = i
                jlist[j] = j
                break
    return jlist, ilist,np.asarray(var_m)

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
                ind1, RH1_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + filename1)))
                ind2, RH2_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + filename2)))
                indexlist = []
                for i in np.arange(len(RH1_m)):
                    if RH1_m[i] != '' or RH2_m[i] != '':
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
    ind1, RH1_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[0])))
    ind2, RH2_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[1])))
    ind3, RH3_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[2])))
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
    print "The intraclass correlation coefficient is: " 
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
    ind1, RH1_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[0])))
    ind2, RH2_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[1])))
    ind3, RH3_m = RH_matrix(date_m, Set_content(RH_ar(path + "/" + dirs[2])))
    indexlist = []
    for i in np.arange(len(RH1_m)):
        if RH1_m[i] == 0.0 or RH2_m[i] == 0.0 or RH3_m[i] == 0.0:
            indexlist.append(i)
    RH1 = np.delete(RH1_m, indexlist)
    RH2 = np.delete(RH2_m, indexlist)
    RH3 = np.delete(RH3_m, indexlist)
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

def matrix_writer(filename, date_m, dataset):
    import os, csv
    if os.path.isfile(filename):
        os.remove(filename)
    ilist, jlist, var = RH_matrix(date_m,dataset)
    print filename, len(ilist),len(jlist), len(var)
    with open(filename,'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter = ' ')
        for i in range(len(var)):
            csv_writer.writerow([ilist[i],jlist[i],var[i]])
    return ""



def matrix_csv_main():
    foldername = "Matrices-ice_F"
    if os.path.isdir(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    p_path = "Pressure-data"
    p_folder = "KrangadeA-Lufftrykk.csv"
    matrix_writer(foldername + "/" + 'pmatrix.txt',date_m, Set_content(RH_ar(p_path + "/" + p_folder)))
    RH_path = "Moisture-data"
    RH_folder = "Krångede A.csv"
    matrix_writer(foldername + "/" + 'RH.txt', date_m, Set_content(RH_ar(RH_path + "/" + RH_folder)))
    path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    dirs = os.listdir(path)
    criterion = 9
    samefiles = []
    for i in range(len(dirs)-1):
        if dirs[i+1][0:4] == dirs[i][0:4]:                                    
            samefiles.append([dirs[i][0:4],[dirs[i],dirs[i+1]]])
    counter = 1
    for elements in samefiles:
        station_no, files = elements
        cols = dataset(path + files[0], criterion)[0]
        for name in cols.dtype.names:
            if name.find("T107") >= 0:
                matrix_writer(foldername + "/" + '%s-T107-ice_F.txt'% station_no,date_m, [cols["Date Field"], cols["Time Field"], cols[name]])
#            if name.find("termodiff") >= 0:
#                matrix_writer(foldername + "/" + '%s-termodiff.txt'% station_no,date_m, [cols["Date Field"], cols["Time Field"], cols[name]])
    return ""

def matrix_reader(filename):
    infile = open(filename,'r')
    lines = infile.readlines()
    ilist, jlist, varlist = [],[],[]
    for line in lines:
        row = line.strip('\r\n').split(' ')
        ilist.append(row[0])
        jlist.append(row[1])
        varlist.append(row[2])
    return ilist,jlist,varlist

def zlisting():
    import os
    path_n = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    criterion = 9
    zlist = []
    for filename in os.listdir(path_n):
        if filename.find('TOP') >= 0:                         
            cols = dataset(path_n + filename, criterion)[0]
            for name in cols.dtype.names:
                if name.find("T107") >= 0:
                    z_t = eval(name.split(' ')[-1])
                if name.find("termodiff") >= 0:
                    z_dt = eval(name.split(' ')[-1])
            zlist.append([filename[0:4],[z_t, z_dt]])
    return zlist

def Lapsecriteria_loop(precision, tolerance):
    import os,shutil
    path_m = "Matrices/"
    cp = 1003.78597374 #J/kgK
    g0 = 9.80665 # m/s2
    varGamma = round(g0/cp,precision)
    termlist, templist = [],[]
    for filename in os.listdir(path_m):
        if filename.find('pmatrix') >= 0: 
            ilist_p, jlist_p, p_ar = matrix_reader(path_m + filename)
        elif filename.find('RH') >= 0:
            ilist_rh, jlist_rh, RHar = matrix_reader(path_m + filename)
        elif filename.find('termodiff') >= 0:
            termlist.append([filename[0:4], filename])
        elif filename.find('T107') >= 0:
            templist.append([filename[0:4], filename])

    # From the function zlisting. The z-values are written to speed up the code.
    infile = open('zlist.txt','r')
    lines = infile.readlines()
    st_z, z_t,z_dt = [],[],[]
    for line in lines:
        row = line.strip('\n').split(', ')
        st_z.append(row[0])
        z_t.append(eval(row[1]))
        z_dt.append(eval(row[2]))
    critlist_all = []
    for i in range(len(st_z)):
        indexlist = []
        ind_it, ind_jt, t = matrix_reader(path_m + templist[i][1])
        ind_idt, ind_jdt, dt = matrix_reader(path_m + termlist[i][1])
        for j in range(len(RHar)):
            if p_ar[j] == '' or RHar[j] == '' or t[j] == '':
                indexlist.append(j)
        #Converting the lists of strings to float and integer arrays.
        #Converting from hPa to Pa
        p1 = np.array(np.delete(np.asarray(p_ar),indexlist),float)*100.
        RH = np.array(np.delete(np.asarray(RHar),indexlist),float)/100
        t_ar = np.array(np.delete(np.asarray(t),indexlist),float)
        dt_ar = np.array(np.delete(np.asarray(dt),indexlist),float) 
        ind_i = np.array(np.delete(np.asarray(ind_idt), indexlist),int) 
        ind_j = np.array(np.delete(np.asarray(ind_jdt), indexlist),int)        
        
        if z_t[i] == 2.0:
            z1 = z_dt[i]
            z2 = z_t[i]
            t1 = t_ar
            t2 = t1 - dt_ar
        if z_t[i] != 2.0:
            z1 = 2.0
            z2 = z_t[i]
            t2 = t_ar
            t1 = dt_ar + t2
        rho = -0.005150475461058842*t2 + 1.296159304053393 #kg/m3
        p2 = p1 - rho*g0*z2 #Pa
        Q1 = 0.622*RH*sat_vap_p_ar("liq",t1)/p1
        Q2 = 0.622*RH*sat_vap_p_ar("liq",t2)/p2
        crit_ar = np.around(np.array(t1*(1 + 0.611*Q1)-(t2*(1 + 0.611*Q2)))/(z2 - z1),decimals = precision)
        critlist,indexlist = [],[]
        for k in range(len(crit_ar)):
            if -varGamma - tolerance <= crit_ar[k] <= -varGamma + tolerance:
                critlist.append(crit_ar[k])
                indexlist.append([ind_i[k],ind_j[k]])
        critlist_all.append([st_z[i], critlist,indexlist])   
    return critlist_all

def New_matrix_writer_main():
    """
    This function writes copies data points from a dataset with a lower temporal resolution to a matrix of a higher resolution. 
    """
    import os,shutil
    rh_file = "Moisture-data/Krångede A.csv"
    RH_m = RH_ar(rh_file)
    p_file = "Pressure-data/KrangedeA-Lufttrykk.csv"
    p_m = RH_ar(p_file)
    path_n = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    path_m = "Matrices_n"
    if os.path.isdir(path_m):
        shutil.rmtree(path_m)
    os.mkdir(path_m)
    path_mp = path_m + "/" + "pmatrices"
    os.mkdir(path_mp)
    
    path_mrh = path_m + "/"+ "rhmatrices"
    os.mkdir(path_mrh)
    
    criterion = 9
    icefiles = []
    for filename in os.listdir(path_n):
        if filename.find('ICE') >= 0:                         
            icefiles.append([filename[0:4],filename])
    for file in icefiles:
        filename_p = path_mp + "/" + "%s-pmatrix.txt" % file[0]
        filename_rh = path_mrh + "/" + "%s-rhmatrix.txt" % file[0]
        cols = dataset(path_n + file[1], criterion)[0]
        c_d = cols['Date Field']
        c_t = cols['Time Field']
        RH_d = RH_m['Datum']
        RH_t = RH_m['Tid (UTC)']
        RH = RH_m['Relativ Luftfuktighet']
        p_d = p_m['Datum']
        p_t = p_m['Tid (UTC)']
        p = p_m['Lufttryck reducerat havsytans nivå']
        new_matrix_write_loop(filename_p, c_d, c_t, RH_d, RH_t, RH)
        new_matrix_write_loop(filename_rh, c_d, c_t, p_d, p_t, p)

def new_matrix_write_loop(filename,c_date,c_time,var_date,var_time, var_val):
    import csv,datetime
    with open(filename,'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter = ',')
        counter = 0
        for j in np.arange(1,len(var_date)):
            for i in np.arange(counter, len(c_date)):
                c_date_s = stringtodate(c_date[i])
                c_time_s = stringtotime(c_time[i])
                c_d_comb = datetime.datetime.combine(c_date_s,c_time_s)
                var_date_s0 = stringtodate(var_date[j-1])
                var_time_s0 = stringtotime(var_time[j-1])
                var_d_comb0 = datetime.datetime.combine(var_date_s0,var_time_s0)
                var_date_s1 = stringtodate(var_date[j])
                var_time_s1 = stringtotime(var_time[j])
                var_d_comb1 = datetime.datetime.combine(var_date_s1, var_time_s1)
                if var_d_comb0 <= c_d_comb <= var_d_comb1:
                    csv_writer.writerow([c_date[i], c_time[i], var_val[j]])
                    counter +=1
                else:
                    break
        print filename, "length: variabel:", len(c_date), "Number of rows in file: ", counter

def Qplots(date_m):
    import matplotlib.pyplot as plt
    import os, matplotlib
    
    matplotlib.rc('font', **{'family' : "sans-serif", 'sans-serif': 'Arial'})
    params = {'text.latex.preamble' : [r'\usepackage{amsmath}'], 
            'axes.linewidth' : 0.6, 'xtick.labelsize' : 4, 'ytick.labelsize': 5, 'legend.framealpha': 0.7, 'legend.fontsize': 6, 'figure.titlesize': 7, 'grid.linewidth':1.0, 'font.size': 8}
    plt.rcParams.update(params)
    
    path = "../Figures"
    filename_p = "Matrices/pmatrix.txt"
    filename_t = "Pressure-data/tmatrix.txt"
    filename_rh = "Matrices/RH.txt"
    filename_plot = path + "/" + "KrangadeA-Qplot.png"
    if os.path.isfile(filename_plot):
        os.remove(filename_plot)
    ilist_p,jlist_p, p_a = matrix_reader(filename_p)
    ilist_t,jlist_t, t_a = matrix_reader(filename_t)
    ilist_rh, jlist_rh, rh_a = matrix_reader(filename_rh)
    indexlist = []
    for i in range(len(p_a)):
        if p_a[i] == '' or t_a[i] == '' or rh_a[i] == '':
            indexlist.append(i)
    p = np.array(np.delete(np.asarray(p_a),indexlist),float)*100
    t = np.array(np.delete(np.asarray(t_a),indexlist),float)
    RH = np.array(np.delete(np.asarray(rh_a),indexlist),float)/100
    il = np.array(np.delete(np.asarray(ilist_rh),indexlist),float)
    a = np.delete(date_m,indexlist)
    monthlist, ilist = [],[]
    for i,date in enumerate(a):
        pairlist = []
        if date[6]> a[i-1][6]:
            ilist.append(i)
            monthlist.append(date)
    date = np.delete(a,indexlist)
    fig = plt.figure()
    Q = (0.622*RH*sat_vap_p_ar("liq",t)/p)*1000
    plt.plot(Q, lw = 0.2,color = 'b')
    plt.xticks(ilist, monthlist, rotation = 90)
    plt.ylabel("Specific humidity (g/kg)",fontsize = 6)
    fig.set_size_inches(4,3)
    plt.title("The development of specific throughout the observation period", fontsize = 7, weight = 'semibold')  
    fig.tight_layout()
    fig.savefig(filename_plot, dpi = 1000)

def LapseCriteria_writer(date_m,path,topfiles,tolerance):
    import os,csv,shutil
    
    foldername = "Criteria-tol_%g" % tolerance
    if os.path.isdir(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    crit_all = Lapsecriteria(tolerance)
    for k,row in enumerate(crit_all):
        st_n, critlist, indexlist = row
        st_no, file = topfiles[k]
        if st_n == st_no:
    	    cols = dataset(path+file,9)[0]
    	    date = cols['Date Field']
    	    time = cols['Time Field']
            filename = "%s-lapsecriteria_tol_%g.txt" % (st_n,tolerance)
            with open(foldername + "/" + filename, 'wb') as outfile:
                csv_writer = csv.writer(outfile, delimiter = ' ')
                csv_writer.writerow(["This file is running: ", file])
                csv_writer.writerow(["Index", "Date Field", "Time Field",])
                for i in range(len(critlist)):
                    i_el, j_el = indexlist[i]
                    csv_writer.writerow([j_el, date[j_el], time[j_el], critlist[i]])
    	else:
    	    print "The files aren't synchronised."
    	    break
    return ""

def Lapse_stats(path, precision, tolerance):
    import csv,shutil,os
    import matplotlib.pyplot as plt
    foldername = path + "/" + "Criteria-statistics_tol_%g" % tolerance

    os.mkdir(foldername)
    plotfolder = foldername + "/" + "Plots"
    histfolder = foldername + "/" + "Histograms"
    os.mkdir(plotfolder)
    os.mkdir(histfolder)
    crit_all = Lapsecriteria(precision, tolerance)
    path_n = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    criterion = 9
    ice_list = [] 
    for filename in os.listdir(path_n):
        if filename.find('ICE') >= 0: 
            ice_list.append(filename)

    for k, row in enumerate(crit_all):
        st_n, critlist, indexlist = row
        cols = dataset(path_n + ice_list[k], criterion)[0]
        date = cols['Date Field']
        time = cols['Time Field']
        dir_list,speedlist,speed_std = [],[],[]
        for name in cols.dtype.names:
            if name.find('Dir') >= 0 and name.find('A_Avg') >= 0:
                dir_list.append(name)
            elif name.find('Ano') >= 0 and name.find('A_Avg') >= 0:
                speedlist.append(name)
            elif name.find('Ano') >= 0 and name.find('A_Std') >= 0:
                speed_std.append(name)
        if len(dir_list) != 1:
            del dir_list[1]
        dir = cols[dir_list[0]]
        speed = cols[speedlist[0]]
        speed_st = cols[speed_std[0]]
        if len(critlist) != 0: 
            filename = foldername + "/" + "%s-lapsecriteria_tol_%g.txt" % (st_n, tolerance)
            print filename
            plotfilename = plotfolder + "/" + "%s-lapsecriteria_plot_tol_%g.png" % (st_n,tolerance)
            histfilename = histfolder + "/" + "%s-lapsecriteria_hist_tol_%g.png" % (st_n,tolerance)
            #Lapseplot(critlist, plotfilename, tolerance)
            #Lapsehist(critlist, histfilename, tolerance)
            Lapse_statwriter(st_n, indexlist, critlist,dir,speed,speed_st,filename,precision)
    if len(os.listdir(foldername)) == 2:
        shutil.rmtree(foldername)
                
def Lapse_statwriter(st_n, indexlist,critlist,dir,speed,speed_st,filename, precision):
    import csv
    with open(filename, 'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter = ',')
        csv_writer.writerow(["For station: ", st_n])
        csv_writer.writerow(["With a precision of ", precision , " decimals."])
        csv_writer.writerow(["Standard deviation: ", np.std(critlist)])
        csv_writer.writerow(["Average value: ", np.average(critlist)])
        csv_writer.writerow(["Range: ", max(critlist) - min(critlist)])
        csv_writer.writerow(["Minimum value: ", min(critlist)])
        csv_writer.writerow(["Maximum value: ", max(critlist)])
        csv_writer.writerow(["Number of elements: ", len(critlist)])
        csv_writer.writerow(["Var: ", np.var(critlist)])
        csv_writer.writerow(["              "])
        csv_writer.writerow(["i-index", "j-index", "s", "Direction", "Wind Speed", "Wind Speed (Std)"])
        for i in range(len(critlist)):
            i_el, j_el = indexlist[i]
            if j_el <= len(speed):
                csv_writer.writerow([i_el, j_el, critlist[i],dir[j_el], speed[j_el],speed_st[j_el]]) 
            else:
                csv_writer.writerow(["Element: ", j_el , "has been filtered for ice"])
                continue
    return ""
    
def Lapseplot(crit_ar, filename, tolerance):
    import matplotlib.pyplot as plt
    import matplotlib, os,csv

    matplotlib.rc('text', usetex = True)
    matplotlib.rc('font', **{'family' : "sans-serif"})
    params = {'text.latex.preamble' : [r'\usepackage{amsmath}'], 'axes.linewidth' : 0.6}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(1)
    
    cp = 1003.78597374 # J/kgK
    g0 = 9.80665 # m/s2
    varGamma = g0/cp # K/m
    
    titlefont = {'fontname': 'Arial', 'size': '8', 'weight':'bold'}    
    axisfont = {'fontname': 'Arial', 'size': '7'}
    crit = np.asarray(crit_ar,float)/-varGamma
    line1 = ax.plot(crit, ls = 'None', color = 'yellow', marker = 'o', markersize = 2)


    ax.set_title("The distribution of the static stability \n paramater with a tolerance of %g"%tolerance, **titlefont)
    ax.set_xlabel("Sample number", **axisfont)
    ax.set_ylim([0,1.5])
    ax.set_ylabel("Static stability paramater", **axisfont)
    leg = ax.legend(["Static stability parameter"], fontsize = 6, loc = 'best')
    leg.get_frame().set_alpha(0.6)
    leg.get_frame().set_linewidth(0.2)
    
    fig.set_size_inches(4,3)

    fontProperties = {'family':'Arial', 'weight': 'normal', 'size': 6}
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    ax.set_xticklabels(ax.get_xticks(), fontProperties)
#    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.tight_layout()
    fig.savefig(filename, dpi = 1000)

def Lapsehist(crit_ar,filename, tolerance): 
    """
    This function makes histogram of data of relative humidity. 
    The code is inspired from this website: 
    http://stackoverflow.com/questions/5328556/histogram-matplotlib
    """  
    import os,sys
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import MaxNLocator


    params = {'axes.linewidth' : 0.6 , 'font.family' : 'Sans-serif', 'font.sans-serif': 'Arial', 'xtick.labelsize': 6, 'ytick.labelsize': 6}
    plt.rcParams.update(params)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    titlefont = {'size': '8', 'weight':'semibold'}
#    subtitlefont = {'fontname': 'Arial', 'size': '', 'weight':'normal'}
    axisfont = {'size': '7'}
#    subfont = {'fontname': 'Arial', 'size': '7', 'weight': 'semibold'}

    cp = 1003.78597374 # J/kgK
    g0 = 9.80665 # m/s2
    varGamma = g0/cp # K/m
    varGamma_norm = np.asarray(crit_ar,float)/(-varGamma)

    n, bin = np.histogram(varGamma_norm)
    facecolor='green' 
    alpha=0.7
    ind = np.arange(len(n))

    bar1 = ax.bar(bin[:-1],n, width = 0.000225, linewidth = 0, color = "green",align = 'center')
    ax.set_xlabel("Sample count",**axisfont)
    ax.set_ylabel("Normalized static stability parameter", **axisfont)
    ax.set_title(u"The distribution of the static stability parameters", **titlefont)
    ax.set_xlim(min(bin),max(bin))
    fontProperties = {'family':'Arial', 'weight': 'normal', 'size': 6}
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    ax.set_xticklabels(ax.get_xticks(), fontProperties)
#    xTicks = ax.set_xticklabels(bin[-1])
#    plt.setp(xTicks, rotation = 0, fontsize = 4)
#    ax.set_yticklabels(get_yticks(),fontProperties)
#    ax.set_xlim(ind[0]-0.5*width, ind[-1]+1.5*width)
    
#    plt.xticks(ind + 0.5*width, bin) 
#    plt.yticks(ax.get_yticks(), fontProperties)
    
    fig.set_size_inches(4, 3)
    fig.tight_layout()
    fig.savefig(filename,dpi=1000)

    return ""

def stats():
    """
    This function initiates generation of results of neutral atmospheric stability.
    """
    import os,shutil
    main_path = "Criteria-statistics"
#    tol_steps = np.append(np.array((0)), np.linspace(10**-6, 10**-3, 10))
    precision = [4,5,6,7]
    tol_steps = np.linspace(0,4*10**-4,9)
    for el in precision:
        subfolder = main_path + "/" + "Criteria-prec_%d" % el
        if os.path.isdir(subfolder):
            shutil.rmtree(subfolder)
        os.mkdir(subfolder)
        for tolerance in tol_steps:
            Lapse_stats(subfolder,el,tolerance)
    return ""

def stats_reader(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()
    del lines[0:11]
    ilist, jlist, s, wd, ws, ws_std = [],[],[],[],[],[]
    for line in lines: 
        if line.find('has been filtered for ice') < 0:
            row = line.strip('\r\n').split(',')
            ilist.append(eval(row[0]))
            jlist.append(eval(row[1]))
            s.append(eval(row[2]))
            wd.append(eval(row[3]))
            ws.append(eval(row[4]))
            ws_std.append(eval(row[5]))
    return ilist, jlist, s, wd, ws, ws_std

def stats_comb(path, foldername, tolerance):
    import os,csv, datetime
    for file1 in os.listdir(path):
        if os.path.isdir(file1):
            break
        elif file1.endswith('txt'):
            st_no1 = file1[0:4]
            ilist1, jlist1, s1, wd1, ws1, ws_std1 = stats_reader(path + file1)
            #dataset(filename, criterion,
            for file2 in os.listdir(path):
                if os.path.isdir(file2):
                    break
                elif file2.endswith('txt'):
                    st_no2 = file2[0:4]
                    comblist = []
                    if st_no1 != st_no2:
                        date = datetime.datetime.today()
                        filename = foldername + "/" + "%s-%s-stat-comb.txt" % (st_no1,st_no2)
                        filename2 = foldername + "/" + "%s-%s-stat-comb.txt" % (st_no2,st_no1)
                        if os.path.isfile(filename2):
                            continue
                        ilist2, jlist2, s2, wd2, ws2, ws_std2 = stats_reader(path + file2)
                        std = combanalyzer(st_no1,st_no2)[1][2]
                        dist = combanalyzer(st_no1,st_no2)[1][6]
                        slope = combanayzer(st_no1,stno2)[1][11]
                        rows = []
                        for i in range(len(wd1)):
                            for j in range(len(wd2)):
                                if abs(wd1[i] - wd2[j]) <= tolerance and ws1[i] >= 5 and ws2[j] >= 5.:
                                    a = [ilist1[i], jlist1[i], round(s1[i],6), wd1[i], ws1[i], ws_std1[i]]
                                    b = [ilist2[j], jlist2[j], round(s2[j],6), wd2[j], ws2[j], ws_std2[j]]
                                    a.extend(b)
                                    rows.append(a)
                        if len(rows) != 0:
                            with open(filename,'wb') as outfile:
                                csv_writer = csv.writer(outfile, delimiter = ',')
                                csv_writer.writerow(["Date and time: ", date])
                                csv_writer.writerow(["The number of combinations are ", len(rows)]) 
                                h1 = ["i-index-1",  "j-index-1", "s-1",  "wd-1",  "ws-1", "ws_std1"]
                                h2  = ["i-index-2", "j-index-2", "s-2", "wd-2", "ws-2", "ws_std2"]
                                h1.extend(h2)
                                csv_writer.writerow(h1)
                                csv_writer.writerow(["Standard deviation: ", std])
                                csv_writer.writerow(["Slope parameter: ", slope])
                                csv_writer.writerow(["Distance: ", dist]) 
                                #Deleting duplicates
                                duplist = []
                                for k,line1 in enumerate(rows):
                                    inlist = []
                                    for l,line2 in enumerate(rows):
                                        if line1 == line2 and k != l:
                                            inlist.append(k)
                                    duplist.append(inlist)
                                for el in duplist:
                                    if len(el) > 1: 
                                        for i in el:
                                            del rows[i]
                                #Writing the rows
                                for row in rows:
                                    csv_writer.writerow(row)
    #Deleting an empty folder
    if len(os.listdir(foldername)) == 0:
        shutil.rmtree(foldername)

def combanalyzer(st_no1,st_no2):
    """
    This function returns the elevation statistics of a combination.
    The contents of this file:
    index    Parameter
    0 Date & Time
    1 Combination
    2 Standard deviation (height)
    3 Average value (height)
    4 Median (height)
    5 Range (height)
    6 Distance
    7 Minimum value(height)
    8 Maximum value(height)
    9 Number of elements: 
    10 Var(height)
    11 Flatness parameter 
    12 Standard Median deviation
    """
    import os
    foldername = "Elevation-statistics"
    comb1 = str(st_no1) + "-" + str(st_no2)
    comb2 = str(st_no2) + "-" + str(st_no1)
    for i, file in enumerate(os.listdir(foldername)):
        a = elevation_stats_reader(foldername + "/" + file)
        if a[1] == comb1 or a[1] == comb2:
            return a

def combfinder(M1,M2):
    import os
    path = "Criteria-comb/"
    os.chdir(path)
    counter = 0
    for dir in os.listdir(os.getcwd()):
        if os.path.isdir(path + dir):
            os.chdir(path + dir)
            for subdir in os.listdir(os.getcwd()):
                if os.path.isdir(os.getcwd() + "/" + subdir):
                    os.chdir(os.getcwd() + "/" + subdir)
                    for file in os.listdir(os.getcwd()):
                        if file.find(str(M1)) >= 0 and file.find(str(M2)) >= 0:
                            print os.getcwd() + "/" + file
                            counter += 1
                os.chdir(os.pardir)
        os.chdir(os.pardir)
   
    if counter == 0:
        print "There is no such combination of met. masts that met the requirements of neutral atmospheric stability."
    return ""

def windrose_plot_main():
    """
    This function initiates a windrose plot.
    """
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
	
def logprofile_plot_main():
    """
    This profile initiates a plot of the wind profile at a site.
    """
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

def logprofile_plot_main2():
    """
    This function initiates a plot of the wind profile wind profile at a site.
    """
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

def matrix_writer_main():
    """
    This function initiates the writing of synchronised matrices.
    """
    import datetime,os,shutil
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

    filename1 = "Pressure-data/tmatrix.txt"
    datas1 = "Pressure-data/KrangadeA-temp.csv"
    #matrix_writer(filename1,date_m,Set_content(RH_ar(datas1)))
    filename2 ="Pressure-data/pmatrix.txt"
    datas2 = "Pressure-data/KrangadeA-Lufttrykk.csv"
    #matrix_writer(filename2,date_m, Set_content(RH_ar(datas2)))
    path_n = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
    criterion = 9
    topfiles = []
    for filename in os.listdir(path_n):
        if filename.find('TOP') >= 0:                         
            topfiles.append([filename[0:4],filename])
    ice_F = []
    for filename in os.listdir(path_n):
        if filename.find('ICE') >= 0:
            ice_F.append(filename)
    #matrix_csv()
    
def elevation_plot_main():
    """
    This function initiates the plotting of the elevation profiles between the met masts.
    """
    import os,shutil
    foldername = "Elevation-Profiles_new/"
    if os.path.isdir(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    elevation_comb_loop(foldername)


def elevation_stats_main():
    """
    This function finds statistics of terrain.
    """
    import os,shutil
    foldername = "Elevation-statistics"
    if os.path.isdir(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    elevation_comb_loop(foldername)

def stat_comb_main():
    """
    This function initiates the writing of combination statistics.
    """
    import os, shutil
    path = "Criteria-statistics/Criteria-prec_4/"  
    parentfolder = "Criteria-comb"
    if os.path.isdir(parentfolder):
        shutil.rmtree(parentfolder)
    os.mkdir(parentfolder)
    for p in os.listdir(path):
        if p.startswith('.'):
            continue
        head_folder = parentfolder + "/" + "Criteria-comb_"+ p[p.find("tol_"):]
        if os.path.isdir(head_folder):
            shutil.rmtree(head_folder)
        os.mkdir(head_folder)
        for tol in range(0,21):
            foldername = head_folder + "/" + "Criteria-comb-tol_%d" % tol
            if os.path.isdir(foldername):
                shutil.rmtree(foldername)
            os.mkdir(foldername)
            stats_comb(path + p + "/",foldername, tol)

def date_matrix():
    import datetime,os,shutil,csv 
    startdate = datetime.date(2009,02,23)
    starttime = datetime.time(0,0,0)
    start = datetime.datetime.combine(startdate,starttime)
    enddate = datetime.date(2010,02,22)
    endtime = datetime.time(23,59,59)
    end = datetime.datetime.combine(enddate,endtime)
    date, time = [],[] 
    while start <= end:
        date.append(str(start)[0:10])
        time.append(str(start)[11:])
        start = start + datetime.timedelta(seconds=3600) 
    return np.array(zip(date,time), dtype = [('Date', 'S10'), ('Time','S8')])