import numpy as np

def dataset(filename, criterion):
	#setting __doc__
	"""
	Criterion is the number of elements.
	Add yes if you want more details.
	"""
#	import pprint
#	pp = pprint.PrettyPrinter(indent=4)
	infile = open(filename, 'r')		#open file for reading
	infile.readline()					#put every line as a dictionary element
	lines = []
	for line in infile:
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
	return cols,comments, heights, UTM_coor

def tablewriter(list_, nameoffile, path, dirs, criterion, comments = "no"):
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
	infile = open(filename, 'r')		#open file for reading
	infile.readline()					#put every line as a dictionary element
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
	with open(csv_name,"wb") as outfile:
		csv_writer = csv.writer(outfile)
		for element in list_:
			csv_writer.writerow(element)

def standarddev_writer(nameoffile, path, dirs, criterion):
	with open(nameoffile,"wb") as outfile:
		csv_writer = csv.writer(outfile,delimiter = ' ')
		for filename in dirs:
			csv_writer.writerow(["The following file is running " + filename])
			csv_writer.writerow(["			"])
			cols = list_[0]
			comments = list_[1]
			for name in cols.dtype.names:
				if type(cols[name][0]) == np.float64:
					standard = np.s
					csv_writer.writerow([np.average(cols[name])])	
		
def pixel(file,x,y):
    px = file.GetGeoTransform()[0]
    py = file.GetGeoTransform()[3]
    rx = file.GetGeoTransform()[1]
    ry = file.GetGeoTransform()[5]
    rasterx = int((x - px) / rx)
    rastery = int((y-py) / ry)
    return rasterx,rastery

def linestringsreader(filename):
	infile = open(filename)
	infile.readline()
	LSlist = []
	x1list,y1list,x2list,y2list = [],[],[],[]
	for line in infile:
		row = line.strip('\n').split(',')
		LSlist.append([eval(row[0]),eval(row[1]),eval(row[2]),eval(row[3])])
	return LSlist

def elevraster(linestring, gridresolution, file,data):
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
	import osgeo.gdal as gdal 
#	from gdalconst import *
	dataset = gdal.Open(filename, GA_ReadOnly)
	gt = dataset.GetGeoTransform()
	band = dataset.GetRasterBand(1)
	bandtype = gdal.GetDataTypeName(band.DataType)
	return gt[1], gt[5], dataset, \
	band.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize).astype(np.float) 

def plotter(path, funcxyz, angle, twolist,counter):
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
	ar1, ar2 = LSelement
	A = np.asarray(ar1,float) - np.asarray(ar2,float)
	return int(round(180 + np.arctan2(A[0],A[1])*(180/np.pi)))

def dirarray(angle):
	x,y = np.array((0,1))
	angle = (angle + 90)*(np.pi/180)
	xp = x*np.cos(angle) - y*np.sin(angle)
	yp = y*np.sin(angle) + y*np.cos(angle)
	return np.array((xp,yp))/np.linalg.norm(np.array(xp,yp))

def standard_dev(xlist,ylist):
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
  try:
    float(value)
    return True
  except ValueError:
    return False

def speedheight(namelist):
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
		
def logprofiledata(foldername, station_no, files, Nsectors = 16):
	import os
	subfolder = foldername + "/" + str(station_no)
	os.mkdir(subfolder)
	cols1 = dataset(path+files[0],criterion)[0]
	cols2 = dataset(path+files[1],criterion)[0]
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
			for i in range(len(x1N)):
			    logplotter(dirfoldername, i+1 , name1, name2, dataarray1['Direction'][i], dataarray2['Direction'][i], dataarray1['Anemometer'][i].T, dataarray2['Anemometer'][i].T, station_no, tolerance, x1N[i].T , y1N[i].T, x2N[i].T, y2N[i].T)		
	return ""

def xyNsectors(cols, Nsectors):
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
		directionlist
		while upper <= 360.:
			x_av , y_av, x_dir, y_dir,speedlist = [] , [] ,[] , [], []
			#Filtering for every speed column by direction
			for j in range(len(speedcolname)):
				#Searching through all rows in the dataset
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
	dataarray = np.array(zip(dircollist, sectorlist, directionlist, speedcollist, x_all, y_all), dtype = dt)
	return dircolname, dataarray

def logplotter(folder, counter, name1, name2, direction1, direction2, ano1, ano2, station_no, tolerance, x1 , y1, x2, y2):
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
	    ax.annotate(j+i+2, 
	                (x2[j],y2[j]), 
	                xytext = (24*np.cos(degrees*(np.pi/180))-12*np.sin(degrees*(np.pi/180)), 24*np.sin(degrees*(np.pi/180))+12*np.cos(degrees*(np.pi/180))), 
	                textcoords = 'offset points', 
	                arrowprops = dict(arrowstyle = '-', 
	                connectionstyle = 'arc3,rad=0'),
	                rotation = 0)
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
	fig.savefig(folder + "/" + "Sector_%d-%s-%s-%s.png" % (counter, station_no, direction1, direction2)) 
	plt.hold('off')
	return " "
 
def windrosecontour(ws,wd):
	from windrose import WindroseAxes
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
	ax = WindroseAxes.from_ax()
	ax.contourf(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot)
	ax.contour(wd, ws, bins=np.arange(0, 8, 1), colors='black')
	ax.set_legend()
	plt.show()

def windrosebar(ws,wd):
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
	import os
	os.path.dirname(path)
	dirs = os.listdir(path)
	tiflist = []
	for file in dirs:
		if file.endswith('tif'):
			tiflist.append(file)

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
	
logprof = "yes"

if logprof == "yes":
	import os,shutil
	path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
	dirs = os.listdir(path)
	criterion = 9
#	logfiles = []
#	for filename in dirs:
#		if filename.find('TOPFILT') >= 0: 
#			logfiles.append(filename)
	samefiles = []
	for i in range(len(dirs)-1):	
		if dirs[i+1][0:4]==dirs[i][0:4]:
			samefiles.append([dirs[i][0:4],[dirs[i],dirs[i+1]]])
	foldername = "Vertical-wind-profiles"
	if os.path.isdir(foldername):
		shutil.rmtree(foldername)
	os.mkdir(foldername)
	for elements in samefiles:
		station_no, files = elements
		logprofiledata(foldername, station_no, files)

#	logprofile(cols.dtype.names, cols, 270)
#	print logprofile(path)

#	import pprint
#	pp = pprint.PrettyPrinter(indent = 4)
#	pp.pprint(dataarray1)

"""
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