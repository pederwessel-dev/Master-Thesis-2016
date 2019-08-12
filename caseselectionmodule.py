 # -*- coding: utf-8 -*-

import numpy as np
import datetime,os,sys
print datetime.datetime.today()
reload(sys)
sys.setdefaultencoding('utf-8')

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
	fieldnames = lines[9][0:3] #9 and 10 or 12 and 13
	
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
        
def new_matrix_reader(varname, st_no):
	"""
	This function reads the synchronized matrices of moisture, pressure, temperature, 
	temperature difference wind direction, wind speed, wind speed standard deviation and 
	the measure of stability parameter.
	"""
    import os
    if varname == "RH":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/rhmatrices"
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
        foldername2 = foldername1 + "Matrices_newest/pmatrices"
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
            
    if varname == "t":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/tmatrices"
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
        dt = [('Date', 'S10'),('Time', 'S8'),('t',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    if varname == "dt":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/dtmatrices"
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
        dt = [('Date', 'S10'),('Time', 'S8'),('dt',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    if varname == "wd":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/wdmatrices"
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
        dt = [('Date', 'S10'),('Time', 'S8'),('wd',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    if varname == "ws":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/wsmatrices"
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
        dt = [('Date', 'S10'),('Time', 'S8'),('ws',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    if varname == "ws_std":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/ws_stdmatrices"
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
        dt = [('Date', 'S10'),('Time', 'S8'),('ws_std',float)]
        return np.asarray(zip(date,time,var),dtype = dt)
    if varname == "s_par":
        foldername1 = "/Users/pederwessel/Documents/NMBU/Masteroppgave/Data-New-Format/"
        foldername2 = foldername1 + "Matrices_newest/s_parmatrices"
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
        dt = [('Date', 'S10'),('Time', 'S8'),('s_par',float)]
        return np.asarray(zip(date,time,var),dtype = dt)

    else: 
        return "Please choose between RH, P, dt,s_par, and specify the met mast."
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
	
def comb_RT90_array():
	"""
	This function reads the combinations assiociated with direction and geographical 
	position.
	"""
    infile = open("RT90-comb-dir.txt",'r')
    lines = infile.readlines()
    M1,la1,lo1,M2,la2,lo2,dir1,dir2 = [],[],[],[],[],[],[],[]
    for line in lines:
        row = line.strip('\r\n').split(' ')
        M1.append(row[0])
        lo1.append(row[1])
        la1.append(row[2])
        M2.append(row[3])
        lo2.append(row[4])
        la2.append(row[5])
        dir1.append(row[6])
        dir2.append(row[7])
    dt = np.dtype([('M1','i'), ('lo1',float),('la1',float),('M2','i'),('lo2',float),('la2',float),('Dir1',float),('Dir2',float)])
    return np.asarray(zip(M1,lo1,la1,M2,lo2,la2,dir1,dir2),dtype = dt)

	#The functions needed to read raster data.                   
def pixel(file,x,y):
    """
    This function returns the closest pixel-values of a coordinates in a os.geo-instance.
    Inspired by the lecture slides from the following website:
    http://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides4.pdf
    """ 
    xOrigin = file.GetGeoTransform()[0]
    yOrigin = file.GetGeoTransform()[3]
    pixelWidth = file.GetGeoTransform()[1]
    pixelHeight = file.GetGeoTransform()[5]
    rasterx = int(round((x - xOrigin) / pixelWidth))
    rastery = int(round((y - yOrigin) / pixelHeight))
    return rasterx,rastery

def elevraster(linestring, gridresolution, file,data):
    """
    This function returns the interpolated values along a line on a rasterized image. 
    """
    from shapely.geometry import LineString
    resx,resy = gridresolution
    line = LineString(linestring)
    length = line.length
    #diagonal = np.sqrt(resx**2+resy**2)
    diagonal = resx
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
        dist.append(distance/1000.)
    return x,y,z,dist,round(diagonal,2)
    
def transformer(filename):
    """
    This function reads the raster file.
    """
    import osgeo.gdal as gdal 
    from gdalconst import GA_ReadOnly
    dataset = gdal.Open(filename, GA_ReadOnly)
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    #bandtype = gdal.GetDataTypeName(band.DataType)
    array = band.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize).astype(np.float)
    if nodata is not None:
        array_n = np.zeros((np.shape(array)))
        for i in np.arange(np.shape(array)[0]):
            for j in np.arange(np.shape(array)[1]):
                if array[i,j] != nodata: 
                    array_n[i,j] = array[i,j]
                else:
                    array_n[i,j] = 0.0
        return gt[1], gt[5], dataset, array_n
    else:
        return gt[1], gt[5], dataset, array

def Static_para_array(st, z_t, z_dt, precision=5):
    """
    This function returns the array of the static stability parameter
    """    
    g0 = 9.80665 # m/s2
    p1 = pressure_reduced(st)*100.#Pa
    t = new_matrix_reader("t",st)['t']#degrees Celsius
    date = new_matrix_reader("t",st)['Date']
    time = new_matrix_reader("t",st)['Time']
    dt = new_matrix_reader("dt",st)['dt']#degrees celsius
    RH = new_matrix_reader("RH",st)['RH']/100. #mixing ratio
    if z_t == 2.0:
        z1 = z_t
        z2 = z_dt
        t1 = t
        t2 = t1 - dt
    if z_t != 2.0:
        z1 = 2.0
        z2 = z_t
        t2 = t
        t1 = dt + t2
    rho = -0.005150475461058842*t2 + 1.296159304053393 #kg/m3  
    p2 = p1 - rho*g0*z2 #Pa
    Q1 = 0.622*RH*sat_vap_p_ar("liq",t1)/p1
    Q2 = 0.622*RH*sat_vap_p_ar("liq",t2)/p2
    static_par = (t1*(1 + 0.611*Q1)-(t2*(1 + 0.611*Q2)))/(z1 - z2)
    return date,time, z1, z2, np.around(static_par,decimals = precision)

def CaseSelection_loop(path, stolerance = 0.1, wdtolerance = 5.):
	"""
	This function reads all the files to find a matching pair of combination.
	"""
    import datetime,csv
    infile = open('zlist.txt','r')
    lines = infile.readlines()
    st_z, z_t,z_dt = [],[],[]
    for line in lines:
        row = line.strip('\n').split(', ')
        st_z.append(eval(row[0]))
        z_t.append(eval(row[1]))
        z_dt.append(eval(row[2]))
    tif_filename_f = "Elevation/HEIGHT_XX_P_10_CLIP.tif"
    resx_f,resy_f,set_f, dataset_f = transformer(tif_filename_f)
    
    c = comb_RT90_array()
    for i in np.arange(len(c)):
        filename = path + "/" + "%d-%d-cases_stol_%.2f_wdtol_%.1f.txt" % (c['M1'][i],c['M2'][i], stolerance, wdtolerance)
        print filename
        print i+1, c['M1'][i], c['M2'][i]
        rf = [resx_f, resy_f]
        a  = [np.array((c['lo1'][i],c['la1'][i])), np.array((c['lo2'][i],c['la2'][i]))]
        x_f,y_f,z_f,dist_f, diagonal_f = elevraster(a,rf,set_f,dataset_f)
        dist = round(dist_f[-1] - dist_f[0],5)
        std = np.std(z_f)
        z_M1 = z_f[0]
        z_M2 = z_f[-1]
        
        s_d1 = new_matrix_reader("s_par_new",c['M1'][i])['Date']
        s_t1 = new_matrix_reader("s_par_new",c['M1'][i])['Time']
        s1 = new_matrix_reader("s_par_new",c['M1'][i])['s_par']
        wd1 = new_matrix_reader("wd",c['M1'][i])['wd']
        ws1 = new_matrix_reader("ws",c['M1'][i])['ws']
        
        s_d2 = new_matrix_reader("s_par_new",c['M2'][i])['Date']
        s_t2 = new_matrix_reader("s_par_new",c['M2'][i])['Time']
        s2 = new_matrix_reader("s_par_new",c['M2'][i])['s_par']
        wd2 = new_matrix_reader("wd",c['M2'][i])['wd']
        ws2 = new_matrix_reader("ws",c['M2'][i])['ws']

        #Setting boundaries
        g0 = 9.80665 # m/s2
        cp = 1003.78597374 #J/kgK
        varGamma = round(g0/cp,5)
        us = - varGamma + round(varGamma*stolerance,5)
        ls = - varGamma - round(varGamma*stolerance,5)
        u1 = c['Dir1'][i] + wdtolerance
        l1 = c['Dir1'][i] - wdtolerance
        u2 = c['Dir2'][i] + wdtolerance
        l2 = c['Dir2'][i] - wdtolerance
        
        counter = 0
        with open(filename,'wb') as outfile:
            cw = csv.writer(outfile, delimiter = ' ')
            cw.writerow([datetime.datetime.today()])
            cw.writerow(['Mast 1', 'Mast 2', 'Date (1)', 'Date (2)', 'Static stability parameter (1)', 'Static stability parameter (2)', 'Wind direction (tol_1)', 'Wind direction (tol_2)', 'Wind direction (1)', 'Wind direction (2)', 'Wind speed (1)', 'Wind speed (2)', 'Forest height (1)', 'Forest height (2)', 'Standard deviation (s)', 'Distance'])
            for j in np.arange(len(wd1)):
                for k in np.arange(j,len(wd2)):
                    if l1 <= wd1[j] <= u1 or l2 <= wd1[j] <= u2 or l1 <= wd2[k] <= u1 or l2 <= wd2[k] <= u2: 
                        if ls <= s1[j] <= us and ls <= s2[k] <= us:
                            c1 = datetime.datetime.combine(stringtodate(s_d1[j]), stringtotime(s_t1[j]))
                            c2 = datetime.datetime.combine(stringtodate(s_d2[k]), stringtotime(s_t2[k]))
                            if c1 == c2:
                                cw.writerow([c['M1'][i],c['M2'][i], c1, c2, s1[j], s2[k], wd1[j], wd2[k], c['Dir1'][i], c['Dir2'][i], ws1[j], ws2[k], z_M1, z_M2,std, dist])
                                counter+=1
                                break
                            break
                        break
            cw.writerow(['Total number of cases: ', counter])
        if counter == 0:
            os.remove(filename)
        
def CaseSelection_main():
	"""
	This function initiates the case selection loop.
	"""
    import os, shutil
    
    parentfolder = "Case selection_new"
    #if os.path.isdir(parentfolder):
    #    shutil.rmtree(parentfolder)
    #os.mkdir(parentfolder)
    
    s_tol = np.arange(0,11,5)/100.
    dir_tol = np.arange(0,16,2.5)
    for st in s_tol:
        for di in dir_tol:
            foldername = parentfolder + "/" + "CaseSelection_stol_%.2f_dirtol_%.1f" % (st,di)
            os.mkdir(foldername)
            counter = CaseSelection_loop(foldername, st, di)
            if len(os.listdir(foldername)) == 0:
                shutil.rmtree(foldername)