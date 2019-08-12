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
	
	#Counting number of remaining days of measurements	
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
