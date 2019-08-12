import numpy as np
import sys, os, csv

#filename = "Datareader_example.txt"
#criterion = 9
#print dataset(filename,criterion,"yes")

def tablewriter(list_, nameoffile, path, dirs, criterion, comments = "no"):
	with open(nameoffile,"wb") as outfile:
		from datareader import dataset
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

def heightwriter():
	with open("Dataheights-Ostersund.csv","wb") as outfile:
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
	from numpy.lib.recfunctions import append_fields
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
#		for element in list_[0]:
#			csv_writer.writerow(element)	
#		csv_writer.writerow([' ','	'])
#		for element in list_[1]:
#			csv_writer.writerow(element)

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
	

#filename = 'Dataheights-Ostersund_verified.txt'
#list_ = combinatoric(filename)
#csv_name = "Combinations.csv"
#csv_writer(list_,csv_name)
import pprint
pp = pprint.PrettyPrinter(indent = 4)
"""
from datareader import dataset
path = "../../../NMBU/Masteroppgave-Data/Skogsvalidering WindSim SCA/wind data/TIL SSVAB/"
os.path.dirname(path)
dirs = os.listdir(path)
criterion = 9
nameoffile = 'Standarddeviations.csv'
#list_ = dataset(


I have to select two meteorological masts out of 14. Therefore there are: 
np.math.factorial(14)/np.math.factorial(12) = 182 combinations
"""

