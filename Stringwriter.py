import numpy as np
from datawriter import heights


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
 	
filename_heights = "Dataheights-Ostersund_verified.txt"

filename = ('Combinations.txt')
comb_arr = combreader(filename)
heights = heightsreader(filename_heights)

from datawriter import csv_writer
import pprint
list_ =  comb_UTM_array(comb_arr, heights)

csv_name = "UTM-combination.csv"
#csv_writer(list_,csv_name)

def multilinestring(list_):
	endstring = ""
	for row in list_:
		line = "(%d %d,%d %d)," % (row[2],row[1],row[5],row[4]) 
		endstring+=line 
	return "MULTILINESTRING("+ endstring + ")"

import csv
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
linestrings(list_)