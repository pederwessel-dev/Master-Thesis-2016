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
    
statistics_table()