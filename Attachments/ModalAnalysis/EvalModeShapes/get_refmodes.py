def get_refmodes():
        
        RefModePath = path + '/Scripts/FlutterAnalysis/ReferenceModes/'
        
        RefModes = {} 
        
        for file in os.listdir(RefModePath):
            if file.endswith('.csv'):
                with open(RefModePath + file ,'r') as f:
                    csvreader = csv.reader(f)
                    RefModes[str(file[:-4])] = [float(num[0]) for num in csvreader]
        return RefModes