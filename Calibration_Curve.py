import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sys 
import pandas as pd 
import scipy as sp 




testfilepaths = [('/Users/charlesgordon/Desktop/Charles/062923/062923_6.1_10um.txt','/Users/charlesgordon/Desktop/Charles/062923/062923_1.2dyeblank.txt'),
                 ('/Users/charlesgordon/Desktop/Charles/062923/062923_5.1_1um.txt','/Users/charlesgordon/Desktop/Charles/062923/062923_1.2dyeblank.txt'),
                 ('/Users/charlesgordon/Desktop/Charles/062923/062923_4.1_500nm.txt','/Users/charlesgordon/Desktop/Charles/062923/062923_1.2dyeblank.txt'),
                 ('/Users/charlesgordon/Desktop/Charles/062923/062923_3.1_250nm.txt','/Users/charlesgordon/Desktop/Charles/062923/062923_1.2dyeblank.txt'),
                 ('/Users/charlesgordon/Desktop/Charles/062923/062923_2.1_100nm.txt','/Users/charlesgordon/Desktop/Charles/062923/062923_1.2dyeblank.txt')
                 ]


def integrate_peaks(testfilepaths, labels, blank=False): 
    peaklib = {}
    bottom = 450
    top = 600
    if len(testfilepaths) != len(labels): 
        return "Mismatch in length of labels and number of files"
    for index2, (testfile, blankfile) in enumerate(testfilepaths):
        raw_testdata = pd.read_csv(testfile, delimiter='\t')
        raw_blankdata = pd.read_csv(blankfile, delimiter='\t')
        testdatainrange = raw_testdata['Unnamed: 1'].iloc[(17+(bottom-300)):17+1+(top-300)].astype(float)
        blankdatainrange = raw_blankdata['Unnamed: 1'].iloc[(17+(bottom-300)):17+1+(top-300)].astype(float)                                      
        testsignal = np.asarray(testdatainrange.tolist())
        blanksignal = np.asarray(blankdatainrange.tolist())
        if blank:
            adjusted_signal = testsignal-blanksignal 
        else: 
            adjusted_signal = testsignal 
        print(f"Index: {index2}: Data: {adjusted_signal}")
        integration = 0
        integration = np.trapz(adjusted_signal) 
        peaklib.update({labels[index2]: integration})
    return peaklib



#Calls 
labels = ['10uM Dye', '1uM Dye', '500nM Dye', '250uM Dye', '100nM Dye']


#Peaklib handling 
peaklib = integrate_peaks(testfilepaths, labels)
concentrations = np.array([10000,1000,500,250,100])
integrations = np.array(list(peaklib.values()))
pandasdataframe = pd.DataFrame({"Concentration (nM)":concentrations, "Signal": integrations})



#Graphing 
sns.set_theme(style="whitegrid",palette="dark", context="paper")
p = sns.regplot(data=pandasdataframe,x="Concentration (nM)",y="Signal", order=1,ci=None)
slope, intercept, r, p, sterr = sp.stats.linregress(pandasdataframe['Concentration (nM)'], pandasdataframe['Signal'])
print(f"R^2: {r**2}, P: {p}")
print(f"Slope: {slope} Intercept: {intercept}")
ax = plt.gca()
at = matplotlib.offsetbox.AnchoredText(f"$R^2$ : {r**2}\n P : {p}", prop=dict(size=10), frameon=True, loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
plt.show()

#Slope: 1.1433566510725308 Intercept: 551.1003069581025 | INTEGRATED VALUE
#signal = 0.013881695815014064x + 3.7319249184166665 | PEAK VALUE


