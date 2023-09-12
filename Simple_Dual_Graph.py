import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sys 
import pandas as pd 
import scipy as sp 
import time 

"Script for integrating and plotting simple comparison of fluorometer files " 


"""     Argument Section: 
        Testfilepaths: Takes list of tuples with absolute file paths for each fluorometer file 
        Labels: Takes list of strings that will form X-Axis
        Legend: Takes in list of strings that will form legend 
"""

labels1 = ['Buffer Blank','10mM HCl','IPA']



testfilepaths1 = [('/Users/charlesgordon/Desktop/RS_91123/091123/091123_BufferBlank.txt',''),
                 ('/Users/charlesgordon/Desktop/RS_91123/091123/091123_HCl_Test.txt',''),
                 ('/Users/charlesgordon/Desktop/RS_91123/091123/091123_IPA_Test.txt','')]



def integrate_peaks(testfilepaths, labels, blank=False): 
    peaklib = {}
    bottom = 450
    top = 600
    if len(testfilepaths) != len(labels): 
        return "Mismatch in length of labels and number of files"
    for index2, (testfile, blankfile) in enumerate(testfilepaths):
        raw_testdata = pd.read_csv(testfile, delimiter='\t')
        testdatainrange = raw_testdata['Unnamed: 1'].iloc[(17+(bottom-300)):17+1+(top-300)].astype(float)                                      
        testsignal = np.asarray(testdatainrange.tolist())
        adjusted_signal = testsignal 
        integration = np.trapz(adjusted_signal) 
        peaklib.update({labels[index2]: integration})
    return peaklib
                


testfilepaths1 = testfilepaths1
 

peaks1st = integrate_peaks(testfilepaths1, labels1)


test1x = np.array(list(peaks1st.keys()))
test1y = np.array(list(peaks1st.values()))


#Plotting Most Recent Pass 
fig, ax = plt.subplots(layout='constrained')
ax.scatter(test1x, test1y)
ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Integrated Signal', fontsize=12)
ax.set_title('Contamination Test', fontsize=15)

def sig2moles(x): 
    return (((x-551.1003069581025)/1.1433566510725308)*0.5)

def moles2sig(x): 
    return (1.1433566510725308*2*x + 551.1003069581025)

molar = ax.secondary_yaxis('right', functions=(sig2moles, moles2sig))
molar.set_ylabel('Molar Recovery (pm)', fontsize=12)


plt.grid()
plt.show()

#Plotting 
#Slope: 1.1433566510725308 Intercept: 551.1003069581025 | INTEGRATED VALUE  signal = 1.1433566510725308x + 551.1003069581025 

