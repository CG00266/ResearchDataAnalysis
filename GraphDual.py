import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sys 
import pandas as pd 
import scipy as sp 
import time 

"Script for integrating and plotting molar recovery of fluorometer files" 


"""     Argument Section: 
        Testfilepaths: Takes list of tuples with absolute file paths for each fluorometer file 
        Labels: Takes list of strings that will form X-Axis
        Legend: Takes in list of strings that will form legend 
"""

labels1 = ['Blank','Pass 1','Pass 2','Pass 3','Pass 4']
labels2 = ['Blank','Pass 1','Pass 2','Pass 3','Pass 4']
labels3 = ['Blank','Pass 1','Pass 2','Pass 3']

legend = ["Past Protocol", "Test 1: No Detergent", "Test 2: Different nitrogen supply"]

testfilepaths1 = [('/Users/charlesgordon/Downloads/Charles/072423/Analytival_Blank.txt',''),
                 ('/Users/charlesgordon/Downloads/Charles/072423/072423_Pass1_ReRead.txt',''),
                 ('/Users/charlesgordon/Downloads/Charles/072423/072423_Pass_2_ReRead.txt',''),
                 ('/Users/charlesgordon/Downloads/Charles/072423/072423_Pass_3_ReRead.txt',''),
                 ('/Users/charlesgordon/Downloads/Charles/072423/072423_Pass_4_ReRead.txt','')]


"08/17/23 Passes: Minor procedural modifications, no detergent "

testfilepaths2 = [('/Users/charlesgordon/Downloads/Charles/081723/081723_Analytical_Blank.txt','null'),
                 ('/Users/charlesgordon/Downloads/Charles/081723/081723_Pass1.txt','null'),
                 ('/Users/charlesgordon/Downloads/Charles/081723/081723_Pass2.txt','null'),
                 ('/Users/charlesgordon/Downloads/Charles/081723/081723_Pass3.txt', 'null'),
                 ('/Users/charlesgordon/Downloads/Charles/081723/081723_Pass4.txt','null')
                ]

"08/24/23 Passes: Different air supply" 

testfilepaths3 = [('/Users/charlesgordon/Downloads/Charles/082423/082423_Analytical_Blank.txt',"null"),
                  ('/Users/charlesgordon/Downloads/Charles/082423/082423_Pass1.txt',"null"),
                  ('/Users/charlesgordon/Downloads/Charles/082423/082423_Pass2.txt',"null"),
                  ('/Users/charlesgordon/Downloads/Charles/082423/082423_Pass3.txt',"null")
]



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
        integration = 0
        integration = np.trapz(adjusted_signal) 
        peaklib.update({labels[index2]: integration})
    return peaklib
                


testfilepaths1 = testfilepaths1
testfilepaths2 = testfilepaths2
testfilepaths3 = testfilepaths3 

peaks1st = integrate_peaks(testfilepaths1, labels1)
peaks2nd = integrate_peaks(testfilepaths2, labels2)
peaks3rd = integrate_peaks(testfilepaths3, labels3)

test1x = np.array(list(peaks1st.keys()))
test1y = np.array(list(peaks1st.values()))
test2x = np.array(list(peaks2nd.keys()))
test2y = np.array(list(peaks2nd.values()))
test3x = np.array(list(peaks3rd.keys()))
test3y = np.array(list(peaks3rd.values()))


#Plotting Most Recent Pass 
fig, ax = plt.subplots(layout='constrained')
ax.plot(test1x, test1y, label=legend[0])
ax.plot(test2x, test2y, label=legend[1])
ax.plot(test3x, test3y, label=legend[2])
ax.scatter(test1x, test1y)
ax.scatter(test2x, test2y)
ax.scatter(test3x, test3y)
ax.set_xlabel('Pass', fontsize=12)
ax.set_ylabel('Integrated Signal', fontsize=12)
ax.set_title('Procedural Blank Comparison', fontsize=15)

def sig2moles(x): 
    return (((x-551.1003069581025)/1.1433566510725308)*0.5)

def moles2sig(x): 
    return (1.1433566510725308*2*x + 551.1003069581025)

molar = ax.secondary_yaxis('right', functions=(sig2moles, moles2sig))
molar.set_ylabel('Molar Recovery (pm)', fontsize=12)


plt.grid()
plt.legend()
plt.show()

#Plotting 
#Slope: 1.1433566510725308 Intercept: 551.1003069581025 | INTEGRATED VALUE  signal = 1.1433566510725308x + 551.1003069581025 

