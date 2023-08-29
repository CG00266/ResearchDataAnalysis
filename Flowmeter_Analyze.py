import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import pandas as pd
import scipy 
import seaborn as sns 
import os 
import pathlib 
import math 


def graph_data(flowmeterdata, start=None, end=None, linewidth=.05, info=False):
    """Graph flowmeter data with matplotlib. Filename, optional start time period, end time period, and linewidth for graph
    >>> graph_data("/Users/charlesgordon/Downloads/Calibration_5ul")
    """
    
    #Initializing Data Bins
    with open(flowmeterdata, "r") as file: 
        data = pd.DataFrame(data=file).T
        index = 1
        time = np.array([])
        flow_rate = np.array([])
        current_time = 0 
        integration = 0

    #Getting values from flow meter data 
    if start or end: 
        index = 1 + start//.020
        while index < min(data.size, end//.020):
            current_column = data[index][0]
            current_time = current_time + .020
            current_flow = current_column.split()[1]
            time = np.concatenate((time, float(current_time)), axis=None)
            flow_rate = np.concatenate((flow_rate, float(current_flow)), axis=None)
            index += 1
            integration += .020*float(current_flow)
    else: 
        while index < data.size:
            current_column = data[index][0]
            current_time = current_time + 0.020
            current_flow = current_column.split()[1]
            time = np.concatenate((time, float(current_time)), axis=None)
            flow_rate = np.concatenate((flow_rate, float(current_flow)), axis=None)
            index += 1
            integration += .020*float(current_flow)

    
    #Graphing 
    #Options: Graph linewidth, start, and end adjustable 
    plt.plot(time, flow_rate, linewidth=linewidth) 
    plt.xlabel("Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (uL/min)")
    if start is not None or end is not None: 
        plt.axis([start,end,min(flow_rate),max(flow_rate)])
        
    #Data Analysis
    #Options: Displayed if info=True 
    if info: 
        print(f"Data mean: {np.mean(flow_rate)} +/- {np.std(flow_rate, ddof=1) / np.sqrt(np.size(flow_rate))}")
    plt.show()


def graph_directory(directory, start=False, end=False, linewidth=.05):
    """Plot up to 12 plots at once in the same figure:
        directory is a directory of flow files
        start is the start point for each file
        end is the end point for each file 
        """
    data = {}
    path = directory 
    for file in os.listdir(directory):
        file_path = f'{path}/{file}'
        #os.path.abspath(file) 
       #os.fsdecode(file) 

        with open(file_path, "r") as filetemp: 
            temp = pd.DataFrame(data=filetemp).T
            index = 1
            time = np.array([])
            flow_rate = np.array([])
            current_time = 0 
            integration = 0

    #Getting values from flow meter data 
        if start or end: 
            index = 1 + start 
            while index < min(temp.size, end):
                current_column = temp[index][0]
                current_time = current_time + .020
                current_flow = current_column.split()[1]
                time = np.concatenate((time, float(current_time)), axis=None)
                flow_rate = np.concatenate((flow_rate, float(current_flow)), axis=None)
                index += 1
                integration += .020*float(current_flow)
                #Trapezoidal integration, (previous reading + next reading)/2 * time 
        else: 
            while index < temp.size:
                current_column = temp[index][0]
                current_time = current_time + 0.020
                current_flow = current_column.split()[1]
                time = np.concatenate((time, float(current_time)), axis=None)
                flow_rate = np.concatenate((flow_rate, float(current_flow)), axis=None)
                index += 1
                integration += .020*float(current_flow)
        
        data[os.fsdecode(file)] = [time, flow_rate]

    #Graphing 
    #Do this algorithmically later if needed, for now just hardcoded 
    shapes = {2:[1,2],3:[1,3],4:[2,2],5:[1,5],6:[2,3],7:[1,7],8:[2,4],9:[3,3],10:[2,5],11:[1,11],12:[3,4]}
    shape = shapes[len(data.keys())]
    figure, axs = plt.subplots(shape[0],shape[1])
    row = 0
    column = 0  

    fig, axs = plt.subplots(shape[0],shape[1])
    index = 0
    titles = []
    for title in data.keys():
        titles.append(title)

    for ax, graph_data in zip(axs.flat,data.values()):
        ax.plot(graph_data[0],graph_data[1], linewidth=linewidth)
        ax.set_title(titles[index])
        index += 1

    plt.show()


    def superimpose(directory):
        pass 


    def plot_higherdimension(data,x,y,size,hue):
        """Plot higher dimensional data with seaborn relplot
        data1 = pd.read_csv('/Users/charlesgordon/Downloads/FastPump_FlowReadings.csv')
        data2 = pd.read_csv('/Users/charlesgordon/Downloads/FLow_Real3.csv')
        """
        sns.set_theme(style="whitegrid")
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        data = pd.read_csv(data)
        graph = sns.replot(data=data,x=x,y=y,size=size,hue=hue,palette=cmap)
        graph.despine(left=True, bottom=True)
        plt.show()


def integrate(flowmeterdata, ranges):
    """Take in flow meter csv file, and a list of lists with each being start and end range for integration. 
    Returns list of integration values in same order as list, using trapezoidal approximation
    """
    with open(flowmeterdata, "r") as file: 
        data = pd.DataFrame(data=file).T
        integrations = []

    #Getting values from flow meter data 
    for range in ranges: 
        integration = 0 
        index = 1 + range[0]//.020
        while index < min(data.size, (range[1]//.020)+1):
            current_row = data[index][0]
            next_row = data[index+1][0]
            current_flow = float(current_row.split()[1])
            next_flow = float(next_row.split()[1])
            integration += ((current_flow+next_flow)/2) * (.020/60)
            index += 1
        integrations.append(integration)
    for range, volume in zip(ranges,integrations):
        print(f'{range[0]}:{range[1]} | {volume} uL ')
    return integrations
        
    
    #Graphing 
    #Options: Graph linewidth, start, and end adjustable 
    plt.plot(time, flow_rate, linewidth=linewidth) 
    plt.xlabel("Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (uL/min)")
    if start is not None or end is not None: 
        plt.axis([start,end,min(flow_rate),max(flow_rate)])
        
    #Data Analysis
    #Options: Displayed if info=True 
    if info: 
        print(f"Data mean: {np.mean(flow_rate)} +/- {np.std(flow_rate, ddof=1) / np.sqrt(np.size(flow_rate))}")
        print("Integration: ", integration, "uL ")
    plt.show()


"""
def linear_regression(data, xlabel, ylabel, ylimits, xlimits):
    data=pd.read_csv(data)
    ax = sns.regplot(x="Pressure Head (psi) ",y="Flow Rate (uL/min)",data=data)
    if xlimits: 
        ax.set_ylim(0,30)
    if ylimits:
        ax.set_xlim(0,30)
    at = matplotlib.offsetbox.AnchoredText("$R^2$ = 0.991", prop=dict(size=5), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    

    g = sns.lmplot(x="Pressure Head (psi) ", y="Flow Rate (uL/min) ", hue="Destination ", col="Destination ",data=data)
    def annotate(data, **kws):
        r, p = sp.stats.pearsonr(data['Pressure Head (psi) '], data['Flow Rate (uL/min) '])
    ax = plt.gca()
    at = matplotlib.offsetbox.AnchoredText(f"$R^2$ : {sigfig.round(r,sigfits=3)}/n P : {sigfig.round(p,sigfigs=3)}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

g.map_dataframe(annotate)
plt.show()


def regplot_array():
#Plot multiple regression plots in one figure 
sns.set_theme(color_codes=True)
g = sns.lmplot(x="Pressure Head (psi) ", y="Flow Rate (uL/min) ", hue="Destination ", col="Destination ",
data=data, aspect=.4, x_jitter=.02, col_wrap=2, height=3)
def annotate(data, **kws):
    ax = plt.gca()
    r, p = sp.stats.pearsonr(data['Pressure Head (psi) '], data['Flow Rate (uL/min) '])
    at = matplotlib.offsetbox.AnchoredText(f"$R^2$ : {sigfig.round(r,sigfigs=3)}\n P : {sigfig.round(p,sigfigs=3)}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

g.map_dataframe(annotate)
g = ((g.set(ylim=[0,25],xlim=[0,xticks=[1.3,1.5,1.7,2])).fig.subplots_adjust(wspace=.04))
plt.show()




def annotate2(data, **kws):
    ax = plt.gca()
    std1_3, se1_3 = (data[''])
    sd1_5, se1_5 =
    sd1_7, se1_7 =
    sd2, se2 = 


data_CS1 = pd.read_csv('/Users/charlesgordon/Downloads/CS1.csv')
sns.set_theme(color_codes=True)
fig, ax= plt.subplots()
g = sns.regplot(data=data_R1E, x='Pressure Head (psi)', y='Flow Rate (uL/min)', x_jitter=.05, scatter=False, ax=ax)
x=np.array([1.3,1.5,1.7,2])


sns.scatterplot(x=x,y=mean_R1E,ax=ax)
ax.errorbar(x, mean_R1E, yerr=yerrd_R1E, fmt='none', capsize=5, zorder=1, color='C0')
ax.set_ylim(0,20)
plt.show()



g.map_dataframe(annotate)
g = ((g.set(ylim=[0,25],xticks=[1.3,1.5,1.7,2])).fig.subplots_adjust(wspace=.04))

"""



#Past Usage and tips for plotting

#0.05 good linewidth for macro plotting 
# plt.plot(time, flow_rate, linewidth=.2)
# plt.xlabel("Time (s)")
# plt.ylabel("Flow Rate (uL/min)")
#plt.axis([0,2475,0,45])
#plt.text(1850,40.6, "5uL Mean: 4.405 +/- 0.0143\n15uL Mean: 6.690 +/- 0.021\n25uL Mean: 20.585 +/- 0.080\n 35uL Mean: 28.566 +/- 0.068", fontsize = 5, bbox=dict(boxstyle="round",
#                   ec=(1., 0.5, 0.5),
#                   fc=(1., 0.8, 0.8),color="b"
#                   ))
#plt.show()

#0-1250 5uL/min
#1300-1650 15uL/min
#1750-1950 25uL/min
#2015-2450 35uL/min
#Alternative Regex method    
#current_time = re.match("-?(\d|.)*\\t", current_column).group()[0:-1]
#current_flow = re.match("-?(\d|.)*\\n", current_column).group()[0:-1]


# y = flow rate x = pressure hue = location 
# 2 plot pressures dependence of each with regression line 