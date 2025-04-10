import sys, os, select
import csv
import math
import pandas as pd
import numpy as np
import random
import string
import json
import seaborn as sns
import itertools
from scipy.spatial import distance
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker as tck
import matplotlib.patches as mpatches
from collections import Counter
from scipy.stats import pearsonr as corr
import colorsys
sys.path.insert(0, '/home/nik/BCPNNSim-olfaction/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/nik/BCPNNSim-olfaction/works/apps/olflang/'
buildPATH = '/home/nik/BCPNNSim-olfaction/works/build/apps/olflang/'
figPATH = '/home/nik/BCPNNSim-olfaction/works/apps/olflang/Figures/DualNet/Diagnostics/OrthoPats/'

kelly_colors = ['#BE0032', #Red
                '#F3C300', #Mustard
                '#8b4513', #saddlebrown
                '#F38400', #Peach
                '#A1CAF1', #Baby Blue
                '#C2B280', #Grunge greenish brown
                '#848482', #Grey
                '#008856', #Green
                '#E68FAC', #Pink
                '#0067A5', #Blue
                '#F99379', #Light Pink
                '#604E97', #Violet
                '#F6A600', #Orange
                '#B3446C', #Wine Stain red
                '#DCD300', #Slime yellowish green
                '#ff0000', #Maroon
                '#8DB600', #Parrot Green
                '#654522', #Brown
                '#E25822', #Dark Pink
                '#2B3D26', #Dark NAvy Blue
                '#F2F3F4', #Off White/Greyish White
                '#222222' #Black
                ]

def cosdist(pat,act) :

    bothpats = np.vstack((pat,act))

    return (1 - squareform(pdist(bothpats,metric='cosine')))[0,1]


def empatdist1(dorun = True,parfilename = "olflangmain1.par",figno = 1,clr = True,field = "act", r0 = 0,
               r1 = None) :


    if dorun : os.system("./olflangmain1")



    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))

    N = H * M

    if field=="inp" :

        data1 = Utils.loadbin("inp1.log",N)
        data2 = Utils.loadbin("inp2.log",N)

    elif field=="dsup" :

        data1 = Utils.loadbin("dsup1.log",N)
        data2 = Utils.loadbin("dsup2.log",N)

    elif field=="act" :

        data1 = Utils.loadbin("act1.log",N)  # N x 960 (number of simulation steps) array
        data2 = Utils.loadbin("act2.log",N)

    else : raise AssertionError("No such field")

    if len(data1)!=len(data2) : raise AssertionError("Illegal len(data1)!=len(data2)")

    if r1==None : r1 = max(len(data1),len(data2))

    trpats1 = Utils.loadbin("trpats1.bin",N)

    trpats2 = Utils.loadbin("trpats2.bin",N)

    trpats = trpats2 # np.concatenate((trpats1,trpats2))

    X1 = []

    for dat in data2 :

        x1 = []

        for trpat in trpats :

            x1.append(cosdist(trpat,dat))

        X1.append(x1)

    return X1



def convolve(data,H,M):
    N = H*M
    convolved_data = np.zeros(shape=(data.shape[0],M))
    for i in range(N):
        j = i%M
        convolved_data[:,j] += data[:,i]/H

    return convolved_data

def actplot(dorun = False,HCshow=20,MCmin = 540,showparams=False, parfilename = PATH+"olflangmain1.par",mode = "img",figno = 1,clr = True,
            field = "act",r0 = 0,r1 = None,c0 = 0,c1 = None,stride = 1) :

    
    os.chdir(buildPATH)
    if dorun :
        os.system("./olflangmain1")

    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    N =  H * M

    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))
    N2 =  H2 * M2


    igain = float(Utils.findparamval(parfilename,"igain"))
    again = float(Utils.findparamval(parfilename,"again"))
    taum= float(Utils.findparamval(parfilename,"taum"))
    taua= float(Utils.findparamval(parfilename,"taua"))
    adgain = float(Utils.findparamval(parfilename,"adgain"))


    taup = float(Utils.findparamval(parfilename,"taup"))
    taucond = float(Utils.findparamval(parfilename,"taucond"))

    recuwgain = float(Utils.findparamval(parfilename,"recuwgain"))
    assowgain = float(Utils.findparamval(parfilename,"assowgain"))
    bgain = float(Utils.findparamval(parfilename,"bgain"))

    etrnrep = int(Utils.findparamval(parfilename,"etrnrep"))

    nmean = float(Utils.findparamval(parfilename,"nmean"))
    namp = float(Utils.findparamval(parfilename,"namp"))

    #for printing important param at the right side of the image
    textstr = 'H = %d\nM = %d\n\nigain = %.2f\nagain = %.2f\ntaum = %.4f\ntaua = %.4f\nadgain = %.1f\n\ntaup = %.1f\ntaucond = %.3f\n\nrecuwgain = %.2f\nassowgain = %.2f\nbgain = %.2f\n\netrnrep = %d\n\nnmean = %.3f\nnamp = %.3f'%(H, M, igain, again, taum, taua, adgain, taup, taucond,recuwgain,assowgain,bgain,etrnrep, nmean, namp)

    text_file = open("simstages.txt", "r")
    simstages = [int(line) for line in text_file]
    text_file.close()



    if field=="inp" :

        data1 = Utils.loadbin("inp1.log",N)
        data2 = Utils.loadbin("inp2.log",N2)

    elif field=="dsup" :

        data1 = Utils.loadbin("dsup1.log",N)
        data2 = Utils.loadbin("dsup2.log",N2)

    elif field=="expdsup" :
        data1 = Utils.loadbin("expdsup1.log",N)
        data2 = Utils.loadbin("expdsup2.log",N2)

    elif field=="act" :

        data1 = Utils.loadbin("act1.log",N)
        data2 = Utils.loadbin("act2.log",N2)

    elif field=="ada" :

        data1 = Utils.loadbin("ada1.log",N)
        data2 = Utils.loadbin("ada2.log",N2)

    elif field == "bwsup" :

        data1 = Utils.loadbin("bwsup1.log",N)
        data2 = Utils.loadbin("bwsup2.log",N2)

    elif field == "expdsup" :

        data1 = Utils.loadbin("expdsup1.log",N)
        data2 = Utils.loadbin("expdsup2.log",N2)

    if HCshow>0:
        data1 = data1[:,0:HCshow*M]
        data2 = data2[:,0:HCshow*M2]

    else : raise AssertionError("Illegal field")




    if len(data1)!=len(data2) : raise AssertionError("Illegal len(data1)!=len(data2)")

    if r1==None : r1 = len(data1)

    if c1==None : c1 = max(len(data1[0]),len(data2[0]))

    if figno<1 : return

    if clr : plt.clf()

    data1 = data1.T
    data2 = data2.T

    

    if mode=="img" :

         ax1 = plt.subplot(2,1,1)
         colormap1 = ax1.imshow(data1[r0:r1, c0:],interpolation='none',aspect='auto',cmap = 'jet')
         plt.title("LTM #1",fontsize = 14)
         cbar = plt.colorbar(colormap1,ax=ax1)
         plt.xlabel("Timesteps",fontsize=16)
         plt.ylabel("MCs",fontsize=16)
         plt.xticks(fontsize = 16)
         plt.yticks(fontsize = 16)
         cbar.ax.tick_params(labelsize=14) 
    else :

        ax1.plot(data1[c0:c1:stride,r0:r1])


    if mode=="img" :
        ax2 = plt.subplot(2,1,2)
        colormap2 = ax2.imshow(data2[r0:r1,c0:],interpolation='none',aspect='auto',cmap = 'jet')
        plt.title("LTM #2", fontsize = 14)
        cbar = plt.colorbar(colormap2,ax=ax2)
        plt.xlabel("Timesteps",fontsize=16)
        plt.ylabel("MCs",fontsize=16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        cbar.ax.tick_params(labelsize=14) 
    else :

        ax2.plot(data2[c0:c1:stride,r0:r1])

    plt.subplots_adjust(hspace=0.3)
    plt.gcf().set_size_inches(20, 10)

    if showparams:
        plt.figtext(0.82,0.2, textstr, fontsize=14)

    plt.show()



def pos(MCpre,MCpos,N):
    return N*MCpre+MCpos





def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def datasize(filename) :

    if not os.path.isfile(filename) : return 0

    return os.path.getsize(filename)


def doplot(filename,C,ax,r0 = 0,r1 = None,c0 = 0,stride = 1) :

    if datasize(filename)>0 :
        data = Utils.loadbin(filename,C)
        if r1==None : r1 = len(data)
        ax.plot(data[r0:r1,0::stride])




def doimshow(filename,R,C,ax,idx = -1) :

    if datasize(filename)>0 :
        
        data = Utils.loadbin(filename,R*C)
        colormap1 = ax.imshow(data[0:R, 0:],interpolation='none',aspect='auto',cmap = 'jet')
        ax.imshow(data[idx].reshape(R,C),interpolation='none',aspect='auto',cmap = 'jet')
        plt.colorbar(colormap1,ax=ax)





def get_weights_over_pattern(wij, patterns):
    
    # 10, 90?
    within  = np.zeros((patterns.shape[0],patterns.shape[1]))
    between = np.zeros((patterns.shape[0], wij.shape[0]*wij.shape[0]))
        
    
    for p in range(patterns.shape[0]):
        
        # get non zero values
        pat  = patterns[p]
        indx = np.nonzero(pat)[0]
        c1, c2 = 0, 0
        for i in indx:
            for j in range(patterns.shape[1]):
                if j in indx: 
                    within[p,c1] = wij[i,j] 
                    c1 += 1
                else:
                    between[p,c2] = wij[i,j] 
                    c2 += 1
            
    return within[:,:c1], between[:,:c2]    

def get_assoc_w(wij, pats1, pats2):
    
    # 10, 90?
    associated  = np.zeros((pats1.shape[0],pats1.shape[1]))
    unassociated = np.zeros((pats1.shape[0], wij.shape[0]*wij.shape[0]))
        
    
    for p in range(pats1.shape[0]):
        
        # get non zero values
        pat1  = pats1[p]
        pat2 = pats2[p]
        pat1_indx = np.nonzero(pat1)[0]
        pat2_indx = np.nonzero(pat2)[0]
        c1, c2 = 0, 0
        for i in pat1_indx:
            for j in range(pats1.shape[1]):
                if j in pat2_indx: 

                    # print("pat1: {} pat1_idx: {} pat2_idx: {}, w: {}".format(p,i,j,wij[i,j]))
                    associated[p,c1] = wij[i,j] 
                    c1 += 1
                else:
                    unassociated[p,c2] = wij[i,j] 
                    c2 += 1
            
    return associated[:,:c1], unassociated[:,:c2]    







def plot_heats(d1,d2,lab1,lab2):
    '''
        plot heatmaps for 2 distance matrices to compare
    '''
    fig,ax = plt.subplots(1,2)
    sns.heatmap(d1,ax=ax[0],annot=True,annot_kws={"size": 8})
    sns.heatmap(d2,ax=ax[1],annot=True,annot_kws={"size": 8})
    ax[0].set_title(lab1)
    ax[1].set_title(lab2)
    plt.show()

def plot_dists(d1,d2,lab1,lab2):
    '''
        plot distributions fo 2 distance matrices to compare
    '''
    d1 = d1.to_numpy()[np.triu_indices_from(d1,k=1)]   
    d2 = d2.to_numpy()[np.triu_indices_from(d2,k=1)]   
    df = pd.DataFrame(zip(d1,d2),columns = [lab1,lab2])
    sns.displot(data=df,bins=50)
    plt.show()


def visualise_associations(savef=0):
    '''
        Visualise associations between the two networks
    '''
    #fname = PATH+"patstat_16od_16descs.txt"   ###Change when using different assoc file
    a = patstat #np.loadtxt(fname)

    if a.shape[1]==3:
        a = np.delete(a,2,1)

    ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']

    #descs for 0.5 thresh in extract_top_descs
    descs_en1 = ['Gasoline', 'Leather', 'Cinnamon', 'Mint', 'Banana', 'Lemon', 'Licorice',
         'Shoe.Polish', 'Terpentine', 'Pine.Needle', 'Garlic', 'Onion', 'Coffee',
         'Chocolate', 'Apple', 'Fruit', 'Dentist', 'Spice', 'Clove',
         'Perfume', 'Rose', 'Flower', 'Mushroom', 'Fish']

    #descs for 0.33 thresh
    descs2 = ['bensin', 'läder', 'tvål', 'parfym', 'kanel', 'mint', 'banan',
       'frukt', 'äpple', 'citron', 'lakrits', 'terpentin', 'skokräm',
       'tallbarr', 'vitlök', 'lök', 'kaffe', 'choklad', 'nejlika',
       'krydda', 'tandläkare', 'ros', 'blomma', 'svamp', 'fisk', 'sill',
       'skaldjur']

    descs_en2 = ['Gasoline','Leather','Soap','Perfume','Cinnamon','Mint','Banana','Fruit','Apple','Lemon','Licorice',
                'Turperntine','Shoe.Polish','Pine.Needle','Garlic','Onion','Coffee','Chocolate','Clove','Spice','Dentist',
                'Rose','Flower','Mushroom','Fish','Herring','Seafood']

    #si_4clusters
    descs_en3 = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
    'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
    'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']


    #si_2clusters
    descs_en4 = ['bensin', 'terpentin', 'skokräm', 'läder', 'tvål', 'parfym', 'krydda',
       'gummi', 'kemisk', 'kanel', 'vanilj', 'blomma', 'pepparmint', 'mentol',
       'mint', 'polkagris', 'banan', 'frukt', 'äpple', 'päron', 'citron',
       'apelsin', 'citrus', 'lakrits', 'anis', 'tallbarr', 'fernissa',
       'målarfärg', 'vitlök', 'lök', 'äcklig', 'kaffe', 'choklad', 'nejlika',
       'tandläkare', 'ananas', 'karamell', 'ros', 'svamp', 'champinjon',
       'fisk', 'sill', 'skaldjur', 'illa']

    ##correct_descs_maxassoc4
    descs_en5 = ['bensin', 'petroleum', 'bensinmack', 'diesel', 'läder', 
    'skokräm', 'skinn', 'kanel', 'kanelbulle', 'pepparmint', 'mint', 'mentol', 
    'banan', 'skumbanan', 'citron', 'citrus', 'lime', 'citronmeliss', 'lakrits', 'anis', 
    'salmiak', 'saltlakrits', 'terpentin', 'fernissa', 'målarfärg', 'lösningsmedel', 'vitlök', 
    'lök', 'stekt.lök', 'purjolök', 'kaffe', 'kaffesump', 'snabbkaffe', 'kaffeböna', 'äpple', 'tandläkare', 
    'nejlika', 'kryddnejlika', 'sjukhus', 'ananas', 'ros', 'rosenvatten', 'rosenolja', 'svamp', 'champinjon', 
    'kantarell', 'mögelsvamp', 'fisk', 'sill', 'skaldjur', 'räka']
    





#this is with the word heavy which show if some labels receive many associations. 
    assocs_dict = {}
    for key, value in a:
        if value not in assocs_dict:
            assocs_dict[int(value)] = [int(key)]
        else:
            assocs_dict[int(value)].append(int(key))

    print(assocs_dict)
    fig, ax = plt.subplots(figsize=(20, 8))
    gap = 1

    for i, (key, vals) in enumerate(assocs_dict.items()):
        x = np.mean(vals)
        indent = len(a[:, 1]) / 5
        
        # Position for the clouds
        top_y = 0  # Position for the bottom cloud (ODORS now at the bottom)
        bottom_y = 1  # Position for the top cloud (DESCRIPTORS now at the top)

        # Scatter points for the bottom cloud (ODORS)
        ax.scatter(key + indent, top_y, color=kelly_colors[key], s=200, alpha=0.6)
        ax.annotate(ODORS_en[key], xy=(key + indent - 0.3, top_y - 0.3), rotation=45, rotation_mode='anchor', size=14)  # Moved left

        for v in vals:
            # Scatter points for the top cloud (DESCRIPTORS)
            ax.scatter(v, bottom_y, color=kelly_colors[key], s=200, alpha=0.6)
            ax.plot([v, key + indent], [bottom_y, top_y], color=kelly_colors[key], alpha=0.6)
            ax.annotate(descs_en3[v], xy=(v - 0.3, bottom_y + 0.3), rotation=-45, rotation_mode='anchor', size=14)  # Moved left

    # Adjust annotations to reflect new positions
    ax.annotate('ODORS', xy=(a[:, 1].mean() + indent - 0.1, top_y - 0.8), weight='heavy', size=16, ha='center')  # Moved left
    ax.annotate('DESCRIPTORS', xy=(a[:, 1].mean() + indent - 0.1, bottom_y + 0.8), weight='heavy', size=16, ha='center')  # Moved left
    
    ax.set_ylim(-1, 2)
    ax.set_xlim(-1, (np.array(a).max() + 1))
    plt.axis('off')
    fig.tight_layout()
    plt.show()



#all bold now weighted to the number of associations
    assocs_dict = {}
    for key, value in a:
        if value not in assocs_dict:
            assocs_dict[int(value)] = [int(key)]
        else:
            assocs_dict[int(value)].append(int(key))

    print(assocs_dict)
    fig, ax = plt.subplots(figsize=(10, 14))
    gap = 1

    for i, (key, vals) in enumerate(assocs_dict.items()):
        x = np.mean(vals)
        indent = len(a[:, 1]) / 5
        
        # Position for the clouds
        top_y = 0  # Position for the bottom cloud (ODORS now at the bottom)
        bottom_y = 1  # Position for the top cloud (DESCRIPTORS now at the top)

        # Scatter points for the bottom cloud (ODORS)
        ax.scatter(key + indent, top_y, color=kelly_colors[key], s=200, alpha=0.6)
        ax.annotate(ODORS_en[key], xy=(key + indent - 0.3, top_y - 0.35), rotation=60, rotation_mode='anchor', size=14, weight='bold')  # Moved left

        for v in vals:
            # Scatter points for the top cloud (DESCRIPTORS)
            ax.scatter(v, bottom_y, color=kelly_colors[key], s=200, alpha=0.6)
            ax.plot([v, key + indent], [bottom_y, top_y], color=kelly_colors[key], alpha=0.6)
            ax.annotate(descs_en3[v], xy=(v - 0.4, bottom_y + 0.08), rotation=60, rotation_mode='anchor', size=14, weight='bold')  # Moved left

    # Adjust annotations to reflect new positions with bold weight
    ax.annotate('ODORS', xy=(a[:, 1].mean() + indent - 0.1, top_y - 0.8), size=16, ha='center', weight='bold')  # Moved left
    ax.annotate('DESCRIPTORS', xy=(a[:, 1].mean() + indent - 0.1, bottom_y + 0.8), size=16, ha='center', weight='bold')  # Moved left

    ax.set_ylim(-1, 2)
    ax.set_xlim(-1, (np.array(a).max() + 1))
    plt.axis('off')
    fig.tight_layout()
    plt.show()



def get_colors(c,assocs,delta=10):
    '''
        helper function to take hex color of odor, convert to hsv and change V by delta % to color associations
    ''' 
    rgb = Utils.hex2rgb(c)
    hsv = colorsys.rgb_to_hsv(rgb[0]/255,rgb[1]/255,rgb[2]/255)


    cols = [colorsys.hsv_to_rgb(hsv[0],hsv[1],hsv[2])]
    v = hsv[0]

    if assocs == 0:
        return([hsv[0],hsv[1],v-(delta*v/100)])

    for i in range(assocs):
        v -= delta*v/100
        # col = [hsv[0],hsv[1],v]
        col = [v,hsv[1],hsv[2]]
        col = list(colorsys.hsv_to_rgb(col[0],col[1],col[2]))
        
        cols.append(col)


    return cols

def plot_weights_assocwise(mode='group_by_nassocs'):
    '''
    plot weights by number of associations or for each odor
    '''
    parfilename=PATH+'olflangmain1.par'
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N1 = H1 * M1
    N2 = H2*M2



    if patstat.shape[1]==2:
        patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])
    else:
        patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2','trn_effort'])

    ############ REMEMBER TO CHANGE DESCS WHEN NEEDED
    descs_en = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
    'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
    'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']




    ODORS_en = np.array(['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish'])

    # descs_en = ODORS_en

    os.chdir('/home/nik/BCPNNSim-olfaction/works/build/apps/olflang/')
    trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)       
    trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N2)    




    w11 = Utils.loadbin(buildPATH+"Wij11pre_si_4clusters_1comma5dampenFactor_overlap.bin",N1,N1)
    w22 = Utils.loadbin(buildPATH+"Wij22pre_si_4clusters_1comma5dampenFactor_overlap.bin",N2,N2)
    w21 = Utils.loadbin(buildPATH+"Wij21pre_si_4clusters_1comma5dampenFactor_overlap.bin",N2,N1)
    w12 = Utils.loadbin(buildPATH+"Wij12pre_si_4clusters_1comma5dampenFactor_overlap.bin",N1,N2)
    b22 = Utils.loadbin(buildPATH+"Bj22pre_si_4clusters_1comma5dampenFactor_overlap.bin")
    b21 = Utils.loadbin(buildPATH+"Bj21pre_si_4clusters_1comma5dampenFactor_overlap.bin")
    b11 = Utils.loadbin(buildPATH+"Bj11pre_si_4clusters_1comma5dampenFactor_overlap.bin")
    b12 = Utils.loadbin(buildPATH+"Bj12pre_si_4clusters_1comma5dampenFactor_overlap.bin")


    #w11 = Utils.loadbin(buildPATH+"Wij11.bin",N1,N1)
    #w22 = Utils.loadbin(buildPATH+"Wij22.bin",N2,N2)
    #w21 = Utils.loadbin(buildPATH+"Wij21.bin",N1,N2)
    #w12 = Utils.loadbin(buildPATH+"Wij12.bin",N2,N1)
    #b22 = Utils.loadbin(buildPATH+"Bj22.bin")
    #b21 = Utils.loadbin(buildPATH+"Bj21.bin")
    #b11 = Utils.loadbin(buildPATH+"Bj11.bin")
    #b12 = Utils.loadbin(buildPATH+"Bj12.bin")

    ###binarize graded inputs (until I find a better way to think of attractor weights for graded patterns)
    if np.any(((trpats2 >0) & (trpats2 <1))):
        trpats2 = np.where(trpats2>0.4,1,0)


    p1 = np.zeros(trpats1.shape[0]*H1).astype(int) 
    p2 = np.zeros(trpats2.shape[0]*H2).astype(int)

    for i in range(trpats1.shape[0]): 
        p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

    p1=p1.reshape(trpats1.shape[0],H1)
    p2=p2.reshape(trpats2.shape[0],H2)

    print(p1.shape)
    ndescs = patstat_pd.LTM1.max()+1
    nods = patstat_pd.LTM2.max()+1
    ####Get mean similarity of all descs and od patterns with other patterns
    p1_meansim = np.zeros(ndescs)
    for i in range(ndescs):
        curpat  = p1[i]
        for j in range(ndescs):
            if j!=i:
                p1_meansim[i] += np.count_nonzero(p1[i]==p1[j])/(ndescs-1)

    p2_meansim = np.zeros(nods)
    for i in range(nods):
        curpat  = p2[i]
        for j in range(nods):
            if j!=i:
                p2_meansim[i] += np.count_nonzero(p2[i]==p2[j])/(nods-1)



    if mode == 'assocw_group_by_nassocs':

        ltm1_nassocs = patstat_pd.LTM1.value_counts().sort_index()
        ltm2_nassocs = patstat_pd.LTM2.value_counts().sort_index()
        print(ltm1_nassocs.unique())
        print(ltm2_nassocs.unique())

        w12_nassocwise_attr_weights = [[] for i in range(ltm1_nassocs.max())]        
        w21_nassocwise_attr_weights = [[] for i in range(ltm2_nassocs.max())]
        for i,n in enumerate(ltm2_nassocs.unique()):
            print('\n\tOd->Lang Nassoc: {} \n'.format(n))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(ltm2_nassocs[ltm2_nassocs==n].index),['LTM1','LTM2']].to_numpy()

            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(desc,od))
                langpat = p1[desc]
                odpat = p2[od]

                for pre in odpat:
                    for post in langpat:
                                w21_nassocwise_attr_weights[n-1].append(w21[pre,post])

        for i,n in enumerate(ltm1_nassocs.unique()):
            print('\n\tLang->Od Nassoc: {} \n'.format(n))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM1.isin(ltm1_nassocs[ltm1_nassocs==n].index),['LTM1','LTM2']].to_numpy()

            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(desc,od))
                langpat = p1[desc]
                odpat = p2[od]

                for pre in langpat:
                    for post in odpat:
                            w12_nassocwise_attr_weights[n-1].append(w12[pre,post])


        plot_colors = ['royalblue','forestgreen','tomato','orange']
        bins1 = np.histogram([item for sublist in w12_nassocwise_attr_weights for item in sublist],bins=150)[1]
        bins2 = np.histogram([item for sublist in w21_nassocwise_attr_weights for item in sublist],bins=150)[1]

        fig,ax = plt.subplots(1,2,figsize=(6,8),sharey=True)
        # for i,n in enumerate(nassocs.unique()):  ##for i in range(nassocs.max()-1,-1,-1)
        #     print('len {} assoc_weights: {}'.format(n,len(nassocwise_attr_weights[i])))
        #     sns.histplot(nassocwise_attr_weights[i],stat='probability', kde='True', bins=bins, color=plot_colors[i],label='{} Associations'.format(n),ax=ax)
        # ax.legend()

        x1 = np.arange(ltm1_nassocs.max())
        x2 = np.arange(ltm2_nassocs.max())
        # labels1 = ['{} Associations'.format(i+1) for i in range(ltm1_nassocs.max())]
        # labels2 = ['{} Associations'.format(i+1) for i in range(ltm2_nassocs.max())]
        labels1 = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM1.isin(ltm1_nassocs[ltm1_nassocs==i+1].index),'LTM1'].unique().size) for i in range(ltm1_nassocs.max())]
        labels2 = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM2.isin(ltm2_nassocs[ltm2_nassocs==i+1].index),'LTM2'].unique().size) for i in range(ltm2_nassocs.max())]

        #sns.boxplot(data=w12_nassocwise_attr_weights,palette=plot_colors,ax=ax[0])
        #sns.boxplot(data=w21_nassocwise_attr_weights,palette=plot_colors,ax=ax[1])
        sns.violinplot(data=w12_nassocwise_attr_weights,palette=plot_colors,ax=ax[0])
        sns.violinplot(data=w21_nassocwise_attr_weights,palette=plot_colors,ax=ax[1])
        #ax=ax[0]
        #ax.hist(x=w12_nassocwise_attr_weights)
        #ax=ax[1]
        #ax.hist(x=w21_nassocwise_attr_weights)
        means1 = [np.mean(j) for j in w12_nassocwise_attr_weights]
        means2 = [np.mean(j) for j in w21_nassocwise_attr_weights]
        print ('mean of w12 is')
        print (means1) 
        print ('mean of w21 is')
        print (means2) 
        ax[0].scatter(x1,means1,marker='o',color='white',s=100)
        # for i in range(len(x1)):
        #     ax[0].text(x1[i],means1[i],means1[i])
        ax[0].set_xticks(x1)
        ax[0].set_xticklabels(labels1)

        #ax[1].scatter(x2,means2,marker='o',color='white',s=100)
        # for i in range(len(x2)):
        #     ax[1].text(x2[i],means2[i],means2[i])
        ax[1].set_xticks(x2)
        ax[1].set_xticklabels(labels2)

        ax[0].set_xlabel('Grouped by (Lang->Od Associations)',size=14)
        ax[1].set_xlabel('Grouped by (Od->Lang Associations)',size=14)
        ax[0].set_title('AssocWeights (Lang->Od)')
        ax[1].set_title('AssocWeights (Od->Lang)')
        ax[0].set_ylabel('Weights',size=14)
            
        plt.show()

    elif mode=='w21_group_by_odor':

        assocwise_attr_weights = [[] for i in range(16)]
        for i in range(16):
            print('\n\tOdor: {} \n'.format(i+1))
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            for desc,od in od_desc_pairs:
                langpat = p1[desc]
                odpat = p2[od]

                for pre in odpat:
                    for post in langpat:
                        assocwise_attr_weights[i].append(w21[pre,post])

        bins = np.histogram([item for sublist in assocwise_attr_weights for item in sublist],bins=150)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(16):
        #     sns.histplot(assocwise_attr_weights[i],stat='count', kde='True', bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax)

        sns.boxplot(data=assocwise_attr_weights,palette=kelly_colors,ax=ax)
        means = [np.mean(j) for j in assocwise_attr_weights]
        ax.scatter(np.arange(nods),means,marker='o',color='grey',s=50)
        ax.set_xticks(np.arange(nods))
        ax.set_xticklabels(ODORS_en,rotation=45)
            
        #ax.set_xlabel('Weights',size=14)
        ax.set_ylabel('Weight',size=14)
            
        plt.show()

    elif mode=='w21_group_by_indivassocs':

        assocwise_attr_weights = [[] for i in range(patstat_pd.shape[0])]
        counter = 0
        nassocs = patstat_pd.LTM2.value_counts().sort_index()

        for i in range(trpats2.shape[0]):
            print('\n\tOdor: {} \n'.format(i+1))
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            for desc,od in od_desc_pairs:
                langpat = p1[desc]
                odpat = p2[od]

                for pre in odpat:
                    for post in langpat:
                        assocwise_attr_weights[counter].append(w21[pre,post])
                counter+=1





        hist,bins = np.histogram([item for sublist in assocwise_attr_weights for item in sublist],bins=150)
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        counter = 0
        for od in range(trpats2.shape[0]):
            na = nassocs[od]
            od_assocs = patstat_pd.loc[patstat_pd.LTM2==od,'LTM1'].values
            col = get_colors(kelly_colors[od],na)
            # print(col)

            #####Need to fix labelling of distriubtions. Counter should not be used in descs_en labelling
            for i in range(na):
                print(od,od_assocs[i],ODORS_en[od],descs_en[od_assocs[i]])

                sns.histplot(assocwise_attr_weights[counter],stat='count', kde='True', bins=bins, color=col[i],label='{}-{}'.format(ODORS_en[od],descs_en[od_assocs[i]]),ax=ax)

                ax.axvline(np.mean(assocwise_attr_weights[counter]), color=col[i], linestyle='dashed', linewidth=1)
                ax.annotate('{}-{}'.format(ODORS_en[od],descs_en[od_assocs[i]]),xy=(np.mean(assocwise_attr_weights[counter]),1),xycoords=('data','axes fraction'),c=col[i],rotation=45,annotation_clip=False)
                counter+=1

            
        ax.set_xlabel('Weights',size=14)
        ax.set_ylabel('Count',size=14)
        fig.tight_layout()   
        plt.show()


    elif mode=='bias_group_by_nassocs':

        b2 = (b22+b12)/2
        b1 = (b21+b11)/2

        nassocs = patstat_pd.LTM2.value_counts().sort_index()
        nassocs_ltm1 = nassocs #patstat_pd.LTM1.value_counts().sort_index()

        ##Are all descs unique or there is sharing between odors
        if nassocs_ltm1.max()==1:
            uniquedesc_flag = 1
        else:
            uniquedesc_flag = 0

        if uniquedesc_flag == 1:
            nassocwise_attr_od_biases = [[] for i in range(nassocs.max())]
            nassocwise_attr_lang_biases = [[] for i in range(nassocs.max())]

            odors_added = []    ###To avoid repetitions in bias appending
            for i in range(nassocs.max()):
                print('\n\tNassoc: {} \n'.format(i+1))
                ### Get pairs of odor descriptors from odors that have the same no of assocs
                od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']].to_numpy()

                for desc,od in od_desc_pairs:
                    print('Desc: {}   Od: {}'.format(desc,od))
                    langpat = p1[desc]
                    odpat = p2[od]
                    if not od in odors_added:
                        print('Desc: {}   Od: {}'.format(desc,od))
                        nassocwise_attr_od_biases[i].extend(b2[odpat])
                        nassocwise_attr_lang_biases[i].extend(b1[langpat])
                        odors_added.append(od)
                    else:
                        nassocwise_attr_lang_biases[i].extend(b1[langpat])

        else:
            nassocwise_attr_od_biases = [[] for i in range(nassocs.max())]
            nassocwise_attr_lang_biases = [[] for i in range(nassocs_ltm1.max())]

            odors_added = []    ###To avoid repetitions in bias appending
            for i in range(nassocs.max()):
                print('\n\tNassoc: {} \n'.format(i+1))
                ### Get pairs of odor descriptors from odors that have the same no of assocs
                od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']].to_numpy()

                for desc,od in od_desc_pairs:
                    print('Desc: {}   Od: {}'.format(desc,od))
                    odpat = p2[od]
                    if not od in odors_added:
                        nassocwise_attr_od_biases[i].extend(b2[odpat])
                        odors_added.append(od)

            descs_added = []
            for i in range(nassocs_ltm1.max()):
                print('\n\tNassoc: {} \n'.format(i+1))
                ### Get pairs of odor descriptors from odors that have the same no of assocs
                od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs_ltm1[nassocs_ltm1==i+1].index),['LTM1','LTM2']].to_numpy()
                # print(od_desc_pairs.shape[0])
                for desc,od in od_desc_pairs:
                    print('Desc: {}   Od: {}'.format(descs_en[desc],ODORS_en[od]))
                    descpat = p1[desc]
                    if not desc in descs_added:
                        if len(np.where(b1[descpat]>-4)[0])>0 and i==0:
                            print(desc,od)
                        nassocwise_attr_lang_biases[i].extend(b1[descpat])
                        descs_added.append(desc)



        b1_flat = [item for sublist in nassocwise_attr_lang_biases for item in sublist]    
        b2_flat = [item for sublist in nassocwise_attr_od_biases for item in sublist]
        plot_colors = ['royalblue','forestgreen','tomato','orange']
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['royalblue','forestgreen','tomato','orange']) 
        #plot_colors = ['tomato','forestgreen','royalblue','gold']
        bins1 = np.histogram(b1_flat,bins=50)[1]
        bins2 = np.histogram(b2_flat,bins=50)[1]
        fig,ax = plt.subplots(1,2,figsize=(6,8),sharey=True)
        x1 = np.arange(nassocs_ltm1.max())
        x2 = np.arange(nassocs.max())
        # for i in range(nassocs.max()):
        #     print(patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']])
        labels_langnet = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM2.isin(nassocs_ltm1[nassocs_ltm1==i+1].index),'LTM1'].unique().size) for i in range(nassocs_ltm1.max())]
        labels_odnet = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),'LTM2'].unique().size) for i in range(nassocs.max())]
        # for i in range(nassocs.max()-1,-1,-1):
            # sns.histplot(nassocwise_attr_lang_biases[i],stat='probability', kde=False, bins=bins1, color=plot_colors[i],label='{} Associations'.format(i+1),ax=ax[0])
            # sns.histplot(nassocwise_attr_od_biases[i],stat='probability', kde=False, bins=bins2, color=plot_colors[i],label='{} Associations'.format(i+1),ax=ax[1])

        sns.violinplot(data=nassocwise_attr_lang_biases,ax=ax[0])
        sns.violinplot(data=nassocwise_attr_od_biases,ax=ax[1])
        #sns.boxplot(data=nassocwise_attr_lang_biases,ax=ax[0])
        #sns.boxplot(data=nassocwise_attr_od_biases,ax=ax[1])
        means1 = [np.mean(j) for j in nassocwise_attr_lang_biases]
        means2 = [np.mean(j) for j in nassocwise_attr_od_biases]

        print('LangBiases_Mean: ',means1,'\n OdBiases_mean: ',means2)
        print(np.mean([item for sublist in nassocwise_attr_od_biases for item in sublist]))
        #ax[0].scatter(x1,means1,marker='o',color='white',s=100) #this is for the means
        #ax[1].scatter(x2,means2,marker='o',color='white',s=100)
        ax[0].set_xticks(x1)
        ax[0].set_xticklabels(labels_langnet)
        ax[1].set_xticks(x2)
        ax[1].set_xticklabels(labels_odnet)

        # ax[0].legend()
        # ax[1].legend()     
        # ax[0].set_xlabel('Weights',size=14)
        ax[0].set_ylabel('Bias',size=14)
        # ax[0].set_ylim([-6,-2])
        # ax[1].set_xlabel('Bias',size=14)
        if not uniquedesc_flag:
            ax[0].set_xlabel('LTM1 (Lang) -> LTM2 (Od) Associations')
            ax[1].set_xlabel('LTM2 (Od) -> LTM1 (Lang) Associations')

        ax[0].set_title('Lang Net Biases')
        ax[1].set_title('Odor Net Biases')
        plt.show()

    elif mode=='bias_indiv_odors':


        b2 = (b22+b12)/2

        od_biases = [[] for i in range(nods)]

        odors_added = []    ###To avoid repetitions in bias appending
        for i in range(nods):
            print('Odor: {}'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            odpat = p2[i]
            od_biases[i].extend(b2[odpat])


        # od_biases = np.array(od_biases)
        biases_mean = [np.mean(x) for x in od_biases]
        biases_std = [np.mean(x) for x in od_biases] 


        print(biases_mean)
        print(p2_meansim)

        # r,pval = corr(biases_mean,p2_meansim)
        # print(r,pval)
        plot_colors = ['royalblue','forestgreen','tomato','orange']
        bias_idx = np.argsort(biases_mean)

        # fig,(ax) = plt.subplots(1,1,figsize=(18,10),sharex=True)
        # for i,od in enumerate(bias_idx):
        #     nassocs = patstat_pd.loc[patstat_pd.LTM2==od,'LTM1'].shape[0]
        #     #print(od,nassocs)
        #     ax.bar(i,biases_mean[od],yerr=biases_std[od],color=plot_colors[nassocs-1])

        # patches = []
        # for i in range(len(plot_colors)):
        #     patches.append(mpatches.Patch(color=plot_colors[i],label='{} Associations'.format(i+1)))



        # ax.set_xticks(np.arange(nods))
        # ax.set_xticklabels(np.array(ODORS_en)[bias_idx],rotation=45) 
        # ax.set_xlabel('Odors',size=14)
        # ax.set_ylabel('Bias',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p2_meansim[bias_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(nods))
        # ax2.set_xticklabels(np.array(ODORS_en)[bias_idx],rotation=45)  

 
        b2_flat = [item for sublist in od_biases for item in sublist]

        bins = np.histogram(b2_flat,bins=100)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(trpats2.shape[0]):
        #     sns.histplot(od_biases[i],stat='count', kde=True,bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax,alpha=0.7)

        sns.violinplot(data=od_biases,palette=kelly_colors,ax=ax)
        #x_min, x_max = ax.get_xlim()
        # for i in range(trpats2.shape[0]):
        #     xs = np.linspace(x_min, x_max, 200)
        #     shape, location, scale = stats.lognorm.fit(od_biases[i])
        #     ax.plot(xs, stats.lognorm.pdf(xs, s=shape, loc=location, scale=scale), color=kelly_colors[i], ls=':')
        ax.set_xticks(np.arange(nods))
        ax.set_xticklabels(ODORS_en,rotation=45)    
        ax.set_ylabel('Biases',size=14)
        
        for i in range(nods):
            print(max(od_biases[i]),min(od_biases[i]))


        plt.show()

    elif mode=='bias_indiv_descs':


        b1 =  (b11+b21)/2

        desc_biases = [[] for i in range(ndescs)]
        print(ndescs,nods)
        print(desc_biases[-1])
        for i in range(ndescs):
            # print('Desc: {}'.format(descs_en[i]))
            descpat = p1[i]
            if i == 15:
                #print(desc_biases[i])
                print(b1[descpat])
            desc_biases[i].extend(b1[descpat])

        biases_mean = [np.mean(x) for x in desc_biases]
        biases_std = [np.std(x) for x in desc_biases]

        print(desc_biases[0])

        r,pval = corr(biases_mean,p1_meansim)
        print(r,pval)

        bias_idx = np.argsort(biases_mean)

        b1_flat = [item for sublist in desc_biases for item in sublist]

        bins = np.histogram(b1_flat,bins=100)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))


        # for i in range(trpats1.shape[0]):
        #     if i>=len(kelly_colors):
        #         color = get_colors(kelly_colors[i%len(kelly_colors)],0)
        #     else:
        #         color = kelly_colors[i]
        #     sns.histplot(desc_biases[i],stat='count', kde=True,bins=bins, color=color,label=descs_en[i],ax=ax,alpha=0.7)
        
        sns.violinplot(data=desc_biases,palette=kelly_colors,ax=ax)
        ax.set_xticks(np.arange(ndescs))
        ax.set_xticklabels(descs_en,rotation=45)
        # ax.legend(bbox_to_anchor=(1,1.1))     
        # ax.set_xlabel('Biases',size=14)
        # ax.set_ylabel('Count',size=14)


        # fig,(ax,ax2) = plt.subplots(2,1,figsize=(15,8),sharex=True)

        # for i,desc in enumerate(bias_idx):
        #     associated_odors = patstat_pd.loc[patstat_pd.LTM1==desc,'LTM2']


        #     for j,od in enumerate(associated_odors.values):
        #         print(od,associated_odors.size)
        #         height = biases_mean[desc]/associated_odors.size
        #         if j==0: #bottom
        #             if associated_odors.size == 1:
        #                 ax.bar(i,biases_mean[desc],yerr=biases_std[desc],color=kelly_colors[od])
        #             else:
        #                 ax.bar(i,height,color=kelly_colors[od])
        #         elif j==associated_odors.size-1:
        #             ax.bar(i,height,yerr=biases_std[desc],bottom = j*height, color=kelly_colors[od])
        #         else:
        #             ax.bar(i,height,bottom = j*height, color=kelly_colors[od])

        # ##### create legend for odors and colors
        # patches = []
        # for i in range(nods):
        #     patches.append(mpatches.Patch(color=kelly_colors[i],label=ODORS_en[i]))


        # # ax2 = ax.twinx()
        # # ax2.plot(np.array(descs_en)[bias_idx],p1_meansim[bias_idx])

        # ax.set_xticks(np.arange(ndescs))
        # ax.set_xticklabels(np.array(descs_en)[bias_idx],rotation=45) 
        # ax.set_xlabel('Descriptors',size=14)
        # ax.set_ylabel('Bias',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p1_meansim[bias_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(ndescs))
        # ax2.set_xticklabels(np.array(descs_en)[bias_idx],rotation=45) 

        plt.show()

    elif mode=='biasdist_bothnets':

       b2 = (b22+b12)/2

       od_biases = [[] for i in range(nods)]

       odors_added = []    ###To avoid repetitions in bias appending
       for i in range(nods):
            print('Odor: {}'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            odpat = p2[i]
            od_biases[i].extend(b2[odpat])

       od_biases = np.array(od_biases)
       biases_std = od_biases.std(axis=1)
       b2_flat = [item for sublist in od_biases for item in sublist]

       bins2 = np.histogram(b2_flat,bins=50)[1]
       

       b1 =  (b11+b21)/2

       desc_biases = [[] for i in range(ndescs)]

       for i in range(ndescs):
            print('Desc: {}'.format(descs_en[i]))
            descpat = p1[i]
            desc_biases[i].extend(b1[descpat])

       desc_biases = np.array(desc_biases)
       biases_std = desc_biases.std(axis=1)
       b1_flat = [item for sublist in desc_biases for item in sublist]

       bins1 = np.histogram(b1_flat,bins=50)[1]  

       fig,ax = plt.subplots(2,1,figsize=(10,8),sharex=True) 
       sns.histplot(b1_flat,stat='count', kde=True,bins=bins1, color='tab:blue',ax=ax[0])
       sns.histplot(b2_flat,stat='count', kde=True,bins=bins2, color='tab:orange',ax=ax[1])
       ax[0].set_title('LTM1 Bias Distribution')
       ax[1].set_title('LTM2 Bias Distribution')
       ax[0].set_ylabel('Count')
       ax[1].set_ylabel('Count')
       ax[0].axvline(x=np.mean(b1_flat),linestyle='--',linewidth=1,color='grey')
       ax[1].axvline(x=np.mean(b2_flat),linestyle='--',linewidth=1,color='grey')
       xmin = np.min(b1_flat+b2_flat)-0.5
       xmax = np.max(b1_flat+b2_flat)+0.5
       ax[0].set_xlim([xmin,xmax])
       ax[1].set_xlim([xmin,xmax])
       plt.show()

    elif mode=='recuwdist_bothnets':

       w11_exc=[]
       w11_inh= [] 

       for pat in p1:
            for j in pat.astype(int):
                for k in range(trpats1.shape[1]): #Over every unit
                    if k in pat:
                        w11_exc.append(w11[j,k])
                    else:
                        w11_inh.append(w11[j,k])


       w22_exc=[]
       w22_inh= [] 

       for pat in p2:
            for j in pat.astype(int):
                for k in range(trpats2.shape[1]): #Over every unit
                    if k in pat:
                        w22_exc.append(w22[j,k])
                    else:
                        w22_inh.append(w22[j,k])


       w11_all = w11_exc+w11_inh
       bins1 = np.histogram(w11_all,bins=75)[1]
       
       w22_all = w22_exc+w22_inh
       bins2 = np.histogram(w22_all,bins=75)[1]
 

       fig,ax = plt.subplots(2,1,figsize=(10,8),sharex=True) 
       sns.histplot(w11_exc,stat='probability', kde=False,bins=bins1, color='tab:blue',ax=ax[0],label='Excitatory Connections')
       sns.histplot(w11_inh,stat='probability', kde=False,bins=bins1, color='tab:red',ax=ax[0],label='Inhibitory Connections')
       sns.histplot(w22_exc,stat='probability', kde=False,bins=bins2, color='tab:blue',ax=ax[1],label='Excitatory Connections')
       sns.histplot(w22_inh,stat='probability', kde=False,bins=bins2, color='tab:red',ax=ax[1],label='Inhibitory Connections')

       ax[0].set_title('LTM1 Recurrent Weight Distribution')
       ax[1].set_title('LTM2 Recurrent Weight Distribution')
       ax[0].set_ylabel('Density')
       ax[1].set_ylabel('Density')

       ax[0].axvline(x=np.mean(w11_exc),linestyle='--',linewidth=1,color='tab:blue')
       ax[0].axvline(x=np.mean(w11_inh),linestyle='--',linewidth=1,color='tab:red')
       ax[1].axvline(x=np.mean(w22_exc),linestyle='--',linewidth=1,color='tab:blue')
       ax[1].axvline(x=np.mean(w22_inh),linestyle='--',linewidth=1,color='tab:red')

       xmin = np.min(w11_all+w22_all)-0.5
       xmax = np.max(w11_all+w22_all)+0.5
       ax[0].set_xlim([xmin,xmax])
       ax[1].set_xlim([xmin,xmax])
       ax[0].set_ylim([0,1])
       ax[1].set_ylim([0,1])
       ax[0].legend(loc='upper left')
       ax[1].legend(loc='upper left')
       plt.show()


    elif mode=='recurrentw_odorwise':
        odorwise_recurrent_weights = [[] for i in range(nods)]

        for i in range(nods):
            print('\n\tOdor: {} \n'.format(i+1))
            odpat = p2[i]
            for pre in odpat:
                for post in odpat:
                        odorwise_recurrent_weights[i].append(w22[pre,post])


        odorwise_recurrent_weights = np.array(odorwise_recurrent_weights)
        recuw_mean = odorwise_recurrent_weights.mean(axis=1)
        recuw_std = odorwise_recurrent_weights.std(axis=1)
        recuw_idx = np.argsort(recuw_mean)


        # r,pval = corr(recuw_mean,p2_meansim)
        # print(r,pval)

        bins = np.histogram([item for sublist in odorwise_recurrent_weights for item in sublist],bins=150)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(trpats2.shape[0]):
        #     sns.histplot(odorwise_recurrent_weights[i],stat='count', kde='True', bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax)
        #     ax.axvline(np.mean(odorwise_recurrent_weights[i]),linestyle='--',color=kelly_colors[i])

        sns.violinplot(data=odorwise_recurrent_weights.T,palette=kelly_colors,label=ODORS_en,ax=ax)
        ax.set_xticks(np.arange(trpats2.shape[0]))
        ax.set_xticklabels(ODORS_en,rotation=45)
        # ax.set_xlabel('Weights',size=14)
        # ax.set_ylabel('Count',size=14)

        # fig,(ax) = plt.subplots(1,1,figsize=(15,10),sharex=True)
        # for i,od in enumerate(recuw_idx):
        #         ax.bar(i,recuw_mean[od], yerr=recuw_std[od], color=kelly_colors[od])


        # ##### create legend for odors and colors
        # patches = []
        # for i in range(nods):
        #     patches.append(mpatches.Patch(color=kelly_colors[i],label=ODORS_en[i]))



        # ax.set_xticks(np.arange(nods))
        # ax.set_xticklabels(np.array(ODORS_en)[recuw_idx],rotation=45) 
        # ax.set_xlabel('Odors',size=14)
        # ax.set_ylabel('Mean recurrent weight',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p2_meansim[recuw_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(nods))
        # ax2.set_xticklabels(np.array(ODORS_en)[recuw_idx],rotation=45)   
        plt.show()


    elif mode=='odornet_recurrentw_nassocwise':

        nassocs = patstat_pd.LTM2.value_counts().sort_index()

        nassocwise_recu_weights = [[] for i in range(nassocs.max())]
        for i in range(nassocs.max()):
            print('\n\tNassoc: {} \n'.format(i+1))
            ### Get odors with same no of associations
            nassoc_ods = nassocs[nassocs==i+1].index
            for od in nassoc_ods:
                odpat = p2[od]
                for pre in odpat:
                    for post in odpat:
                        nassocwise_recu_weights[i].append(w22[pre,post])




        plot_colors = ['tomato','forestgreen','royalblue','gold']
        bins = np.histogram([item for sublist in nassocwise_recu_weights for item in sublist],bins=150)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        for i in range(nassocs.max()-1,-1,-1):
            print('len {} assoc_weights: {}'.format(i+1,len(nassocwise_recu_weights[i])))
            sns.histplot(nassocwise_recu_weights[i],stat='probability', kde='True', bins=bins, color=plot_colors[i],label='{} Associations'.format(i+1),ax=ax)
            
        ax.legend()
            
        ax.set_xlabel('Weights',size=14)
        ax.set_ylabel('Count',size=14)
            
        plt.show()

    elif mode=='recurrentw_descwise':

        descwise_recurrent_weights = [[] for i in range(ndescs)]
        for i in range(ndescs):
            print('\nDesc: {} '.format(i+1))
            descpat = p1[i]
            for pre in descpat:
                for post in descpat:
                        descwise_recurrent_weights[i].append(w11[pre,post])

        # bins = np.histogram([item for sublist in odorwise_recurrent_weights for item in sublist],bins=150)[1]
        # fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(16):
        #     sns.histplot(odorwise_recurrent_weights[i],stat='count', kde='True', bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax)

        # ax.legend()
            
        # ax.set_xlabel('Weights',size=14)
        # ax.set_ylabel('Count',size=14)
            
        # plt.show()

        descwise_recurrent_weights = np.array(descwise_recurrent_weights)
        recuw_mean = descwise_recurrent_weights.mean(axis=1)
        recuw_std = descwise_recurrent_weights.std(axis=1)
        recuw_idx = np.argsort(recuw_mean)
        fig,(ax) = plt.subplots(1,1,figsize=(15,10),sharex=True)



        r,pval = corr(recuw_mean,p1_meansim)
        print(r,pval)

        ####BAR PLOT
        # for i,desc in enumerate(recuw_idx):
        #     associated_odors = patstat_pd.loc[patstat_pd.LTM1==desc,'LTM2']


        #     for j,od in enumerate(associated_odors.values):
        #         #print(od,associated_odors.size)
        #         height = recuw_mean[desc]/associated_odors.size
        #         if j==0: #bottom
        #             if associated_odors.size == 1:
        #                 ax.bar(i,recuw_mean[desc],yerr=recuw_std[desc],color=kelly_colors[od])
        #             else:
        #                 ax.bar(i,height,color=kelly_colors[od])
        #         elif j==associated_odors.size-1:
        #             ax.bar(i,height,yerr=recuw_std[desc],bottom = j*height, color=kelly_colors[od])
        #         else:
        #             ax.bar(i,height,bottom = j*height, color=kelly_colors[od])

        # ##### create legend for odors and colors
        # patches = []
        # for i in range(nods):
        #     patches.append(mpatches.Patch(color=kelly_colors[i],label=ODORS_en[i]))



        # ax.set_xticks(np.arange(ndescs))
        # ax.set_xticklabels(np.array(descs_en)[recuw_idx],rotation=45) 
        # ax.set_xlabel('Descriptors',size=14)
        # ax.set_ylabel('Mean recurrent weight',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p1_meansim[recuw_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(ndescs))
        # ax2.set_xticklabels(np.array(descs_en)[recuw_idx],rotation=45)   

        ####VIOLIN/BOX PLOT
        sns.boxplot(data=descwise_recurrent_weights.T,palette=kelly_colors,ax=ax)
        #ax2 =ax.twinx()
        ax.plot(np.arange(ndescs),p1_meansim,marker='o',color='tab:red',label='Mean Overlap')
        ax.legend()
        ax.set_xticks(np.arange(ndescs))
        ax.set_xticklabels(np.array(descs_en),rotation=45) 
        ax.set_xlabel('Descriptors',size=14)
        ax.set_ylabel('Weights',size=14)
        plt.show()
        
    elif mode=='recurrentw_descs_nassocwise':

        nassocs = patstat_pd.LTM2.value_counts().sort_index()
        nassocwise_recuw = [[] for i in range(nassocs.max())]

        odors_added = []    ###To avoid repetitions in bias appending
        for i in range(nassocs.max()):
            print('\n\tNassoc: {} \n'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']].to_numpy()

            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(desc,od))
                odpat = p2[od]
                if not od in odors_added:
                    nassocwise_attr_od_biases[i].extend(b2[odpat])
                    odors_added.append(od)

        descs_added = []
        for i in range(nassocs_ltm1.max()):
            print('\n\tNassoc: {} \n'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs_ltm1[nassocs_ltm1==i+1].index),['LTM1','LTM2']].to_numpy()
            # print(od_desc_pairs.shape[0])
            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(descs_en[desc],ODORS_en[od]))
                descpat = p1[desc]
                if not desc in descs_added:
                    if len(np.where(b1[descpat]>-4)[0])>0 and i==0:
                        print(desc,od)
                    nassocwise_attr_lang_biases[i].extend(b1[descpat])
                    descs_added.append(desc)


    elif mode=='assocw_particular_odor':

        for od in range(1):
            assocs = patstat_pd.loc[patstat_pd.LTM2==od,'LTM1'].values
            c = get_colors(kelly_colors[od],len(assocs))
            
            w21_assocwise = [[] for i in range(len(assocs))]
            for i,desc in enumerate(assocs):
                odpat = p2[od]
                descpat = p1[desc]
                for pre in odpat:
                    for post in descpat:
                        w21_assocwise[i].append(w21[pre,post])

            w12_assocwise = [[] for i in range(len(assocs))]
            for i,desc in enumerate(assocs):
                odpat = p2[od]
                descpat = p1[desc]
                for pre in descpat:
                    for post in odpat:
                        w12_assocwise[i].append(w12[pre,post])


            fig,(ax,ax2) = plt.subplots(1,2,figsize=(15,10))
            colors = ['tomato','forestgreen','royalblue','gold']
            bins = np.histogram([item for sublist in w21_assocwise for item in sublist],bins=100)[1]
            labels = []
            for i,desc in enumerate(assocs):
            #   sns.histplot(w21_assocwise[i],bins=bins,label='{}-{} assoc'.format(ODORS_en[od],descs_en[desc]),ax=ax,color=c[i])
                labels.append('{}-\n{} assoc'.format(ODORS_en[od],descs_en[desc]))
            sns.violinplot(data=w21_assocwise,ax=ax,palette=c)
            means = [np.mean(x) for x in w21_assocwise]
            ax.scatter(np.arange(len(means)), means, color='white',s=100)
            ax.set_xticks(np.arange(len(w12_assocwise)))
            ax.set_xticklabels(labels,rotation=0,size=14)
            ax.set_title('Odor -> Language Net',size=14)
            # ax.set_xlim([0,0.5])
            # ax.legend()

            bins = np.histogram([item for sublist in w12_assocwise for item in sublist],bins=100)[1]
            labels = []
            for i,desc in enumerate(assocs):
            #         sns.histplot(w12_assocwise[i],bins=bins,label='{}-{} assoc'.format(descs_en[desc],ODORS_en[od]),ax=ax2,color=c[i])
                labels.append('{}-\n{} assoc'.format(descs_en[desc],ODORS_en[od]))
            means = [np.mean(x) for x in w12_assocwise]
            sns.violinplot(data=w12_assocwise,ax=ax2,palette=c)
            ax2.scatter(np.arange(len(means)), means, color='white',s=100)
            ax2.set_xticks(np.arange(len(w21_assocwise)))
            ax2.set_xticklabels(labels,rotation=0,size=14)
            # ax2.set_xlim([0,0.5])
            ax2.set_title('Language -> Odor Net',size=14)
            # ax2.legend()
            fig.tight_layout()
            plt.show()
                                                                                               



def run():
   
    visualise_associations()
    plot_weights_assocwise('assocw_group_by_nassocs')
    plot_weights_assocwise('bias_group_by_nassocs')

patstat = np.loadtxt('/home/nik/BCPNNSim-olfaction/works/apps/olflang/patstat_si_nclusters4_topdescs.txt')

run()
