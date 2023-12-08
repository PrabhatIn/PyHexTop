##==============================PyHexTop python code for toopology optimization===================================
##______________________________Calling the required libraries____________________________________________________
import numpy as np 
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csc_matrix, spdiags
import math
import matplotlib.pyplot as plt
from matplotlib import colors
##______________________________PyHexTop function___________________________________________________________________
def PyHexTop(HNex, HNey, rfill, volfrac, penal, ft):
##______________________________Material Properties_________________________________________________________________
    E0 =1.0
    Emin = E0 * 1e-9
##______________________________Element connectivity, nodal coordinates, Finite element analysis preparation__________
    NstartVs = np.reshape(np.arange(1, (1 + 2*HNex)*(1 + HNey)+1), (1 + 2*HNex, 1 + HNey), order='F')    
    DOFstartVs = np.reshape(2*NstartVs[0:-1, 0:-1] - 1,[2 * HNex*HNey, 1],order='F')
    NodeDOFs = np.tile(DOFstartVs, (1, 8)) + np.tile(np.append(np.add([2, 3, 0, 1], 2*(2*HNex + 1)), [0, 1, 2, 3]), (2*HNex*HNey, 1))
    ActualDOFs = NodeDOFs[np.setdiff1d(np.arange(1, (2*HNex*HNey + 1)), np.arange(2*HNex, 2*HNex*HNey + 1, 2 * HNex) + np.arange(1, HNey+1) % 2)-1, :]
    HoneyDOFs = np.concatenate((ActualDOFs[1::2, 0:2], ActualDOFs[0:: 2, :], ActualDOFs[1::2, 6:8]), dtype = np.int64,axis=1)  #using concatenate because matrices are of different dimensions
    Ncyi = np.tile(np.reshape(np.tile(np.transpose([[-0.25, 0.25]]), (HNey+1, 1)), [2, HNey+1], order='F') + np.sort(1.5 * np.tile(np.arange(0, HNey+1), (2, 1)), axis=0), (HNex+1,1))   
    Ncyi[::, 0::2] = np.flipud(Ncyi[::, 0::2])
    Ncyf = Ncyi[0:-1] #final array containing y-cordinates
    origin = math.cos(math.pi / 6)   
    Ncxf = (np.transpose(np.tile(np.arange(0.0, (2 * HNex * origin)+0.01, origin), (1, HNey+1)))) #nodal coordinates
    HoneyNCO = 1 * (np.concatenate((Ncxf, Ncyf.T.reshape(-1, 1)), axis=1))
    if HNey %2 ==0:
        HoneyDOFs[-1: -HNex:-1, 0:6] -= 2
        HoneyNCO = np.delete(HoneyNCO, [(2*HNex+1)*HNey, (2*HNex+1)*(HNey+1)-1], axis=0)
    Nelem, Nnode = HoneyDOFs.shape[0], HoneyNCO.shape[0] 
    #_____________creating the force matrix___________________________
    row = np.array([2*((2*HNex+1)*HNey+1)-1])
    col = np.array([0])
    data = np.array([-1])   
    F = csc_matrix((data, (row, col)), shape=(2*Nnode, 1))
    #_____________Displacement initlization and DOFs details______________ 
    U = np.zeros((2*Nnode, 1))
    fixeddofs = np.append(2 * (np.arange(1, (2*HNex+1)*HNey+2,  2*HNex+1))-1, (2*(2*HNex+1))) #
    alldofs = np.arange(1, 2*Nnode + 1)    
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    #_____________element stiffness assembly preparation_____________________
    iK = np.reshape(np.kron(HoneyDOFs, np.ones((12, 1))).T,[144*Nelem, 1], order='F') -1 
    jK = np.reshape(np.kron(HoneyDOFs, np.ones((1, 12))).T,[144*Nelem, 1], order='F') -1
    iK = iK.astype(int)
    jK = jK.astype(int)  
    KE = E0 * np.array([[616.43012, 92.77147, -168.07333, 65.54377, -232.28511, -0.00032, -120.65312, -83.28564, -71.60020, -92.77115, -23.81836, 17.74187],
                        [92.77147, 509.30685, 101.02751, -71.90335, 0.00032, -18.03857, -83.28564, -24.48314, -92.77179, -178.72347, -17.74187, -216.15832],
                        [-168.07333, 101.02751, 455.74522, 0.00000, -168.07333, -101.02751, -71.60020, -92.77179, 23.60185, -0.00000, -71.60020, 92.77179],
                        [65.54377, -71.90335, 0.00000, 669.99176, -65.54377, -71.90335, -92.77115, -178.72347, -0.00000, -168.73811, 92.77115, -178.72347],
                        [-232.28511, 0.00032, -168.07333, -65.54377, 616.43012, -92.77147, -23.81836, -17.74187, -71.60020, 92.77115, -120.65312, 83.28564],
                        [-0.00032, -18.03857, -101.02751, -71.90335, -92.77147, 509.30685,17.74187, -216.15832, 92.77179, -178.72347, 83.28564, -24.48314],
                        [-120.65312, -83.28564, -71.60020, -92.77115, -23.81836, 17.74187,616.43012, 92.77147, -168.07333, 65.54377, -232.28511, -0.00032],
                        [-83.28564, -24.48314, -92.77179, -178.72347, -17.74187, -216.15832,92.77147, 509.30685, 101.02751, -71.90335, 0.00032, -18.03857],
                        [-71.60020, -92.77179, 23.60185, -0.00000, -71.60020, 92.77179, -168.07333, 101.02751, 455.74522, 0.00000, -168.07333, -101.02751],
                        [-92.77115, -178.72347, -0.00000, -168.73811, 92.77115, -178.72347,65.54377, -71.90335, 0.00000, 669.99176, -65.54377, -71.90335],
                        [-23.81836, -17.74187, -71.60020, 92.77115, -120.65312, 83.28564, -232.28511, 0.00032, -168.07333, -65.54377, 616.43012, -92.77147],
                        [17.74187, -216.15832, 92.77179, -178.72347, 83.28564, -24.48314, -0.00032, -18.03857, -101.02751, -71.90335, -92.77147, 509.30685]]) / 1000
##______________________________FILTER PREPARATION___________________________________________________________________________________________________________
    Cxx = np.tile(np.append(math.sqrt(3)/2 * np.arange(1, 2*HNex, 2),math.sqrt(3) * np.arange(1, HNex)), (1, math.ceil(HNey/2)))
    Cyy = np.tile(3/4, (HNex, HNey)) + np.tile(3 / 2 * np.arange(0, HNey), (HNex, 1))
    Cyy = np.delete(Cyy.flatten(order='F'), [np.arange(HNex + 1, np.size(Cyy) , 2 * HNex)])
    ct = np.concatenate((Cxx[:, 0:np.size(Cyy)].T, np.transpose([Cyy])), axis=1) # Center coodinate
    DD = np.zeros((Nelem, 1))                                                    # Initializing 
    newx = np.zeros((1))
    newy = np.zeros((1))
    newz = np.zeros((1))
    for j in range(0,Nelem):
        Cent_dist = (np.sqrt(np.square(ct[j,0] - ct[:,0]) + np.square(ct[j,1] - ct[:,1])))
        I = np.add(np.where(Cent_dist <= rfill),1)
        J = np.ones((1,np.size(I)))
        J = J.astype(int)
        newx = np.concatenate((newx,I[0,:]),axis=0)
        newy = np.concatenate((newy, (J+j)[0,:]), axis = 0)
        newz = np.concatenate((newz, Cent_dist[I-1][0,:]), axis = 0)      
    DD = np.hstack((newx[1:,].reshape((-1,1)),newy[1:,].reshape((-1,1)), newz[1:,].reshape((-1,1))))
    DD = DD.astype(int)
    HHs = coo_matrix((1- DD[:,2]/ rfill, (DD[:,1]-1, DD[:,0]-1))).toarray()
    HHs = (spdiags(1 / (np.sum((HHs), axis=1)), col, Nelem, Nelem) * HHs)   
##______________________________Initialization for optimization_______________________________________________________
    x = volfrac * np.ones((Nelem,1))
    xPhys, loop, change, maxiter, dv, move = x, 0, 1.0, 200, np.ones((Nelem,1)), 0.2   
    freedofs = np.array((freedofs-1), dtype = np.int32)
    start_time = time.time()
##______________________________Start optimization_____________________________________________________________________
    while (change > 0.01 and loop < maxiter):
        loop += 1
        #_____________finite element analysis____________________________________________________
        sK = ((KE.reshape(-1).reshape((-1,1))) * (Emin+(xPhys.T) **penal*(E0-Emin))).reshape((144*Nelem, 1), order='F')          
        K = coo_matrix((sK[:,0], (iK[:,0], jK[:,0])))
        check = K.tocsc()[freedofs,:][:,freedofs]      
        U[freedofs, 0] = spsolve(check, F[freedofs,0])
        #_____________Objective and sensitivities evaluation____________________________________         
        ce = (U[HoneyDOFs-1, 0].dot(KE) * U[HoneyDOFs-1, 0]).sum(1)               
        c = sum((Emin + xPhys**penal*(E0-Emin))*(ce.reshape((-1,1)))).sum()       
        dc = (-penal * (E0 - Emin) * xPhys**(penal-1))*ce.reshape((-1,1))
        #_____________Using filters______________________________________________________________
        if ft ==1 :
            dc = np.dot(HHs, (x*dc))/np.maximum(0.001, x)    
        elif ft ==2:
            dc = np.dot(HHs.T, dc)
            dv = np.dot(HHs.T, dv)    
        #_____________Optimality criteria update__________________________________________________
        xOpt = x
        xUpp, xLow = xOpt + move, xOpt - move
        OcC = xOpt * np.sqrt(-dc/dv)
        inL = np.concatenate(([0], [np.mean(OcC)/volfrac]))        
        while (inL[1]-inL[0])/(inL[1]+inL[0]) > 1e-3:
            lmid = 0.5 * (inL[1] + inL[0])
            x = np.maximum(0.0, np.maximum(xLow, np.minimum(1.0, np.minimum(xUpp, OcC/lmid))))
            if np.mean(x) > volfrac : inL[0] = lmid
            else: inL[1] = lmid        
        if ft == 1 or ft == 0 :
            xPhys = x        
        elif ft ==2 :             
            xPhys = HHs.dot(x)      
        change = np.amax(abs(xOpt -x))
##______________________________Results printing___________________________________________________________________
        print(f"It: {loop}   Obj: {c : 11.4f}     Vol: {np.mean(xPhys) : 7.3f}    ch:{change : 7.3f}")      
##______________________________Plotting Designs____________________________________________________________________
    plt.figure(figsize=(10,5))
    plt.scatter(ct[:, 0], ct[:, 1],  c=1.0-xPhys, cmap='copper')
    plt.axis('off')
    plt.savefig(f'{HNex}x{HNey}_{ft}_{rfill:.2f}.eps')
    plt.show()
##______________________________driver code__________________________________________________________________________
PyHexTop(HNex=150,HNey=50, rfill = 6* np.sqrt(3), volfrac = 0.5, penal = 3, ft=2)

############################################################################
# This code is written by A. Agarwal, A. Saxena and P. Kumar               #
# Department of Mechanical and Aerospace Engineering,Indian Institute of   #
# Technology Hyderabad and Department of Mechanical Engineering,Indian     #
# Institute of Technology Kanpur                                           #
#                                                                          #
#  PyHexTop code is presented for the educational purposes and its         #
#  details can be found in "PyHexTop: a compact Python code for            #
#  topology optimization using hexagonal elements"                         #
#  The code is based on the 90-line code published in Prabhat Kumar (2023) #
#  "HoneyTop90: A 90-line MATLAB code for topology optimization using      #
#   honeycomb tessellation, 2023, Optimization and Engineering             #  
#                                                                          #        
#  Please write : pkumar@mae.iith.ac.in or aditi.s.agarwal02@gmail.com     #
#  for any comments                                                        #
#                                                                          #                                              
#  Disclaimer:                                                             #
#  The authors reserve all rights but do not guaranty that the code is     #
#  free from errors and the authors shall not be liable in any event       #
#  caused by the use of the code.                                          #
############################################################################