from tqdm import tqdm
from scipy.signal import periodogram
import glob
import json as json
import numpy as np
import pandas as pd
import os.path
import os
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import scipy as sp
import pickle
from scipy.spatial.distance import cdist
from .edgedetection import canny

class AllData:
    def __init__(self, rootpath):
        self.rootpath = rootpath
        self.force_setpoint = 0
        self.tipradius = 0
        self.interphase_mode = ""
        self.contact_mechanics = ""
        self.distance_mode =""
        return None

    def loadfiles(self):
        """
        Load data from files
        .h5 : include all information , including all fforce curve
        RetAdhesionForce.npy : Critical force map
        RetIndentationModulus.npy : Modulus map
        """
        
        path=f"Data/{self.rootpath}/Data_CR"

        filelist = os.listdir(glob.glob(path)[0])
        for x in filelist:
            if "SumSquaredError.npy" in x:
                self.SSE=np.load(path+f"/{x}")
        if self.mode=="Ext":
            path_=f"Data/{self.rootpath}/Data_extend"
            filelist_ = os.listdir(glob.glob(path_)[0])
            for x in filelist_:
                if "AdhesionForce.npy" in x:
                    self.Fcrit = np.load(path_+f"/{x}")*1e9
                if "IndentationModulus.npy" in x:
                    self.Mod = np.load(path_+f"/{x}")/1e9
                #     plt.imsave(
                # f"Data/{self.rootpath}/Output_extend/modulus.png", self.Mod)
                if "IndentationModulusErr.npy" in x:
                    self.ModErr = np.load(path_+f"/{x}")/1e9
                if "Indentation.npy" in x:
                    self.indentation= np.load(path_+f"/{x}")*1e9

                if "SumSquaredError.npy" in x:
                    self.SSE=np.load(path_+f"/{x}")

                if "Deflection.npy" in x:
                    self.deflection=np.load(path_+f"/{x}")
        else:
            path_=f"Data/{self.rootpath}/Data_CR"
            for x in filelist:
                if "AdhesionForce.npy" in x:
                    self.Fcrit = np.load(path+f"/{x}")*1e9
                if "IndentationModulus.npy" in x:
                    self.Mod = np.load(path+f"/{x}")/1e9
                #     plt.imsave(
                # f"Data/{self.rootpath}/Output_extend/modulus.png", self.Mod)
                if "Indentation.npy" in x:
                    self.indentation= np.load(path+f"/{x}")*1e9
                if "IndentationModulusErr.npy" in x:
                    self.ModErr = np.load(path+f"/{x}")/1e9

                if "SumSquaredError.npy" in x:
                    self.SSE=np.load(path_+f"/{x}")

                if "Deflection.npy" in x:
                    self.deflection=np.load(path_+f"/{x}")*1e9

                if "ContactRadius.npy" in x:
                    self.CR=np.load(path_+f"/{x}")*1e9
        
         


    def preprocessing(self):
        """
        Interpolation and wavelet denoising for Modulus map
        """
        
        if type(self.Mod) == np.ndarray and type(self.Fcrit) == np.ndarray and type(self.indentation) == np.ndarray:
            pass
        else:
            raise ValueError("No available map")

    def find_interphase_sigmoid(self, Mod):
        """
        Find interphase by sigmoid fitting.
        Interphase is defined as a center point between 2 points which 2nd derivaitve of the fitted sigmoid get 0.
        return : 0/1 matrix which has 1 when the index match to interphase, otherwise 0
        """
        def sigmoid(x, k2, k3, k1, x0):
            return k2-(k3 / (1 + np.exp(-k1*(x-x0))))

        def sigmoid_first_derivative(x, k2, k3, k1, x0):
            return -1*k1*k3*np.exp(-k1*(x-x0))/(np.exp(-k1*(x-x0))+1)**2
            

        interphase_index = np.zeros(int(Mod.shape[1]))
        interphase_matrix = np.zeros(self.shape)

        x_data = np.arange(int(Mod.shape[0]))
        x_index = np.arange(0, int(Mod.shape[1]))
        for i in x_index:
            y_data = Mod[:, i]
            valid = np.nonzero(~(np.isnan(y_data) |
                                 np.isinf(y_data)))
            y_data_ = y_data[valid]
            x_data_ = x_data[valid]

            y_data_normalized = (y_data_-y_data_.min()) / \
                (y_data_.max()-y_data_.min())
            try:
                popt, pcov = sp.optimize.curve_fit(
                    sigmoid, x_data_, y_data_normalized, method='lm')
                y_deriv = sigmoid_first_derivative(x_data_, *popt)
                crit_x = x_data_[np.nanargmax(y_deriv)]

                interphase_matrix[crit_x,i] = 1
                
            except RuntimeError:
                pass
            except IndexError:
                pass 
        np.save(f"Data/{self.rootpath}/Output_CR/interphase",interphase_matrix)
        self.interphase_matrix=interphase_matrix

        return interphase_matrix
    
    
    def find_interphase_canny(self,Mod,sigma):

        a = canny(sp.ndimage.gaussian_gradient_magnitude(Mod, sigma=sigma), sigma, .0, 1., mode="nearest")

        self.interphase_matrix=a
        return a

    
    def interphase_mat2coord(self,interphase_matrix):
        return np.vstack(np.where(interphase_matrix==1)).T


    

    def build_distant_matrix_interpolation(self,interphase_index):
        """
        Calculate the distance from the nearest interphase for all points.
        The indexes of interphase are interporated as a linear function, and calculate exact distance.
        input : interphase_matrix
        output : distant_matrix
        """
        x = np.arange(0, len(interphase_index))
        a, b = np.polyfit(
                x, interphase_index, 1)
        distant_matrix = np.empty(self.shape)
        for i in interphase_index:
            crit_x = i[0]
            if crit_x != None:
                for j in np.arange(self.shape[1]):
                    if j <= crit_x:
                        distant_matrix[j, i[1]] = (
                            np.abs(a*i-j+b)/(np.sqrt(a**2+1)))*self.length_per_cell
                    else:
                        distant_matrix[j, i[1]] = 0
            else:
                distant_matrix[:, i[1]] = None

        return distant_matrix*self.length_per_cell

    def build_distant_matrix_euclid(self,interphase_index):
        """
        Calculate the distance from the nearest interphase for all points.
        input : interphase_matrix
        output : distant_matrix
        """
        coordinate=np.stack(np.meshgrid(np.arange(0,self.shape[0]),np.arange(0,self.shape[0])), -1).reshape(-1, 2)

        # Calculate the distance between each point in A and B
        distances = cdist(coordinate, interphase_index)

        # Find the index of the closest point in interphase for each coordinate
        closest_indices = np.argmin(distances, axis=1)

        # Calculate the closest distances
        distant_matrix = np.array(distances[np.arange(len(coordinate)), closest_indices]).reshape(self.shape).T
        for i in interphase_index:
            crit_x = i[0]
            if crit_x != None:
                for j in np.arange(self.shape[1]):
                    if j >= crit_x:
                        distant_matrix[j, i[1]] = -distant_matrix[j, i[1]]
                    else:
                        pass
            else:
                distant_matrix[:, i[1]] = None
        return distant_matrix*self.length_per_cell


    def contactradius_ff(self,R,model="Schwarz"):
        """
        Calculate contact radius of each points based on each critical force,modulus ,and tip radius
        output : contact radius matrix
        """

        if type(self.Fcrit) != np.ndarray:
            raise ValueError("No available Critical ForceMap")
        if type(self.Mod) != np.ndarray:
            raise ValueError("No available Modulus Map")
        if type(self.indentation) != np.ndarray:
            raise ValueError("No available indentation Map")
        elif self.force_setpoint == None:
            raise ValueError("No indentation force")
        elif self.tipradius == None:
            raise ValueError("No tip_radius")
        else:
            # plt.imsave(
            # f"Data/{self.rootpath}/Output/force.png", np.nan_to_num(self.Fcrit,nan=0,posinf=0))
            valid = ~(np.isnan(self.Fcrit[0, :]) |
                      (np.isnan(self.Mod[0, :]))|
                      (np.isnan(self.indentation[0, :]))
                      )
        if model == "Schwarz":
            a_ff,rhat,dhat=self.Schwarz(valid,R)
        elif model == "Hertz":
            a_ff=self.Hertz(valid)
        else:
            a_ff =0
        return a_ff,rhat,dhat


    def Schwarz(self,valid,R):
        tau = 1
        reduced_critcal_force = (tau - 4) / 2
        indentation_force =np.mean(self.deflection[0,valid]*self.k)
        indentation_modulus = np.mean(self.Mod[0, valid])
        tip_radius=R
        critical_force=np.mean(self.Fcrit[0, valid])

        reference_force = -critical_force / reduced_critcal_force
        reduced_contact_radius = (
                (3 * reduced_critcal_force + 6) ** (1 / 2)
                + (indentation_force / reference_force  -reduced_critcal_force) ** (1 / 2)
                ) ** (2 / 3)
        A_ff = reduced_contact_radius * (
                indentation_modulus / (reference_force *tip_radius)
                ) ** (-1 / 3)
        reduced_depth=reduced_contact_radius**2-4*(reduced_contact_radius*(reduced_critcal_force+2)/3)**(1/2)
        calc_depth=reduced_depth*((reference_force/indentation_modulus)**2/tip_radius)**(1/3)
        self.calc_Depth=calc_depth
        return A_ff,reduced_contact_radius,reduced_depth
    
    def Schwarz_local(self):
        tau = 1
        reduced_critcal_force = (tau - 4) / 2
        indentation_force =self.force_setpoint
        indentation_modulus = self.Mod
        tip_radius=self.tipradius
        critical_force=self.Fcrit
        reference_force = -critical_force / reduced_critcal_force
        reduced_contact_radius = (
                (3 * reduced_critcal_force + 6) ** (1 / 2)
                + (indentation_force / reference_force  -reduced_critcal_force) ** (1 / 2)
                ) ** (2 / 3)
        A_ff = reduced_contact_radius * (
                indentation_modulus / (reference_force *tip_radius)
                ) ** (-1 / 3)
        return A_ff

    
    def Hertz(self,valid):
        nu=0.5
        reduced_modulus=np.mean(self.Mod[0,valid])/(1-nu**2)
        indentation_=np.mean(self.indentation[0,valid])
        A_ff=3*self.force_setpoint/(indentation_*reduced_modulus*4)
        return A_ff


    def modulus_ff(self):
        """
        Far field modulus
        """
        valid = ~(np.isnan(self.Fcrit[0, :]) |
                      (np.isnan(self.Mod[0, :])))
        return np.mean(self.Mod[0, valid])
    
    def find_interphase(self,Mod_,mode):
        Mod=sp.ndimage.gaussian_filter(Mod_,sigma=self.filter)
        # plt.imsave(
        #     f"Data/{self.rootpath}/Output_extend/modulus-filter.png", Mod)
        if mode=="Canny":
            sigma=np.linspace(1,5,10)
            after_gaussian=sp.ndimage.gaussian_gradient_magnitude(Mod, sigma=sigma[0])
            a = canny(after_gaussian, sigma[0], .0, 1., mode="nearest")
            for x in sigma:
                a_=self.find_interphase_canny(Mod,x)
                if np.min(np.where(a_==1)[1])<=np.min(np.where(a==1)[1])  and  np.max(np.where(a_==1)[1])>=np.max(np.where(a==1)[1]) :
                    a=a_
                else:
                    pass
            interphase_matrix=a
        elif mode=="Sigmoid":
            interphase_matrix=self.find_interphase_sigmoid(Mod)
        else:
            raise ValueError("Invalid mode")
        self.interphase_matrix=interphase_matrix
        return self.interphase_mat2coord(interphase_matrix)
    
    def build_distant_matrix(self, interphase_index,mode):
        if mode=="Interpolation":
            distant_matrix = self.build_distant_matrix_interpolation(interphase_index)
        elif mode=="Euclid":
            distant_matrix=self.build_distant_matrix_euclid(interphase_index)
        else:
            raise ValueError("Invalid mode")
        # plt.imsave(
        #     f"Data/{self.rootpath}/Output_extend/distance.png", distant_matrix)
        return distant_matrix


    def normalize_distant(self,R):
        """
        Normalize distant from the substrate by each contact radius
        """

        substrate_interphase = self.find_interphase(self.Mod,mode=self.interphase_mode)
        self.interface = substrate_interphase
        self.distant_matrix=self.build_distant_matrix(self.interface,mode=self.distance_mode)
        a_ff,ahat,dhat = self.contactradius_ff(R,model=self.contact_mechanics)
        normalized_dist = self.distant_matrix/a_ff
        return normalized_dist

    def calc_a_cap(self):
        result = []
        for i in range(self.interphase_matrix.shape[0]): # type: ignore
            for j in range(self.interphase_matrix.shape[0]):# type: ignore
                if self.interphase_matrix[i][j] == 1:
                    result.append(self.indentation[i][j])
        indentation_=np.mean(result)
        A_cap=np.sqrt(indentation_*(2*self.tipradius-indentation_))
        return A_cap

# There are 2 matrix which have same size. One has only 0 or 1, the other has float value. The function extract float value from 2nd matrix, at the same index of the 1 in the 1st matrix.   

    def export(self):
        """
        export normalized_distant/modulus corr for each force setpoint
        """
        # try:
        norm_dist=self.normalize_distant(self.tipradius).reshape(self.shape[0]*self.shape[1])
        dist = self.distant_matrix.reshape(self.shape[0]*self.shape[1])
        # CR_ff=np.mean(self.Schwarz_local()[0,:])
        self.CR_ff=np.mean(self.CR[0,:])
        Mod_ff = np.mean(self.Mod[0,:])
        #A_cap=np.mean(self.Schwarz_local()[self.interface[:, 0]-1, self.interface[:, 1]])
        self.A_cap=np.mean(self.CR[self.interface[:, 0]-5, self.interface[:, 1]])
        print(self.A_cap)
        hmod=self.Mod*3*0.75/4
        SSE=self.SSE.reshape(self.shape[0]*self.shape[1])
        mod=self.Mod.reshape(self.shape[0]*self.shape[1])
        norm_mod = mod/Mod_ff
        x_int_Aff=(dist-self.thickness)/self.CR_ff
        data = np.vstack((dist, norm_dist, mod, norm_mod,SSE,x_int_Aff)).T
        df = pd.DataFrame(
        data, columns=["distance", "normalized_distance", "modulus","normalized_modulus","SSE","x_int_Aff"])
        df=df[~((df["normalized_distance"]>1)&(df["normalized_modulus"]>=20))& (df["SSE"]<10000)]
        df=df.dropna()
        df["ContactRadius"]=self.CR_ff
        df["A_cap"]=self.A_cap
        df["ForceSetpoint"] = self.force_setpoint
        df["RelativeInterphaseThickness"]=self.thickness/self.contactradius_ff(self.tipradius)[0]
        df["FileName"]=self.rootpath
        if self.mode=="Ext":
            df.to_csv(f"Data/{self.rootpath}/Output_extend/dist_vs_mod.csv")
        else:
            df.to_csv(f"Data/{self.rootpath}/Output_CR/dist_vs_mod_cr.csv")
        return df
    

    def export_note(self):
        """
        Pickle all the data
        """
        dict = {
            "force_setpoint": self.force_setpoint,
            "length_per_cell": self.length_per_cell,
            "shape": self.shape,
            "tip_radius": self.tipradius,
            "contact_radius": self.CR_ff,
            "A_cap" : self.A_cap
        }
        with open(f"Data/{self.rootpath}/Output_CR/notes.txt", "w") as handle:
            print(dict, file=handle)
            # pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dict


    def set_params(self, params):
        # if self.h5 != None:
        #     k, self.force_setpoint, self.length_per_cell, self.shape = self.persenote()
        if params != None:
            self.force_setpoint = params["force_setpoint"]
            self.length_per_cell = params["length_per_cell"]
            self.shape = (int(params["shape"][0]),int(params["shape"][1]))
            self.tipradius = params["tip_radius"]
            self.R_Std=params["tip_radius_std"]
            self.thickness=params["thickness"]
            self.interphase_mode = params["interphase_mode"]
            self.contact_mechanics = params["contact_mechanics"]
            self.distance_mode =params["distance_mode"]
            self.probe=params["probe"]
            self.filter=params["filter_size"]
            self.mode=params["mode"]
            self.k=params["k"]
        else:
            raise ValueError("No Available Parameters")


