# %%
import pandas as pd
from tqdm import tqdm 
import os
import glob
import json as json
import numpy as np
from src.h5toimap import AllData
from src.curvefit import curve_fit
from src.shift import shift_calc
from src.QuickPlot import Plot_
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#%%|
def curve(x,a,b):
     return a*(x**(-b))+1
#%%
df=pd.read_excel(r"Detail.xlsx")

#%%
# extract columns of pandas dataframe which certain column has a value in given list 
List_=df[df["group"].isin([201])]["FileName"]
indexes=[]
for v in List_:
    indices = df[df["FileName"] == v].index.tolist()
    indexes+=indices

#%%
mode="Ret"
#%%
list_ = []
for i in indexes:
    try:
        root_path = df['FileName'].iloc[i]
        print(f"{i}: {root_path}\n")
        Datas = AllData(root_path)
        Datas.set_params(params={
            "force_setpoint": df.iloc[i]['ForceSetpoint'],
            "length_per_cell": df.iloc[i]['ROI']/df.iloc[i]['Point'],
            "shape": [df.iloc[i]['Point'], df.iloc[i]['Point']],
            "tip_radius": df.iloc[i]['Tip radius_mean'],
            "tip_radius_std" : df.iloc[i]['Tip radius_std'],
            "t": df.iloc[i]['Temp'],
            "concentration": df.iloc[i]['Concentration'],
            "probe":df.iloc[i]['Probe'],
            "thickness":df.iloc[i]['InterphaseThickness'],
            "interphase_mode" : "Sigmoid",
            "contact_mechanics": "Schwarz",
            "distance_mode":"Euclid",
            "filter_size":[0,3],
            "mode":"Ret",
            "k":df.iloc[i]['k']

        })
        
        Datas.loadfiles()
        Datas.preprocessing()
        df_ = Datas.export()
        print(np.mean(Datas.Schwarz_local()[Datas.interface[:, 0]-5, Datas.interface[:, 1]]))
        Datas.export_note()
        df["Depth"].iloc[i]=np.mean(np.array(Datas.indentation[0,:]))
        A_ff,rhat,dhat=Datas.contactradius_ff(Datas.tipradius,model="Schwarz")
        df["RelativeInterphaseThickness"].iloc[i]=df.iloc[i]["InterphaseThickness"]/A_ff
        df["Farfieldcontactradius(Schwarz)"].iloc[i]=A_ff
        df["a_hat"].iloc[i]=rhat
        df["d_hat"].iloc[i]=dhat
        df["calc_depth"].iloc[i]=Datas.calc_Depth
            
    except:
        pass
        


    # 
    # print(Datas.contactradius_ff(model="Schwarz"))


       #%%


df.to_excel("Detail_.xlsx")


#%%
List__=df[df["group"].isin([105])]["FileName"]
print(List__)
#%%
popt,_=curve_fit(df,List__,curve,mode=mode)
for l,i in enumerate(List__):
    filelist = os.listdir(glob.glob(f"Data/{i}/Output_extend")[0])
    y=[s for s in filelist if ".csv" in s][0]
    df_ = pd.read_csv(f"Data/{i}/Output/{y}")

    df_shift=shift_calc(df_,curve,popt[0],popt[1])
    df_shift.to_csv(f"Data/{i}/Output/dist_vs_mod.csv")


#%%

fig=Plot_(List_,"Distance from the substrate [nm]","Modulus [GPa]",X="distance", Y="modulus", COLOR="FileName",y_axis=[0,20],x_axis=[-200,500],path=f"Figure/{1000}nm_normalize.png",mode="Ret",Err=False)

#%%
import numpy as np
x=np.linspace(0,60,1000)
y=curve(x,popt[0],popt[1])
fig.axes[0].plot(x,y,color="black")


# %%
fig

# %%
