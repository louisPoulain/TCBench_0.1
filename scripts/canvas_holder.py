import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
import numpy as np



class canvasHolder():
    def __init__(self, lat_list, lon_list):

        self.lat_list, self.lon_list = lat_list, lon_list
                
        self.Coords=[self.lon_list, self.lat_list]
                
        #print(self.Coords[0].shape, self.Coords[1].shape, self.lonlat[0].shape, self.lonlat[1].shape)
        
        #self.proj0=ccrs.Stereographic(central_latitude=46.7,central_longitude=2.0)
        
        self.proj_plot=ccrs.Robinson()
        self.axes_class= (GeoAxes,dict(projection=self.proj_plot))
        
    
    def project(self, ax=None):
        if ax is None:
            ax = plt.axes(projection=self.proj_plot)
        ax.set_extent([min(self.lat_list), max(self.lat_list), min(self.lon_list), max(self.lon_list)], crs=self.proj_plot)
        return ax
    
    
    def plot_abs_error(self, data, var_names,
                    plot_dir, pic_name, col_titles,
                    contrast=False, cmap_wind='viridis', cmap_mslp='Blues', suptitle=''):
            
            """
            
            use self-defined axes structures and projections to plot numerical data
            and save the figure in dedicated directory
            
            Inputs :
                
                data: list[list[xr.Dataset]] -> data to be plotted shape Samples x Channels x Lat x Lon
                            with  Channels being the number of variables to be plotted
                    
                plot_dir : str -> the directory to save the figure in
                
                pic_name : str -> the name of the picture to be saved
                    
                
                contrast : bool (optional) -> check if boundary values for plot 
                                            shoud be imposed (same value for all variables)
                
                cvalues : tuple (optional) -> bottom and top of colorbar plot [one
                for each variable]
                
                withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude
                
        
            Returns :
            """
                
            samples, channels = len(data), len(data[0])
            fig=plt.figure(figsize=(8*samples,3*channels), facecolor='white')
            axes={}
            ims={}
            #print(len(data), [len(data[i]) for i in range(len(data))])
            data_np = np.array(data)
            grid=AxesGrid(fig, 111, axes_class=self.axes_class,
                    nrows_ncols=(channels,samples),
                    axes_pad=(0.3,0.25),
                    cbar_pad=0.25,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    label_mode='')
            coef = -0.5
            for ind in range(samples):
                #plotting each sample 
                
                data_plot=data[ind]
                
                
                for i, var in enumerate(var_names):
                    #print(var)
                    Var=var[0]
                    unit=var[1]
                    #print(Var)
                    if Var=='mslp':
                        Var = "Mean Sea Level Pressure"
                        cmap = cmap_mslp
                    elif Var=='wind' :
                        Var = "Wind magnitude"
                        cmap = cmap_wind
                    else:
                        cmap = 'coolwarm'
                    axes[Var+str(ind)]=self.project(ax=grid[i*samples+ind])

                    if not contrast:
                        ims[Var+str(ind)]=axes[Var+str(ind)].pcolormesh(
                            self.Coords[0],\
                            self.Coords[1],\
                            data_plot[i],\
                            shading='auto',
                            cmap=cmap,alpha=1,transform=ccrs.PlateCarree())
                    else :
                        ims[Var+str(ind)]=axes[Var+str(ind)].pcolormesh(
                            self.Coords[0],\
                            self.Coords[1],\
                            data_plot[i],\
                            shading='auto',
                            cmap=cmap,alpha=1,vmin=data_np.min(axis=(0,2,3))[i], vmax=data_np.max(axis=(0,2,3))[i],
                            transform=ccrs.PlateCarree())
                    
                    
                    axes[Var+str(ind)].add_feature(cfeature.COASTLINE.with_scale('10m')) # adding coastline
                    axes[Var+str(ind)].add_feature(cfeature.BORDERS.with_scale('10m')) # adding borders

                    if i==0:
                        axes[Var+str(ind)].set_title(col_titles[ind], fontsize = 15)#33
                    if ind==0:
                        add_unit = ' ('+unit+')' if unit!='' else ""
                        grid.cbar_axes[i].colorbar(ims[Var+str(ind)]).set_label(label=Var + add_unit, size=20)#33
                        grid.cbar_axes[i].tick_params(labelsize=12)#32
                    
                        
                coef = coef + 0.25
            
            fig.subplots_adjust(bottom=0.005, top=0.92-0.03*len(var_names), left=0.05, right=0.95)
            st=fig.suptitle(suptitle, fontsize='20')#36
            st.set_y(0.98)
            #st.set_y(0.98)
            fig.canvas.draw()
            
            #fig.tight_layout()
            plt.savefig(plot_dir+pic_name, dpi=400, bbox_inches='tight')
            
            
        
    def plot_abs_error_sev_cbar(self, data, var_names,
                plot_dir, pic_name, col_titles,
                contrast=True, cmap_wind='viridis', cmap_mslp='Blues', suptitle=''):
        
        """
        
        use self-defined axes structures and projections to plot numerical data
        and save the figure in dedicated directory
        
        Inputs :
            
            data: np.array -> data to be plotted shape Samples x Channels x Lat x Lon
                        with  Channels being the number of variables to be plotted
                
            plot_dir : str -> the directory to save the figure in
            
            pic_name : str -> the name of the picture to be saved
                
            
            contrast : bool (optional) -> check if boundary values for plot 
                                        shoud be imposed (same value for all variables)
            
            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
            for each variable]
            
            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude
            
    
        Returns :
            
            
        Note :
            
            last docstring review by C .Brochet 15/04/2022
            
        """
            
        
        fig=plt.figure(figsize=(4*data.shape[0],3*len(var_names)), facecolor='white')
        axes={}
        ims={}
        
        grid=AxesGrid(fig, 111, axes_class=self.axes_class,
                nrows_ncols=(len(var_names),data.shape[0]),
                axes_pad=(0.75, 0.25),
                cbar_pad=0.1,
                cbar_location="right",
                cbar_mode="each",
                cbar_size="7%",
                label_mode='')
        coef = -0.5
        for ind in range(data.shape[0]):
            #plotting each sample 
            
            data_plot=data[ind,:,:,:]
            
            
            for i, var in enumerate(var_names):
                #print(var)
                Var=var[0]
                unit=var[1]
                #print(Var)
                if Var=='mlsp' :
                    cmap = cmap_mslp
                else:
                    cmap = cmap_wind
                axes[Var+str(ind)]=self.project(ax=grid[i*data.shape[0]+ind])

                ims[Var+str(ind)]=axes[Var+str(ind)].pcolormesh(
                    self.Coords[0],\
                    self.Coords[1],\
                    data_plot[i,:,:],\
                    cmap=cmap,alpha=1,
                    transform=self.proj_plot, shading="nearest")
                
                
                axes[Var+str(ind)].add_feature(cfeature.COASTLINE.with_scale('10m')) # adding coastline
                axes[Var+str(ind)].add_feature(cfeature.BORDERS.with_scale('10m')) # adding borders

                if i==0:
                    axes[Var+str(ind)].set_title(col_titles[ind], fontsize = 15)#33
                #if ind==0 or Var=='rr':
                    #print('INDEXE',ind)
                    #grid.cbar_axes[i].colorbar(ims[Var+str(ind)], format='%.0e')
                
                cb = grid.cbar_axes[i*data.shape[0]+ind].colorbar(ims[Var+str(ind)])
                if ind==data.shape[0]-1 or Var=='rr':
                    cb.set_label(label=Var + ' ('+unit+')', size=20)#33
                grid.cbar_axes[i*data.shape[0]+ind].tick_params(labelsize=12)#32
                
                    
            coef = coef + 0.25
        #for ind in range(data.shape[0]):
        #    axes['None'+str(ind)] = self.project(ax=grid[ind])
        #    axes['None'+str(ind)].axis('off')
        #    axes['None'+str(ind)].set_title(col_titles[ind])

        #Title='3-fields states'
        fig.subplots_adjust(bottom=0.005, top=0.80 if len(var_names)==1 else 0.98-0.02*len(var_names), left=0.05, right=0.95)
        st=fig.suptitle(suptitle, fontsize='20')#36
        st.set_y(0.98)
        fig.canvas.draw()
        
        #fig.tight_layout()
        plt.savefig(plot_dir+pic_name, dpi=400, bbox_inches='tight')