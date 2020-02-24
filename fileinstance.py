# Written by Josh Surya

import glob
import os
#import plot_transmission
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib.pyplot import cm
import pdb
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

# This class assumes that the name of the IR and corresponding SHG files analyzed have the same names other than the
# suffix
class file_obj:
    def __init__(self, path=None, ir_suffix = None, shg_suffix = None):
        if path == None:
            raise Exception('file path is not indicated')

        if ir_suffix == None:
            self.ir_suffix = 'datIR'
        else:
            self.ir_suffix = ir_suffix

        if shg_suffix == None:
            self.shg_suffix = 'datSHG'
        else:
            self.shg_suffix = shg_suffix

        self.filepath = path
        self.shgfiles = sorted(glob.glob(self.filepath + '\*' + self.shg_suffix + '.dat'))
        self.irfiles = sorted(glob.glob(self.filepath + '\*' + self.ir_suffix + '.dat'))
        # print(self.irfiles)
        # pdb.set_trace()
        self.shg_wavelengths = {}
        self.shg_responses = {}
        self.ir_wavelengths = {}
        self.ir_responses = {}
        self.figure_index=0
        try:
            self.get_shgdata()
            self.get_irdata()
        except:
            print(len(self.shg_wavelengths))
        self.irpeaks = {}
        self.irpeaksinfo = {}
        self.shgpeaks = {}
        self.shgpeaksinfo = {}
        self.index_track = []
        self.index_modes = {}
        #print(self.ir_wavelengths[0][1:10])
        #self.plot_ir_multi(yautoscale=True,ylogscale=True,xrange=[1532.24,1532.2575])

    def get_shgdata(self,resmooth=False):

        filename = self.shgfiles

        for i in range(len(filename)):
            wl_list = []
            response_list = []
            file_read = open(filename[i],'r')

            for line in file_read:
                lst1 = line.split()[0] # [0] takes the first column of the n_th line, [1] takes the second column of the n_th line
                lst2 = line.split()[1]
                wl_list.append(lst1)
                response_list.append(lst2)

            self.shg_wavelengths[i] = np.asarray(list(map(float, wl_list)))
            self.shg_responses[i] = np.asarray(list(map(float,response_list)))
            if resmooth:
                self.ir_wavelengths[i] = np.linspace(self.shg_wavelengths[i][0],self.shg_wavelengths[i][-1],len(self.shg_wavelengths[i]))

    def get_irdata(self,resmooth=False):

        filename = self.irfiles

        for i in range(len(filename)):
            wl_list = []
            response_list = []
            file_read = open(filename[i], 'r')

            for line in file_read:
                lst1 = line.split()[0]  # [0] takes the first column of the n_th line, [1] takes the second column of the n_th line
                lst2 = line.split()[1]
                wl_list.append(lst1)
                response_list.append(lst2)

            self.ir_wavelengths[i] = np.asarray(list(map(float, wl_list)))
            self.ir_responses[i] = np.asarray(list(map(float,response_list)))
            if resmooth:
                self.ir_wavelengths[i] = np.linspace(self.ir_wavelengths[i][0],self.ir_wavelengths[i][-1],len(self.ir_wavelengths[i]))


    # This is called when the user wants multiple figures as opposed to having all graphs displayed on one figure
    def plot_ir_multi(self, yautoscale=False, ylogscale=False, xrange=None, yrange=None,
                       color=None, pattern=None, linewidth=None, save=False,
                       sx=None, sy=None):
        if color == None:
            color = 'b'
        if pattern == None:
            pattern = '-'
        if linewidth == None:
            linewidth = 1.0
        if sx == None:
            sx = 8
        if sy == None:
            sy = 6
        j = 0

        for i in range(len(self.irfiles)):
            plt.figure(self.figure_index, figsize=(sx,sy))
            plt.plot(self.ir_wavelengths[i], self.ir_responses[i], color + pattern)

            if xrange == None:
                pass
            else:
                plt.xlim(xrange[0],xrange[1])

            if yrange == None:
                pass
            else:
                plt.ylim(yrange[0],yrange[1])

            if ylogscale == False:
                pass
            else:
                plt.yscale("log")

            if yautoscale == False:
                pass
            else: # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
                if xrange == None:
                    raise Exception("no xrange set")
                delta = self.ir_wavelengths[i][1]-self.ir_wavelengths[i][0]
                index_min = int((xrange[0]-self.ir_wavelengths[i][0])//delta)
                index_max = int((xrange[1]-self.ir_wavelengths[i][0])//delta)
                y_min = np.amin(self.ir_responses[i][index_min:index_max])
                y_max = np.amax(self.ir_responses[i][index_min:index_max])
                plt.ylim(y_min*0.9,y_max*1.1)

            plt.ylabel('Transmission (a.u.)')
            plt.xlabel('Wavelength (nm)')
            if save == False:
                plt.show()
            else:
                my_path = self.filepath
                plt.savefig(my_path + '/multiplotIR_' + str(j) + '.png', bbox_inches="tight")
                j += 1
                plt.close()
            self.figure_index += 1

    def plot_shg_multi(self, yautoscale=False, ylogscale=False, xrange=None, yrange=None,
                       color=None, pattern=None, linewidth=None, save=False,
                       sx=None, sy=None):

        if color == None:
            color = 'r'
        if pattern == None:
            pattern = '-'
        if linewidth == None:
            linewidth = 1.0
        if sx == None:
            sx = 8
        if sy == None:
            sy = 6
        j = 0
        for i in range(len(self.shgfiles)):
            plt.figure(self.figure_index, figsize=(sx,sy))
            plt.plot(self.shg_wavelengths[i], self.shg_responses[i], color + pattern, linewidth=linewidth)

            if xrange == None:
                pass
            else:
                plt.xlim(xrange[0], xrange[1])

            if yrange == None:
                pass
            else:
                plt.ylim(yrange[0], yrange[1])

            if ylogscale == False:
                plt.yscale("linear")
            else:
                plt.yscale("log")

            if yautoscale == False:
                pass
            else:  # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
                if xrange == None:
                    raise Exception("no xrange set")
                delta = self.shg_wavelengths[i][1] - self.shg_wavelengths[i][0]
                index_min = int((xrange[0] - self.shg_wavelengths[i][0]) // delta)
                index_max = int((xrange[1] - self.shg_wavelengths[i][0]) // delta)
                y_min = np.amin(self.shg_responses[i][index_min:index_max])
                y_max = np.amax(self.shg_responses[i][index_min:index_max])
                plt.ylim(0, y_max * 1.1)

            plt.ylabel('SHG (pW)')
            plt.xlabel('Wavelength (nm)')
            if save == False:
                plt.show()
            else:
                my_path = self.filepath
                plt.savefig(my_path + '/multiplotSHG_' + str(j) + '.png', bbox_inches="tight")
                j += 1
                plt.close()
            self.figure_index += 1


    def plot_shg_single(self, yautoscale=False, ylogscale=False, xrange=None, yrange=None, color=None, pattern=None,
                        sx=None, sy=None, linewidth=None, save=False):
        if color == None:
            color = 'r'
        if pattern == None:
            pattern = '-'
        if sx == None:
            sx = 8
        if sy == None:
            sy = 6
        if linewidth == None:
            linewidth = 1.0
        fig, ax = plt.subplots(len(self.shgfiles), 1, figsize=(sx,sy))
        j = 0

        for i in range(len(self.shgfiles)):
            ax[i].plot(self.shg_wavelengths[i], self.shg_responses[i], color + pattern, linewidth=linewidth)
            ax[i].get_xaxis().set_visible(False)
            if xrange == None:
                pass
            else:
                ax[i].set_xlim(xrange[0], xrange[1])

            if yrange == None:
                pass
            else:
                ax[i].set_ylim(yrange[0], yrange[1])

            if ylogscale == False:
                ax[i].set_yscale("linear")
            else:
                ax[i].set_yscale("log")

            if yautoscale == False:
                pass
            else:  # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
                if xrange == None:
                    raise Exception("no xrange set")
                delta = self.shg_wavelengths[i][1] - self.shg_wavelengths[i][0]
                if delta == 0:
                    delta = (self.shg_wavelengths[i][-1] - self.shg_wavelengths[i][0]) / len(self.shg_wavelengths[i])
                index_min = int((xrange[0] - self.shg_wavelengths[i][0]) // delta)
                index_max = int((xrange[1] - self.shg_wavelengths[i][0]) // delta)
                y_min = np.amin(self.shg_responses[i][index_min:index_max])
                y_max = np.amax(self.shg_responses[i][index_min:index_max])
                ax[i].set_ylim(0, y_max * 1.1)

            if i == len(self.shgfiles)//2:
                ax[i].set_ylabel('SHG (pW)')

            if i == len(self.shgfiles) - 1:
                ax[i].set_xlabel('Wavelength (nm)')
                ax[i].get_xaxis().set_visible(True)

        if save==False:
            plt.show()
        else:
            my_path = self.filepath
            fig.savefig(my_path+'/singleplotSHG_'+str(j)+'.png', bbox_inches="tight")
            j += 1
            plt.close()

    def plot_ir_single(self, yautoscale=False, ylogscale=False, xrange=None, yrange=None, color=None, pattern=None,
                       sx=None, sy=None, linewidth=None, save=False):
        if color == None:
            color = 'b'
        if pattern == None:
            pattern = '-'
        if sx == None:
            sx = 8
        if sy == None:
            sy = 6
        if linewidth == None:
            linewidth = 1.0
        j = 0
        fig, ax = plt.subplots(len(self.irfiles), 1, figsize=(sx,sy))

        for i in range(len(self.irfiles)):
            ax[i].plot(self.ir_wavelengths[i], self.ir_responses[i], color + pattern, linewidth=linewidth)
            ax[i].get_xaxis().set_visible(False)
            # ax[i].get_yaxis().set_visible(False)
            if xrange == None:
                pass
            else:
                ax[i].set_xlim(xrange[0], xrange[1])

            if yrange == None:
                pass
            else:
                ax[i].set_ylim(yrange[0], yrange[1])

            if ylogscale == False:
                ax[i].set_yscale("linear")
            else:
                ax[i].set_yscale("log")

            if yautoscale == False:
                pass
            else:  # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
                if xrange == None:
                    raise Exception("no xrange set")
                delta = self.ir_wavelengths[i][1] - self.ir_wavelengths[i][0]
                index_min = int((xrange[0] - self.ir_wavelengths[i][0]) // delta)
                index_max = int((xrange[1] - self.ir_wavelengths[i][0]) // delta)
                y_min = np.amin(self.ir_responses[i][index_min:index_max])
                y_max = np.amax(self.ir_responses[i][index_min:index_max])
                ax[i].set_ylim(y_min*0.9, y_max * 1.1)

            if i == len(self.irfiles)//2:
                ax[i].set_ylabel('Transmission (a.u.)')

            if i == len(self.irfiles)-1:
                ax[i].set_xlabel('Wavelength (nm)')
                ax[i].get_xaxis().set_visible(True)
                # ax[i].get_yaxis().set_visible(True)

        if save==False:
            plt.show()
        else:
            my_path = self.filepath
            fig.savefig(my_path+'\\singleplotIR_'+str(j)+'.png', bbox_inches="tight")
            j += 1
            plt.close()

    def plot_ir_shg_multi(self, ir_yautoscale=False, shg_yautoscale=False, ir_ylogscale=False, shg_ylogscale=False,
                          xrange=None, ir_yrange=None, shg_yrange=None, ircolor=None, shgcolor=None, irpattern=None,
                          shgpattern=None, sx=None, sy=None, ir_linewidth=None, shg_linewidth=None, title=False,
                          save=False):
        '''This method is used for plotting many separate IR/SHG comparisons, there is a separate method
        (plot_ir_shg_single) that plots all in one figure, this method also assumes that your IR and corresponding
        SHG files are the same in name other than the suffix'''
        assert len(self.irfiles) == len(self.shgfiles)
        if ircolor == None:
            ircolor = 'b'
        if irpattern == None:
            irpattern = '-'
        if shgcolor == None:
            shgcolor = 'r'
        if shgpattern == None:
            shgpattern = '-'
        if sx == None:
            sx = 8
        if sy == None:
            sy = 6
        if ir_linewidth == None:
            ir_linewidth = 1.0
        if shg_linewidth == None:
            shg_linewidth = 1.0
        j = 0

        for i in range(len(self.irfiles)):
            fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(sx,sy))

            ax1.plot(self.ir_wavelengths[i], self.ir_responses[i], ircolor + irpattern, linewidth=ir_linewidth)
            ax2.plot(self.shg_wavelengths[i], self.shg_responses[i], shgcolor + shgpattern, linewidth=shg_linewidth)

            if xrange == None:
                pass
            else:
                ax1.set_xlim(xrange[0], xrange[1])
                ax2.set_xlim(xrange[0], xrange[1])

            if title == False:
                pass
            else:
                ax1.set_title(self.irfiles[i])

            if ir_yrange == None:
                pass
            else:
                ax1.set_ylim(ir_yrange[0], ir_yrange[1])

            if shg_yrange == None:
                pass
            else:
                ax2.set_ylim(shg_yrange[0], shg_yrange[1])

            if ir_ylogscale == False:
                ax1.set_yscale("linear")
            else:
                ax1.set_yscale("log")

            if shg_ylogscale == False:
                ax2.set_yscale("linear")
            else:
                ax2.set_yscale("log")

            if ir_yautoscale == False:
                pass
            else:  # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
                if xrange == None:
                    raise Exception("no xrange set")
                delta = self.ir_wavelengths[i][1] - self.ir_wavelengths[i][0]
                index_min = int((xrange[0] - self.ir_wavelengths[i][0]) // delta)
                index_max = int((xrange[1] - self.ir_wavelengths[i][0]) // delta)
                y_min = np.amin(self.ir_responses[i][index_min:index_max])
                y_max = np.amax(self.ir_responses[i][index_min:index_max])
                ax1.set_ylim(y_min*0.9, y_max * 1.1)

            if shg_yautoscale == False:
                pass
            else:  # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
                if xrange == None:
                    raise Exception("no xrange set")
                delta = self.shg_wavelengths[i][1] - self.shg_wavelengths[i][0]
                index_min = int((xrange[0] - self.shg_wavelengths[i][0]) // delta)
                index_max = int((xrange[1] - self.shg_wavelengths[i][0]) // delta)
                y_min = np.amin(self.shg_responses[i][index_min:index_max])
                y_max = np.amax(self.shg_responses[i][index_min:index_max])
                ax2.set_ylim(0, y_max * 1.1)

            ax1.set_ylabel('Transmission (a.u.)')
            ax2.set_ylabel('SHG (pW)')
            ax2.set_xlabel('Wavelength (nm)')

            if save==False:
                plt.show()
            else:
                my_path = self.filepath
                fig.savefig(my_path+'/ir_shg_'+str(j)+'.png', bbox_inches="tight")
                j += 1
                plt.close()

    def get_irpeaks(self,max_width=0.03, min_dist=0.05, prominence=0.1):

        for i in range(len(self.irfiles)):
            delta = abs(self.ir_wavelengths[i][1]-self.ir_wavelengths[i][0])
            if delta == 0:
                delta = (self.ir_wavelengths[i][-1]-self.ir_wavelengths[i][0]) / len(self.ir_wavelengths[i])
            actual_distance = min_dist//delta
            actual_width = max_width//delta
            total_length = len(self.ir_wavelengths[i])
            tenths = total_length//10
            index = 0
            self.irpeaks[i] = []
            self.irpeaksinfo[i] = {}
            # split entire data array into ten groups so that the local peaks correspond to more global maximums
            # for j in range(10):
            #     maxpoint = np.amax(self.ir_responses[i][index:index+tenths])
            #     test_array = (maxpoint * 1.1 - self.ir_responses[i][index:index + tenths])
            #     index += tenths

                # temp_peaks, temp_peaksinfo = find_peaks(test_array, distance=actual_distance,height=0,
                #                                               prominence=prominence, width=(None, actual_width))
                # self.irpeaks[i] = np.append(self.irpeaks[i],temp_peaks+index)
                # # self.irpeaks[i].append(temp_peaks+index)
                # for key in temp_peaksinfo:
                #     # pdb.set_trace()
                #     if key in self.irpeaksinfo:
                #         self.irpeaksinfo[i][key] = np.append(self.irpeaksinfo[i][key], temp_peaksinfo[key])
                #     else:
                #         self.irpeaksinfo[i][key] = []
                #         self.irpeaksinfo[i][key] = np.append(self.irpeaksinfo[i][key],temp_peaksinfo[key])
                # # self.irpeaksinfo[i].append(temp_peaksinfo)
                # # pdb.set_trace()
                # self.irpeaks[i] = self.irpeaks[i].astype(int)

            maxpoint = np.amax(self.ir_responses[i])  # np.amax(np.log10(self.ir_responses[i]))
            test_array = (maxpoint * 1.1 - np.log10(self.ir_responses[i]))
            self.irpeaks[i], self.irpeaksinfo[i] = find_peaks(test_array, distance=actual_distance,height=0,
                                                              prominence=prominence, width=(None, actual_width))
            # pdb.set_trace()
            # pdb.set_trace()

    def get_shgpeaks(self,max_width=0.03, min_dist=0.05, prominence=0.1, height=10):

        for i in range(len(self.shgfiles)):
            delta = abs(self.shg_wavelengths[i][1]-self.shg_wavelengths[i][0])
            actual_distance = min_dist//delta
            actual_width = max_width//delta
            # test_array = np.log10(self.shg_responses[i])
            self.shgpeaks[i], self.shgpeaksinfo[i] = find_peaks(self.shg_responses[i], height=height, distance=actual_distance,
                                                                prominence=prominence, width=(None, actual_width))
            # pdb.set_trace()
            # print((self.shgpeaks[i]))

    def plot_shg_peaks(self, ylogscale=False, xrange=None, yrange=None,
                       color=None, pattern=None, linewidth=None, save=False,
                       sx=None, sy=None, shg_height=10):
        if color == None:
            color = 'r'
        if pattern == None:
            pattern = 'x'
        if linewidth == None:
            linewidth = 1.0
        if sx == None:
            sx = 8
        if sy == None:
            sy = 6

        self.get_shgpeaks(height=shg_height)

        plt.figure(self.figure_index, figsize=(sx,sy))
        for i in range(len(self.shgfiles)):
            xarray = np.full(len(self.shgpeaks[i]),i)
            xarray_power = 1*10**(xarray/10)
            plt.plot(np.log10(xarray_power), np.log10(self.shg_responses[i][self.shgpeaks[i]]),
                     color + pattern, linewidth=linewidth)

        x_slope2 = np.asarray([0, 0.5, 1])
        y_slope2 = 1.85*x_slope2+1.6
        plt.plot(x_slope2,y_slope2,'b-')

        if xrange == None:
            pass
        else:
            plt.xlim(xrange[0], xrange[1])

        if yrange == None:
            pass
        else:
            plt.ylim(yrange[0], yrange[1])

        if ylogscale == False:
            plt.yscale("linear")
        else:
            plt.yscale("log")

        plt.ylabel('SHG log10(pW)')
        plt.xlabel('power log10(mW)')

        if save == False:
            plt.show()
        else:
            my_path = self.filepath
            plt.savefig(my_path + '/plotSHGpeaks_' + '.png', bbox_inches="tight", dpi=400)
            plt.close()
        # self.figure_index += 1

    def plot_ir(self, index=None, yautoscale=False, ylogscale=False, xrange=None, yrange=None, color=None, pattern=None,
                        sx=None, sy=None, linewidth=None):
        if index == None:
            raise Exception("no index identified")
        if color == None:
            color = 'b'
        if pattern == None:
            pattern = '-'
        if sx == None:
            sx = 8
        if sy == None:
            sy = 8
        if linewidth == None:
            linewidth = 1.0
        fig, ax = plt.subplots(3, 1, figsize=(sx,sy))
        j = 0
        ax[0].plot(self.ir_wavelengths[index], self.ir_responses[index], color + pattern, linewidth=linewidth)
        ax[0].get_xaxis().set_visible(False)
        if xrange == None:
            pass
        else:
            ax[0].set_xlim(xrange[0], xrange[1])

        if yrange == None:
            pass
        else:
            ax[0].set_ylim(yrange[0], yrange[1])

        if ylogscale == False:
            ax[0].set_yscale("linear")
        else:
            ax[0].set_yscale("log")

        if yautoscale == False:
            pass
        else:  # This is needed because matplotlib doesn't rescale based on your xrange, it automatically scales to the global max/min
            if xrange == None:
                raise Exception("no xrange set")
            delta = self.ir_wavelengths[index][1] - self.ir_wavelengths[index][0]
            index_min = int((xrange[0] - self.ir_wavelengths[index][0]) // delta)
            index_max = int((xrange[1] - self.ir_wavelengths[index][0]) // delta)
            y_min = np.amin(self.ir_responses[index][index_min:index_max])
            y_max = np.amax(self.ir_responses[index][index_min:index_max])
            ax[0].set_ylim(y_min*0.9, y_max * 1.1)

            ax[0].set_ylabel('Transmission (a.u.)')
            ax[0].set_xlabel('Wavelength (nm)')
        return fig,ax

    def distances(self, i, peakpoint, fsr_range, n):
        d = []
        d_array = []
        wavelengths = []
        new_d_test = []
        new_d = []

        for j in range(n):
            d.append(self.ir_wavelengths[i][self.irpeaks[i][peakpoint + j + 1]] - \
                   self.ir_wavelengths[i][self.irpeaks[i][peakpoint]])

            new_d_test.append(3e8*(1/self.ir_wavelengths[i][self.irpeaks[i][peakpoint]] - 1/self.ir_wavelengths[i][self.irpeaks[i][peakpoint + j + 1]]))

            if fsr_range[0]<d[j]<fsr_range[1]:
                d_array.append(d[j])
                new_d.append(new_d_test[j])
                wavelengths.append(self.ir_wavelengths[i][self.irpeaks[i][peakpoint]])
                self.index_track.append(peakpoint)  # important because it lets you plot easier after mode analysis

        return np.asarray(d_array), np.asarray(wavelengths), np.asarray(new_d)

    def Qfit(self, fsr_range=None, ng=2.365, radius=50, n_mode=4, wavelength=1550,
             save=False, prominence=0.1, max_width=0.03, min_dist=0.05, xrange=None):

        if not xrange==None:
            for i in self.ir_responses:
                new_wavelengths = []
                new_responses = []
                begin = np.where(self.ir_wavelengths[i]==xrange[0])
                end = np.where(self.ir_wavelengths[i]==xrange[1])
                new_wavelengths.append(self.ir_wavelengths[i][begin[0][0]:end[0][0]])
                new_responses.append(self.ir_responses[i][begin[0][0]:end[0][0]])
                self.ir_wavelengths[i] = new_wavelengths[0]
                self.ir_responses[i] = new_responses[0]

        def fsr_slope_estimate(ng,radius,wavelength):
            wl_max = wavelength/1000+0.07
            wl_min = wavelength/1000-0.07
            return ((1/((1/wl_max)-1/(2*np.pi*radius*ng))-(wl_max))- \
                   (1/((1/wl_min)-1/(2*np.pi*radius*ng))-(wl_min)))/(wl_max-wl_min)

        def get_real_ng(wavelengths, fsr, radius):
            calc_ng=[]
            for i in range(len(wavelengths)):
                calc_ng.append((wavelengths[i]/1000)**2/(2*np.pi*radius*fsr[i]))
            new_ng = np.asarray(calc_ng)
            mean_ng = round(1000*np.mean(new_ng),3)
            return mean_ng

        if fsr_range==None:
            fsr = 1000*1.56**2/(ng*2*np.pi*radius)
            fsr_range = [fsr-0.5,fsr+0.5]
            sr = fsr_slope_estimate(ng=ng,radius=radius,wavelength=wavelength)
            slope_range = [sr-0.001,sr+0.001]
        else:
            sr = fsr_slope_estimate(ng=ng, radius=radius, wavelength=wavelength)
            slope_range = [sr - 0.001, sr + 0.001]

        self.get_irpeaks(max_width=max_width, prominence=prominence,min_dist=min_dist)

        def lorentz(x, I, x0, w, off):  # amp, cent, width, offset. Note that 2*w is FWHM
            return -I * ((np.power(w, 2)) / (np.power((x - x0), 2) + np.power(w, 2))) + off

        def splitlorentz(x, I, I2, x1, delta, w, off):  # amp1, amp2, cent, delta_x, width, offset.
            return -I * ((np.power(w, 2)) / (np.power((x - x1 - delta), 2) + np.power(w, 2))) + -I2 * (
            (np.power(w, 2)) / (np.power((x - x1 + delta), 2) + np.power(w, 2))) + off

        def rmse(prediction, target):  # RMSE function
            return np.sqrt(np.mean((prediction - target) ** 2))

        for i in range(len(self.irfiles)):
            self.index_track = []
            Q = np.zeros(len(self.irpeaks[i]));ext = np.zeros(len(self.irpeaks[i]))
            res = len(self.ir_wavelengths[i])/(np.amax(self.ir_wavelengths[i])-np.min(self.ir_wavelengths[i]))
            peak_lw = 0.005*res
            fit_bw = 3*peak_lw

            fig, ax = self.plot_ir(i,ylogscale=True)
            # pdb.set_trace()
            ax[0].plot(self.ir_wavelengths[i][self.irpeaks[i]], self.ir_responses[i][self.irpeaks[i]], "x")
            axdb = ax[1].twinx()

            diff_array = np.asarray([])
            wl_array = np.asarray([])
            new_diff_array = np.asarray([])

            for j in range(len(self.irpeaks[i])):
                # this loops through every peak and does the fitting for each one
                lb = int(self.irpeaks[i][j]-fit_bw); ub = int(self.irpeaks[i][j] + fit_bw)
                if lb <0:
                    lb = int(0)
                if ub> len(self.ir_wavelengths[i]):
                    ub = len(self.ir_wavelengths[i])

                pkwav = self.ir_wavelengths[i][lb:ub]; pktrans = self.ir_responses[i][lb:ub]

                init_I = np.amax(pktrans) - np.amin(pktrans)
                init_w = np.asarray(self.irpeaksinfo[i]['widths'][j]/res)
                init_x0 = np.asarray(self.ir_wavelengths[i][self.irpeaks[i][j]])
                init_off = np.mean(pktrans)
                init_vals = [init_I, init_x0, init_w, init_off]

                try:
                    lorentz_fit, covar = curve_fit(lorentz, pkwav,
                                           pktrans, p0=init_vals)
                    fitvals = lorentz(pkwav, *lorentz_fit)  # line of fitted peak
                    fitvals[fitvals < min(pktrans)] \
                        = min(pktrans)
                    Q[j] = np.abs((lorentz_fit[1] / lorentz_fit[2]) / 2000)
                    ext[j] = np.abs((10 * np.log10(np.min(pktrans) /
                                                   np.max(pktrans))))
                    ax[0].plot(pkwav, fitvals, c='C1')  # plot the peak fit
                    ax[0].text(lorentz_fit[1], self.ir_responses[i][self.irpeaks[i][j]] * 0.9, "%dk" % Q[j], fontsize=10)
                except (RuntimeError, ValueError):
                    continue

                try:
                    val_1, val_2, val_3 = self.distances(i, j, fsr_range, n=n_mode)
                    diff_array = np.append(diff_array,val_1)
                    wl_array = np.append(wl_array,val_2)
                    new_diff_array = np.append(new_diff_array,val_3)
                except IndexError:
                    continue

            mode_array = diff_array.flatten()
            mode_wl = wl_array.flatten()
            new_mode_array = new_diff_array.flatten()
            self.index_modes = {}
            update_array, update_wl, update_array2, update_wl2 = \
                self.mode_analyze(mode_array,mode_wl,n=n_mode,slope_range=slope_range)
            ax[0].set_ylabel("Transmission (a.u.)")
            ax[0].get_xaxis().set_visible(True)
            ax[0].set_title('Median Q ' + str(np.around(np.median(Q))) + 'k' + '  Median Extinction ' +
                                                            str(np.around(np.median(ext), decimals = 1)) + ' dB')
            axdb.plot(self.ir_wavelengths[i][self.irpeaks[i]], ext, 'o', c='C0', zorder = 1)
            ax[1].plot(self.ir_wavelengths[i][self.irpeaks[i]], Q, 'o', c='C1', zorder = 10)
            ax[1].set_ylabel('Loaded Q/1000', color='C1')
            axdb.set_ylabel('Extinction (dB)', color='C0')
            fig.subplots_adjust(hspace=0.3, bottom=0.1)

            ax[2].plot(mode_wl,mode_array,'x')
            # ax[2].plot(update_wl, update_array, 'o')
            ax[2].set_ylabel("FSR (nm)")
            ax[2].set_xlabel("Wavelength (nm)")

            color = iter(cm.rainbow(np.linspace(0,1,len(update_array2))))
            for b in range(len(update_array2)):
                c = next(color)
                real_ng=get_real_ng(wavelengths=update_wl2[b],fsr=update_array2[b],radius=radius)
                ax[2].plot(update_wl2[b],update_array2[b],'o',c=c,label='n_g'+ str(b) + '='+str(real_ng))
                ax[0].plot(update_wl2[b],self.ir_responses[i][self.irpeaks[i][self.index_modes[b]]],'o',c=c)

            ax[2].legend(loc='upper left')

            if save == False:
                plt.show()
            else:
                my_path = self.filepath
                fig.savefig(self.irfiles[i] + '.png', bbox_inches="tight", dpi=400)
                # fig.clf()
                plt.close()

    def mode_analyze(self, diff_array, wl_array, n, slope_range=[0.0032,0.0054]):
        # this code filters out the noisy nearest neighbour distances by estimated slope
        # the better your guess of what the group index is at a certain wavelength, the better the filter is
        update_diff = []
        update_wl = []
        for i in range(len(wl_array)):
            for j in range(n*n):
                try:
                    if wl_array[i+1+j] == wl_array[i]:
                        continue
                    else:
                        test = ((diff_array[i+1+j]-diff_array[i])/(wl_array[i+1+j]-wl_array[i]))
                        if slope_range[0] < test < slope_range[1]:
                            update_diff.append(diff_array[i])
                            update_wl.append(wl_array[i])
                            break
                except IndexError:
                    continue

        update_diff2 = {}
        update_wl2 = {}
        update_index = {}
        def find_slope(i,n):
            signal = False
            index = 0
            for j in range(n*n):
                try:
                    if wl_array[i+1+j] == wl_array[i]:
                        continue
                    else:
                        test = ((diff_array[i+1+j]-diff_array[i])/(wl_array[i+1+j]-wl_array[i]))
                        if slope_range[0] < test < slope_range[1]:
                            signal = True
                            index = i + 1 + j
                            return signal,index
                except IndexError:
                    continue
            return signal,index

        update_list = []
        m=0
        for i in range(len(wl_array)):
            signal = True
            index = i
            temp_diff = []
            temp_wl = []
            temp_index = []
            if i in update_list:
                continue
            else:
                try:
                    while signal == True:
                        signal, index = find_slope(i=index,n=n)
                        if index in update_list:
                            break
                        update_list.append(index)
                        temp_index.append(self.index_track[index])
                        temp_diff.append(diff_array[index])
                        temp_wl.append(wl_array[index])
                    if len(temp_diff)>=3:
                        update_diff2[m] = temp_diff
                        update_wl2[m] = temp_wl
                        self.index_modes[m] = temp_index
                        m += 1
                except IndexError:
                    break

        return np.asarray(update_diff), np.asarray(update_wl), update_diff2, update_wl2
