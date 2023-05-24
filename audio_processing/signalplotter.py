
import numpy as np
import matplotlib.pyplot as plt

class signalplotter:

    plotduration: int
    samplerate: int
    voiceblocksperplot: int
    voiceblockspersecond: int
    plotlength: int

    # an array that says how many graphs are in which plot
    plotinfos: np.array

    # an array that saves the plotlines to be changed
    plotlines: np.array

    # an array that stores the values that are plotted
    plotvalues: np.array

    fig1: plt.figure


    # setting up the plotter
    def __init__(self, plotduration: int, samplerate: int, voiceblocksperplot: int, voiceblockspersecond: int, plotinfos: np.array, fig:plt.figure) -> np.array:
        print("Initializing plotter")
        self.plotduration = plotduration
        self.samplerate = samplerate
        self.voiceblocksperplot = int(voiceblocksperplot)
        self.voiceblockspersecond = voiceblockspersecond
        self.plotlength = int(self.plotduration*self.samplerate)
        self.plotinfos = plotinfos
        self.fig1 = fig
        self.setup_plots()
        print("Plotter rdy")
        
    
    # setting up hte plots
    def setup_plots(self):
        print("Setting up the plots")
        x = np.arange(self.plotduration*self.samplerate)/self.samplerate
        x_voice = np.arange(self.voiceblocksperplot)/self.voiceblockspersecond
        #print(len(x))
        #print(len(x_voice))

        plt.ion()
        plt1 = self.fig1.add_subplot(411)
        plt2 = self.fig1.add_subplot(412)
        plt3 = self.fig1.add_subplot(413)
        plt4 = self.fig1.add_subplot(414)
        plt1v = plt1.twinx()
        plt2w = plt2.twinx()
        plt3w = plt3.twinx()
        plt4v = plt4.twinx()

        plt1.set_ylabel("raw data")
        plt2.set_ylabel("filtered data")
        plt3.set_ylabel("automatic gained data")
        plt1v.set_ylabel("voice activity")
        plt2w.set_ylabel("worddetection")
        plt4.set_ylabel("words")
        plt3w.set_ylabel("convolved worddetection")
        plt4v.set_ylabel("vocoded words")

        plt1.axis([0, self.plotduration, -2, 2])
        plt2.axis([0, self.plotduration, -2, 2])
        plt3.axis([0, self.plotduration, -1, 1])
        plt4.axis([0, self.plotduration, -1, 1])
        plt1v.axis([0, self.plotduration, -90, -10])
        plt2w.axis([0, self.plotduration, -1, 2])
        plt3w.axis([0, self.plotduration, -1, 2])
        plt4v.axis([0, self.plotduration, -1, 1])

        printblockraw = np.zeros(self.plotlength)
        printblockfiltered = np.zeros(self.plotlength)
        printblockgained = np.zeros(self.plotlength)
        printvoiceactivity = np.zeros(self.voiceblocksperplot)
        printworddetection = np.zeros(self.plotlength)
        printwords = np.zeros(self.plotlength)
        printworddetection2 = np.zeros(self.plotlength)
        printvocoded = np.zeros(self.plotlength)

        self.plotvalues = np.array([[printblockraw, printvoiceactivity], [printblockfiltered, printworddetection], [printblockgained, printworddetection2], [printwords, printvocoded]], dtype=object)

        linenofilter, = plt1.plot(x, printblockraw, 'b-')
        linewithfiler, = plt2.plot(x, printblockfiltered, 'b-')
        linewithgain, = plt3.plot(x, printblockgained, 'b-')
        lineactivity, = plt1v.plot(x_voice, printvoiceactivity, 'r-')
        lineworddetection, = plt2w.plot(x, printworddetection, 'r-')
        linewords, = plt4.plot(x, printwords, 'b-')
        lineworddetection2, = plt3w.plot(x, printworddetection2, 'r-')
        linevocoded, = plt4v.plot(x, printvocoded, 'r-')

        self.plotlines = np.array([[linenofilter, lineactivity], [linewithfiler, lineworddetection], [linewithgain, lineworddetection2], [linewords, linevocoded]], dtype=object)
        print("Plots are rdy to go")

    
    # updating the plots
    def update_lines(self, data: np.array):
        print("uptdating lines")
        for i in range(self.plotinfos.shape[0]):
            for j in range(self.plotinfos[i]):
                self.plotvalues[i][j] = np.append(self.plotvalues[i][j][data[i][j].shape[0]:], data[i][j])
                self.plotlines[i][j].set_ydata(self.plotvalues[i][j])
        
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        print("updated lines")
