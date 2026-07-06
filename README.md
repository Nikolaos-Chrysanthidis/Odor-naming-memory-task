# Odor-naming
All the supporting data, simulation code, synaptic plasticity implementation, and data analysis algorithms used in the experiments can be found in this repository.

README file for the BCPNNSim Threaded API (Version 0.9.6)
-----------------------------------------------------------------------------------

Aims of this code
This code is intended to allow realization and simulation of all different aspects of the BCPNN model. Thus, it allows to build abstract computational networks as well as biologically detailed
networks of cortex with leaky integrate-and-fire model neurons and conductance based synapses. It was initially intended as a template for development of FPGA and ASIC. The plan is to develop it for a wider use, e.g. for parallel execution on clusters. Currently, the code is threaded but a cluster parallel version is planned.

Components of code and distribution
This API allows a user to specify a network built of Population and Projection components. A population is an array of neural units and a projection is a (possibly sparse) matrix of connections between a source and a target population.

The basic classes are Pop/PopH and Prj/PrjH with derived classes like HCU/BCU and BCC. The helper class Logger is used for logging and printing state variables. The helper class Parseparam is used to interpret the parameter files (named *.par) and makes easy to explore
different parameter settings. The helper class Timer is used for easy timing of code sections. 


Olfaction Modelling 

The files here contain the most recent and necessary folders needed to run the olfaction model. 
To get started with running simulations: 

Follow the instructions below to set up the bcpnn simulator.
You should have a “build” folder within /works. Importantly, the build folder should contain apps/olflang/. This folder contains the executable for the model cpp and any weights,biases or network state logging files get stored here at the end of the simulation.
The major files to be concerned with for running the olfaction model are located in works/apps/olflang (not to be confused by works/build/apps/olflang/)
 
olflangmain.cpp (contains the cpp code for the model)
olflangmain1.par (parameter file for the model)

Make sure the filepaths in the code files mentioned above match your local directory.
To run the model go to works/build/olflang and type “make olflangmain1 && ./olflangmain1” 



Important Pointers 
If you want to run the simulations with preloaded weights and biases, you must copy the .bin files for the weights and biases to works/build/apps/olflang. 

Before you run simulations with a set of preloaded weights and biases, make sure to check that the parameter runflag in olflangmain1.par is set to “preload_localasso” . Also make sure that the preloadBW() function in olflangmain.cpp is referring to the correct file names for the preloaded weights and biases. Also, the file olflangmain1.par lists all the the parameters used in the simulations.

The way in which the odors and labels are associated is saved as textfiles starting with the word patstat . The simulations used the file patstat_si_nclusters4_topdescs.txt


Installation procedures using CMake:

1) Ensure that you have CMake version 3.10+ installed.
2) Create a build directory, for example inside the BCPNNSim/works directory by typing: mkdir build
3) Enter that directory: cd build
4) CMAKE from that build directory onto the top-level directory that contains the CMakelists.txt (in this case, "BCPNNSim/works").
   Also, you may provide an installation path.
   For example, type: cmake ../. -DCMAKE_INSTALL_PREFIX:PATH=../.
   (Note that above example will install everything into /usr and requires root)
5) type: make && make install
6) add bin folder to your path: export PATH="<your home catalog>/MyPrograms/BCPNNSim/works/bcpnn/bin:$PATH"


