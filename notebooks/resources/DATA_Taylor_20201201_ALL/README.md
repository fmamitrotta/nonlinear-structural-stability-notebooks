Airfoil Profiles and Transonic Small-Disturbance Theory/Integral Boundary Layer Airfoil Data for the Low-Fidelity Common Research Model

The files contained in this folder are taken from the following webpage:
https://digitalcommons.usu.edu/all_datasets/125/

Name and contact information of PI:
   Jeffrey D. Taylor
   Utah State University
   4130 Old Main Hill Logan, Utah 84322-4130
   jeffdtaylor3891@gmail.com
   ORCiD ID: 0000-0003-3719-0955

Name and contact information of Co-PI:
   Douglas Hunsaker
   Utah State University
   4130 Old Main Hill Logan, Utah 84322-4130
   doug.hunsaker@usu.edu
   ORCiD ID: 0000-0001-8106-7466

Funding source: NASA, National Aeronautics and Space Administration 80NSSC18K1696

Description of Data Collection Process:
   The airfoil profile files contain the x and y coordinates of the airfoil surface, normalized by the local chord for the airfoil sections located at 0, 10, 15, 20, 25, 30, 35, 37, 40, 
   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, and 100% semispan of the Common Research Model wing. Data were extracted from a CAD file of the uCRM-9 wing geometry described in 
   [Brooks, T. R., Kenway, G. K. W., and Martins, J. R. R. A., “Benchmark Aerostructural Models for the Study of Transonic Aircraft Wings,” AIAA Journal, Vol. 56, No. 7, July 2018 pp. 2840- 2855. (doi:10.2514/1.J056603)]
   and available at https://data.mendeley.com/datasets/gpk4zn73xn/1 (doi:10.17632/gpk4zn73xn.1)

   The airfoil database files contain the Lift, Drag, and Moment Coefficient data for the Low-Fidelity Common Research Model wing airfoil sections. Data were collected for the airfoil sections located at 0, 10, 15, 20, 25, 30, 35, 37, 40, 
   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, and 100% semispan, as a function of flap deflection of a parabolic flap spanning the aft 30% of the chord, angle of attack, Reynolds number, and Mach number. 
   The data were obtained using the Transonic Small-Disturbance Theory/Integral Boundary layer method from [Fujiwara, G. E. C., Chaparro, D., and Nguyen, N., “An Integral Boundary Layer Direct Method Applied to 
   2D Transonic Small-Disturbance Equations,” AIAA 2016-3568, 34th AIAA Applied Aerodynamics Conference, Washington, D. C., 13-17 June 2016. (doi:10.2514/6.2016-3568)] 
   *See [Taylor, J.D. and Hunsaker D.F., "Characterization of the Common Research Model Wing for Low-Fidelity Aerostructural Analysis," AIAA SciTech 2021 Virtual Forum, 11-15 & 18-22 January 2021] 
   for additional details about the data collection process.

Description of dataset files:
   The dataset includes 42 txt files. Twenty-one of these files contain the x and y coordinates of the 21 airfoil sections at the spanwise locations listed above (with zero flap deflection). 
   The other 21 files contain airfoil data with the corresponding flap deflection, angle of attack, Reynolds number, and Mach number for each 
   of the 21 airfoil sections (listed above) of the Low-Fidelity Common Research Model. The files are named as follows:
   
   airfoil profiles:
	uCRM-9_wr0_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 0% semispan (wing root)
	uCRM-9_wr10_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 10% semispan
	uCRM-9_wr15_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 15% semispan
	uCRM-9_wr20_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 20% semispan
	uCRM-9_wr25_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 25% semispan
	uCRM-9_wr30_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 30% semispan
	uCRM-9_wr35_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 35% semispan
	uCRM-9_wr37_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 37% semispan (Yehudi Break)
	uCRM-9_wr40_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 40% semispan
	uCRM-9_wr45_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 45% semispan
	uCRM-9_wr50_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 50% semispan
	uCRM-9_wr55_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 55% semispan
	uCRM-9_wr60_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 60% semispan
	uCRM-9_wr65_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 65% semispan
	uCRM-9_wr70_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 70% semispan
	uCRM-9_wr75_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 75% semispan
	uCRM-9_wr80_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 80% semispan
	uCRM-9_wr85_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 85% semispan
	uCRM-9_wr90_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 90% semispan
	uCRM-9_wr95_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 95% semispan
	uCRM-9_wr100_profile.txt: x and y coordinates, normalized by the local wing chord, for the airfoil section located at 100% semispan (wing tip)   

    airfoil_data:
	   wr_0_dataset.txt: airfoil data for the airfoil section located at 0% semispan (wing root)
	   wr_10_dataset.txt: airfoil data for the airfoil section located at 10% semispan
	   wr_15_dataset.txt: airfoil data for the airfoil section located at 15% semispan
	   wr_20_dataset.txt: airfoil data for the airfoil section located at 20% semispan
	   wr_25_dataset.txt: airfoil data for the airfoil section located at 25% semispan
	   wr_30_dataset.txt: airfoil data for the airfoil section located at 30% semispan
	   wr_35_dataset.txt: airfoil data for the airfoil section located at 35% semispan
	   wr_37_dataset.txt: airfoil data for the airfoil section located at 37% semispan (Yehudi Break)
	   wr_40_dataset.txt: airfoil data for the airfoil section located at 40% semispan
	   wr_45_dataset.txt: airfoil data for the airfoil section located at 45% semispan
	   wr_50_dataset.txt: airfoil data for the airfoil section located at 50% semispan
	   wr_55_dataset.txt: airfoil data for the airfoil section located at 55% semispan
	   wr_60_dataset.txt: airfoil data for the airfoil section located at 60% semispan
	   wr_65_dataset.txt: airfoil data for the airfoil section located at 65% semispan
	   wr_70_dataset.txt: airfoil data for the airfoil section located at 70% semispan
	   wr_75_dataset.txt: airfoil data for the airfoil section located at 75% semispan
	   wr_80_dataset.txt: airfoil data for the airfoil section located at 80% semispan
	   wr_85_dataset.txt: airfoil data for the airfoil section located at 85% semispan
	   wr_90_dataset.txt: airfoil data for the airfoil section located at 90% semispan
	   wr_95_dataset.txt: airfoil data for the airfoil section located at 95% semispan
	   wr_100_dataset.txt: airfoil data for the airfoil section located at 100% semispan (wing tip)
    
	

Column headings of data files:
  dataset files:
   trailing_flap_deflection: the deflection, in radians, of the parabolic trailing-edge flap (positive down)
   alpha: the two-dimensional angle of attack, in radians, of the airfoil section. 
   rey: the local Reynolds number of the airfoil section. 
   mach: the local Mach number of the airfoil section.
   trailing_flap_fraction: fraction of the chord (measured from the trailing edge) taken up by the trailing-edge flap. 
   CL: the two-dimensional section lift coefficient.
   CD: the two-dimensional section drag coefficient.
   CM: the two-dimensional section moment coefficient.

  profile files:
    Data in the profile files are organized in two columns. The first column contains the x coordinates of the airfoil surface, normalized by the local chord.
    The second column contains the y coordinates of the airfoil surface, normalized by the local chord. Data begin at the trailing edge, along the upper surface of the airfoil
    to the leading edge, then along the bottom surface of the airfoil to the trailing edge. 

Publications that cite or use this data:
   Data referenced in: Taylor, J.D. and Hunsaker D.F., "Characterization of the Common Research Model Wing for Low-Fidelity Aerostructural Analysis," AIAA SciTech 2021 Virtual Forum, 11-15 & 18-22 January 2021.
