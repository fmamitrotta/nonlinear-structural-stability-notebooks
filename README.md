# Repository Overview #

This repository contains the research material that I developed for my PhD research on aeroelastic optimization with nonlinear structural stability constraints. This research material includes mainly Jupyter notebooks that cover the progression of my understanding of nonlinear structural stability and its application to aeroelastic optimization. The notebooks are written in Python and use the Finite Element Analysis software MSC Nastran as structural solver, leveraging the [pyNastran](https://github.com/SteveDoyle2/pyNastran) package to interact with it. The notebooks are also available in a static format on [nbviewer](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks). Other resources include utility python modules, third-party MSC Nastran input files, and some figures that I created for the notebooks. Originally, the repository included also the MSC Nastran input and output files of the analyses related to the notebooks, but they are in the process of being stored in the University of Bristol's data.bris research data repository due to their size. The link to the data.bris repository will be provided here once the process is completed.

## Citation

If you use any of the code or of the other resources in this repository, please cite one of the following papers:

F. M. A. Mitrotta, A. Pirrera, T. Macquart, J. E. Cooper, A. P. do Prado and P. H. Cabral. _"Proof-of-concept of a Nonlinear Structural Stability Constraint for Aeroelastic Optimizations"_, 2023, RAeS 8th Aircraft Structural Design Conference

F. M. A. Mitrotta, A. Pirrera, T. Macquart, J. E. Cooper, A. P. do Prado and P. H. Cabral. [_"Development of a Nonlinear Structural Stability Constraint for Aeroelastic Optimization"_](https://doi.org/10.2514/6.2024-2412), 2024, AIAA SciTech Forum

F. M. A. Mitrotta, A. Pirrera, T. Macquart, J. E. Cooper, A. P. do Prado and P. H. Cabral. [_"Influence of Load Introduction Method on Wingbox Optimization with Nonlinear Structural Stability Constraints"_](https://conf.ifasd2024.nl/proceedings/documents/31.pdf), 2024, International Forum on Aeroelasticity and Structural Dynamics

F. M. A. Mitrotta, A. Pirrera, T. Macquart, J. E. Cooper, A. P. do Prado and P. H. Cabral. [_"Influence of Skin Curvature on Wingbox Optimization with Nonlinear Structural Stability Constraints"_](https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_0114_paper.pdf), 2024, 34th Congress of the International Council of the Aeronautical Sciences


## Usage

To run the notebooks, you need to use a Python environment with the pyNastran package installed. To run Nastran analyses, you also need to have MSC Nastran installed on your system and you need to set the `NASTRAN_PATH` variable in `/notebooks/resources/pynastran_utils.py` to the path of the Nastran executable. Alternatively, you do not need to run the Nastran analyses to run the notebooks, as with the default setting `run_flag=False` the notebooks will load the results of the analyses from the provided Nastran output files. However, not all the analyses have the results stored in the repository, and until the Nastran input and output files are stored in the data.bris repository, you can ask for specific analyses results by contacting me at [fma.mitrotta@bristol.ac.uk](mailto:fma.mitrotta@bristol.ac.uk).

## List of notebooks

There is a main list of notebooks covering the progression of my understanding of nonlinear structural stability and its application to aeroelastic optimization.

0. [Failed Attempt of the Nonlinear Buckling Analysis of the uCRM-9](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/00_Failed_Attempt_of_the_Nonlinear_Buckling_Analysis_of_the_uCRM-9.ipynb)
1. [Buckling Analysis of Euler's Column](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/01_Buckling_Analysis_of_Euler_Column.ipynb)
2. [Supercritical Pitchfork Bifurcation of Euler's Column](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/02_Supercritical_Pitchfork_Bifurcation_of_Euler_Column.ipynb)
3. [Equilibrium Diagram of a Thin Plate under Uniaxial Compression](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/03_Equilibrium_Diagram_of_a_Thin_Plate_under_Uniaxial_Compression.ipynb)
4. [Nonlinear Buckling Analysis of a Box Beam](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/04_Nonlinear_Buckling_Analysis_of_a_Box_Beam.ipynb)
5. [Sensitivity Study of SOL 106's Nonlinear Analysis Parameters](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/05_Sensitivity_Study_of_SOL_106_Nonlinear_Analysis_Parameters.ipynb)
6. [Verification of SOL 106's Nonlinear Buckling Method](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/06_Verification_of_SOL_106_Nonlinear_Buckling_Method.ipynb)
7. [Nonlinear Buckling Analysis of an Imperfect Euler's Column](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/07_Nonlinear_Buckling_Analysis_of_an_Imperfect_Euler_Column.ipynb)
8. [Nonlinear Buckling Analysis of a Box Beam Reinforced with Ribs](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/08_Nonlinear_Buckling_Analysis_of_a_Box_Beam_Reinforced_with_Ribs.ipynb)
9. [Verification of Shell Elements' Normal Vectors Consistency](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/09_Verification_of_Shell_Elements_Normal_Vectors_Consistency.ipynb)
10. [On the Mechanical Cause of the Box Beam's Bifurcation Break](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/10_On_the_Mechanical_Cause_of_the_Box_Beam_Bifurcation_Break.ipynb)
11. [Equilibrium Diagram of a Box Beam under Uniaxial Compression](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/11_Equilibrium_Diagram_of_a_Box_Beam_Under_Uniaxial_Compression.ipynb)
12. [Nonlinear Buckling Analysis of a Box Beam Reinforced with Ribs and Stiffeners](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/12_Nonlinear_Buckling_Analysis_of_a_Box_Beam_Reinforced_with_Ribs_and_Stiffeners.ipynb)
13. [On the Correct Application of the Nonlinear Buckling Method](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/13_On_the_Correct_Application_of_the_Nonlinear_Buckling_Method.ipynb)
14. [Investigation of the Equilibrium Paths of the Unreinforced Box Beam](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/14_Investigation_of_the_Equilibrium_Paths_of_the_Unreinforced_Box_Beam.ipynb)
15. [Investigation of the Equilibrium Paths of the Box Beam Reinforced with Ribs](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/15_Investigation_of_the_Equilibrium_Paths_of_the_Box_Beam_Reinforced_with_Ribs.ipynb)
16. [Investigation of the Equilibrium Paths of the Box Beam Reinforced with Ribs and Stiffeners](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/16_Investigation_of_the_Equilibrium_Paths_of_the_Box_Beam_Reinforced_with_Ribs_and_Stiffeners.ipynb)
17. [Nonlinear Buckling Analysis of the uCRM-9](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/17_Nonlinear_Buckling_Analysis_of_the_uCRM-9.ipynb)
18. [Development and Nonlinear Structural Stability Analysis of a CRM-like Box Beam Model](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/18_Development_and_Nonlinear_Structural_Stability_Analysis_of_a_CRM-like_Box_Beam_Model.ipynb)
19. [One-variable Optimization of the CRM-like Box Beam](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/19_One-variable_Optimization_of_the_CRM-like_Box_Beam.ipynb)
20. [Two-variables Optimization of the CRM-like Box Beam](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/20_Two-variables_Optimization_of_the_CRM-like_Box_Beam.ipynb)
21. [Optimization of the CRM-like Box Beam with Distributed Load](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/21_Optimization_of_the_CRM-like_Box_Beam_with_Distributed_Load.ipynb)
22. [Optimization of the CRM-like Box Beam with Curved Skin](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/22_Optimization_of_the_CRM-like_Box_Beam_with_Curved_Skin.ipynb)
23. [Optimization of the CRM-like Box Beam under Combined Bending and Torsional Load](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/23_Optimization_of_the_CRM-like_Box_Beam_under_Combined_Bending_and_Torsional_Load.ipynb)
24. [Optimization of the CRM-like Box Beam under an Aerodynamic Load](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/24_Optimization_of_the_CRM-like_Box_Beam_under_an_Aerodynamic_Load.ipynb)
25. [Semi-Aeroelastic Optimization of the Simple Transonic Wing](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/25_Semi-Aeroelastic_Optimization_of_the_Simple_Transonic_Wing.ipynb)

The remaining notebooks are provided to reproduce the figures of the papers mentioned in the Citation section.

- [Proof of Concept of a Nonlinear Structural Stability Constraint for Aeroelastic Optimizations](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/Proof_of_Concept_of_a_Nonlinear_Structural_Stability_Constraint_for_Aeroelastic_Optimization.ipynb)
- [Development of a Nonlinear Structural Stability Constraint for Aeroelastic Optimization](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/Development_of_a_Nonlinear_Structural_Stability_Constraint_for_Aeroelastic_Optimization.ipynb)
- [Influence of Load Introduction Method on Wingbox Optimization with Nonlinear Structural Stability Constraints](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/Influence_of_Load_Introduction_Method_on_Wingbox_Optimization_with_Nonlinear_Structural_Stability_Constraints.ipynb)
- [Influence of Skin Curvature on Wingbox Optimization with Nonlinear Structural Stability Constraints](https://nbviewer.org/github/fmamitrotta/nonlinear-structural-stability-notebooks/blob/main/notebooks/Influence_of_Skin_Curvature_on_Wingbox_Optimization_with_Nonlinear_Structural_Stability_Constraints.ipynb)

## Copyright and License

(c) 2022-2024 Francesco Mario Antonio Mitrotta.

All code is under [BSD-3 clause](https://spdx.org/licenses/BSD-3-Clause.html) and all other content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). 

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://spdx.org/licenses/BSD-3-Clause.html) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
