# Dam Failure Hydraulics
A python-based model for simulating dam failures. The level-pool method and weir equations are used to estimate discharge from a dam via overtopping and/or through breaches. The model was adapted from a DNRME spreadsheet with the aim of being able to efficiently simulate a large volume of design floods and failure scenarios. 

## Dependencies
Python packages required include:
- scipy
- matplotlib
- numpy
- pandas
- json

## Model setup
A test model has been provided to demonstrate the model setup files. The various components and associated files are discussed below. Note that all json files can have comments added using // at the start of the line.

- **Spillway rating curve**: this is a csv file where the first column is the lake level (m AHD) and the second column is the discharge (m³/s).
- **Reservoir storage curve**: this is a csv file where the first column is the lake level (m AHD) and the second column is the storage volume (ML).
- **Inflows file**: this is a csv file conatining a list of lake inflows for different flood events. The first column is *time* (hours) and subsequent columns are inflows (m³/s). The header of each inflow column is used in the scenario list to define which inflow to use in the model.
- **Dam structure file**: this is a json file where details for the embankments, concrete monliths, spillway, and reservoir storage are inserted. A filepath to the spillway rating curve is needed for the spillway details and a filepath to the storage versus lake level (m AHD) is also needed in this file. 
- **Failure event properties file**: this is a json file where the start time (hours), end time (hours), and computation timestep (seconds) are provided. The initial lake level (m AHD) is also provided in this file. A solution method is prescribed in this file, which can be either *direct* or *indirect*. The direct method estimates the lake storage in the next timestep based on the change in storage computed in the current timestep. The indirect method computes the change in storage over the current timestep as a function of the outflow at both the current and next timestep, and uses an optimisation method due to the interdepence of outflow at the next timestep and change in lake level. Testing shows no substantial difference in the results produced by these two methods if a sufficiently small timestep is used for the direct method. Since the indirect method has not been fully tested for overtopping failures, the direct method is recommended. 
- **Simulation list**: This is an excel spreadsheet (.xlsx) that lists the events to be simulated. Filepaths to the inflow file, dam structure file, and failure event properties file are inserted here. The header for the required inflow must be inserted in the *Inflow_id* field. The failure trigger elevation, provided in the dam structure file, can be overridden here using the *Failure_elevation* field. The failure type - *piping*, a *breach* caused by overtopping, or *none* - is provided in the *Failure_type* field. 

