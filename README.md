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
- **Event file**: this is a json file where the start time (hours), end time (hours), and computation timestep (seconds) are provided. The initial lake level (m AHD) is also provided in this file. A solution method is prescribed in this file, which can be either *direct* or *indirect*. The direct method estimates the lake storage in the next timestep based on the change in storage computed in the current timestep. The indirect method computes the change in storage over the current timestep as a function of the outflow at both the current and next timestep, and uses an optimisation method due to the interdepence of outflow at the next timestep and change in lake level. Testing shows no substantial difference in the results produced by these two methods if a sufficiently small timestep is used for the direct method. Since the indirect method has not been fully tested for overtopping failures, the direct method is recommended. 
- **Simulation list**: this is an excel spreadsheet (.xlsx) that lists the events to be simulated. The table below provides a list of fields along with descriptions.
- **Dam failure runs**: this is a json file that controls which simulation list file the model should run. Several simulation list files can be run, one after the other. There is an option in this file to output a result file of the maximum discharge and lake level from each simulation in the simulation list, which is useful for checking how different piping trigger elevations influence the discharge and identifying the median of an ensemble of events. The output location of this summary of maximums file is also defined here. 
- **The run file**: this is a Windows batch file used to run the model (*main.py*). The *dam failure runs file* must be passed as an argument in the bacth file.   

The list of fields in the *Simulation list* along with descriptions for each field is provided in the table below. 
|Field | Description|
|------|------------|
|*Include*|*no* excludes this event from the simulation.|
|*Simulation_name*|The title of the event, which is used as the title in the result plot.|
|*Situation*|Either *flood* for a flood event or *sunny day* for a no-flood event.|
|*Failure_structure*|The name of the structure that fails. More than one structure can be listed using a comma to seperate the structures.|
|*Failure_type*|*piping* or an overtopping *breach* for embankments, overturning *failure* for concrete monoliths, or *none* for no failure scenarios.|
|*Failure_elevation*|If used, overrides the failure trigger elevation provided in the dam structure file.|
|*Inflow_file*|Filepaths to the inflow file.|
|*Inflow_id*|The header for the inflow corresponding with the inflow file.|
|*Output_name*|The path for writing the output files.|
|*Dam_structure_file*|The filepath to the dam structure file.|
|*Event_file*|The filepath to the event file.|
|*Show_plot*|*no* prints the result plot without showing the plot at the end of the simulation, which is a better option for bach runs.|
|*Erosion_rate_ID*| (Optional) The ID for specifying which lateral erosion rate to use from the list of erosion rates in the *dam structure file*|

## Outputs
The model produces five output files:
- A log file (*.txt*)
- A table of discharges (*.csv*)
- A table of dischrages for each structure (*_banks.csv*)
- A plot of the results (*.png*)
- (Optional) a summary of the maximum flows and levels of the events in the simulation list. This file is managed from the *dam failure runs* file.
