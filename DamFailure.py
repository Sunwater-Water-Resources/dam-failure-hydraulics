import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
import math
import sys


class Logger(object):
    def __init__(self, name):
        # first reset the stdout to normal
        sys.stdout = sys.__stdout__
        # now direct the stdout to terminal and log file
        self.terminal = sys.stdout
        self.log = open(name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class FailureEvent:
    def __init__(self, log_filename, name=''):
        # log_filename = name.replace(' ', '_')
        # log_filename = name.replace(':', '-')
        # log_filename = 'log/{}_log.txt'.format(log_filename)
        # log_filename = log_filename.replace(' ', '_')
        print('Log file: {}'.format(log_filename))
        sys.stdout = Logger(log_filename)
        header = """---------------------------------------------------------
Simulating a dam failure using level-pool routing method
Scripting created by Sunwater: May 2023
Code available at: https://github.com/Sunwater-Water-Resources/dam-failure-hydraulics
---------------------------------------------------------"""
        header = '{}\n!!!!\nEvent name: {}\n!!!!\n'.format(header, name)
        print(header)
        self.name = name
        self.breach_properties = {}
        self.event_properties = {}
        self.flow_df = pd.DataFrame()
        self.outflow_df = pd.DataFrame()
        self.storage_df = pd.DataFrame()
        self.flow_id = ''
        self.compute_df = pd.DataFrame()
        self.embankments = []
        self.intact_embankments = []
        self.breach_embankments = []
        self.spillway = Spillway()
        self.has_breached = False
        self.dam_structure = {}
        self.timestep = 0.0
        self.failure_type = 'none'
        self.all_walls = []
        self.solution_method = 'direct'  # can be 'direct' or 'implicit'
        self.max_flow = 0.0
        self.max_level = 0.0
        self.max_flow_volume = 0.0

    def log_message(self, message):
        print(message)

    def __str__(self):
        return '\n----\nEvent Name: {}\nResults:\n{}\n----'.format(self.name, self.compute_df)

    def import_event_properties(self, config_file):
        # open the config file and get contents
        print('Importing the event file:', end='\n\t')
        print(config_file)
        # f = open(config_file)
        with open(config_file, 'r') as jsonfile:
            jsondata = ''.join(line for line in jsonfile if not line.startswith('//'))
            self.event_properties = json.loads(jsondata)
        #self.event_properties = json.load(f)
        # f.close()
        start = self.event_properties['start_time']
        stop = self.event_properties['end_time']
        step = self.event_properties['timestep'] / 3600  # convert from seconds to hours
        self.timestep = step  # hours
        period = stop - start
        number = int(period / step)
        stop = start + number * step
        times = np.linspace(start, stop, number)
        self.compute_df = pd.DataFrame({'Time': times})
        self.compute_df = self.compute_df.set_index('Time')
        self.outflow_df = self.compute_df.copy()
        self.solution_method = self.event_properties['solution_method']
        print('Solution method: {}'.format(self.solution_method))

    def import_dam_structure(self, structure_file):
        # open the config file and get contents
        print('Importing the dam structure file:', end='\n\t')
        print(structure_file)
        # f = open(structure_file)
        with open(structure_file, 'r') as jsonfile:
            jsondata = ''.join(line for line in jsonfile if not line.startswith('//'))
            self.dam_structure = json.loads(jsondata)
        # self.dam_structure = json.load(f)
        # f.close()
        # Collect the embankments
        embankments = self.dam_structure['Embankments']
        for embankment in embankments:
            name = str(embankment['name'])
            self.all_walls.append(name)
            print('Setting up embankment: {}'.format(name))
            new_embankment = Embankment(name)
            new_embankment.import_dict(embankment)
            self.embankments.append(new_embankment)
        # Collect the monoliths
        monoliths = self.dam_structure['Monoliths']
        for monolith in monoliths:
            name = str(monolith['name'])
            self.all_walls.append(name)
            print('Setting up monolith: {}'.format(name))
            new_monolith = Monolith(name)
            new_monolith.import_dict(monolith)
            self.embankments.append(new_monolith)
        # Get the storage versus lake level curve
        print('Importing the dam storage curve:', end='\n\t')
        print(self.dam_structure['storage_file'])
        storage_file = self.dam_structure['storage_file']
        self.storage_df = pd.read_csv(storage_file, index_col=0)
        # Get the spillway
        spillway_dict = self.dam_structure['Spillway']
        name = str(spillway_dict['name'])
        self.all_walls.append(name)
        self.spillway.import_dict(spillway_dict)
        self.spillway.name = name

    def set_failure(self, wall_names, type='breach', situation='sunny day', failure_elevation=-999):
        self.failure_type = type
        if 'Spillway' in wall_names:
            self.failure_type = 'none'
        # apply any failures to embankments and monoliths
        print('Searching for embankment to apply failure: {}'.format(wall_names))
        for embankment in self.embankments:
            if situation == 'flood':
                embankment.apply_shift()
            if embankment.has_shift:
                vertical_shift = embankment.properties['lake_hydraulic_grade_shift']
            else:
                vertical_shift = 0.0
            check = (embankment.name in wall_names)
            if check:
                print('\nSetting breach embankment for {} of type: {}'.format(wall_names, type))
                if type == 'breach' or type == 'failure':
                    # set the failure elevation
                    if failure_elevation > -998:
                        failure_elevation = failure_elevation
                    else:
                        failure_elevation = embankment.properties['breach_failure_elevation']
                    print('Breach initiation elevation: {} m AHD'.format(failure_elevation))
                    embankment.initiate_breach_level = failure_elevation - vertical_shift
                    if embankment.has_shift:
                        print('Vertical shift in initiation level: {} m'.format(vertical_shift))
                        print('Final piping initiation elevation: {} m AHD'.format(embankment.initiate_breach_level))

                    # set other properties
                    embankment.breach_elevation = embankment.properties['crest_elevation']
                    embankment.has_breach = True
                    embankment.has_piping = False
                    embankment.timestep = self.timestep

                elif type == 'piping':
                    # set the failure elevation
                    if failure_elevation > -998:
                        failure_elevation = failure_elevation
                    else:
                        failure_elevation = embankment.properties['Piping_failure_elevation']
                    print('Piping initiation elevation: {} m AHD'.format(failure_elevation))
                    embankment.initiate_breach_level = failure_elevation - vertical_shift
                    if embankment.has_shift:
                        print('Vertical shift in initiation level: {} m'.format(vertical_shift))
                        print('Final piping initiation elevation: {} m AHD'.format(embankment.initiate_breach_level))

                    # set the other properties
                    embankment.breach_elevation = failure_elevation
                    embankment.piping_invert = failure_elevation
                    embankment.has_breach = False
                    embankment.has_piping = True
                    embankment.timestep = self.timestep
                    embankment.piping_soffit = embankment.initiate_breach_level

                self.breach_embankments.append(embankment)
            else:
                self.intact_embankments.append(embankment)
        if not self.breach_embankments:
            print('\n!!!!!!!!!!\nFailure embankment {} not found\n!!!!!!!!\n'.format(wall_names))

        # apply any failures to the spillway
        if self.spillway.name in wall_names:
            print('A breach has been applied to the spillway')
            self.spillway.has_breach = True
            self.spillway.initiate_breach_level = failure_elevation

    def import_inflow(self, inflow_file='none', column_name=''):
        # first column is time in hours and others are flow in cumecs
        if not inflow_file == '':
            self.flow_id = column_name
            print('Importing inflow file using column {}:'.format(column_name), end='\n\t')
            print(inflow_file)
            self.flow_df = pd.read_csv(inflow_file, index_col=0)
            x = self.flow_df.index.to_numpy()
            y = self.flow_df[self.flow_id].to_numpy()
            f = interpolate.interp1d(x, y)
            xnew = self.compute_df.index.to_numpy()
            ynew = f(xnew)
            self.compute_df['Inflow'] = ynew
        else:
            print('No inflow for this event!')
            self.compute_df['Inflow'] = 0.0

    def get_storage(self, lake_level):
        x = self.storage_df.index.to_numpy()
        y = self.storage_df.iloc[:, 0].to_numpy()
        f = interpolate.interp1d(x, y)
        return f(lake_level)

    def get_lake_level(self, volume):
        y = self.storage_df.index.to_numpy()
        x = self.storage_df.iloc[:, 0].to_numpy()
        min_y = np.amin(y)
        if volume < 0.0:
            lake_level = min_y
        else:
            try:
                f = interpolate.interp1d(x, y)
                lake_level = f(volume)
            except ValueError:
                print('Lake level not found for volume {} ML'.format(volume))
                print('Assuming lowest level of {}'.format(min_y))
                lake_level = min_y

        if lake_level < min_y:
            lake_level = min_y
        if np.isnan(lake_level):
            lake_level = min_y
        return lake_level

    def intact_overflow(self, lake_level, time=0.0):
        total_flow = 0.0
        for embankment in self.intact_embankments:
            flow = embankment.intact_overflow(lake_level)
            self.outflow_df.loc[time, embankment.name] = flow
            total_flow += flow
            # if time > 146.2 and time < 146.3:
            #     print('time: {} | embankment flow: {} | total flow: {}'.format(time, flow, total_flow))
        return total_flow

    def add_spillway(self, spillway_file, name=''):
        self.spillway.import_dict(spillway_file)
        self.spillway.name = name

    def failure_flow(self, lake_level, time):
        total_main_flow, total_lateral_flow, total_intact_flow = [0.0, 0.0, 0.0]
        for embankment in self.breach_embankments:
            main_flow, lateral_flow, intact_flow = embankment.failure_flow(lake_level, time)
            total_flow = main_flow + lateral_flow + intact_flow
            self.outflow_df.loc[time, embankment.name] = total_flow

            total_main_flow += main_flow
            total_lateral_flow += lateral_flow
            total_intact_flow += intact_flow

        return [total_main_flow, total_lateral_flow, total_intact_flow]

    def compute_event(self):
        print('\nDoing the computation...')

        # reset the dataframe if this is a new computation
        self.compute_df = self.compute_df[['Inflow']]

        # set initial values
        initial_time = self.compute_df.index[0]
        initial_level = float(self.event_properties['initial_lake_level'])
        initial_storage = self.get_storage(initial_level)
        initial_spill = self.spillway.intact_overflow(initial_level)
        self.outflow_df.loc[initial_time, 'Spillway'] = initial_spill
        initial_intact = self.intact_overflow(initial_level, initial_time)
        initial_inflow = self.compute_df['Inflow'].iloc[0]
        initial_outflow = initial_spill + initial_intact
        initial_inflow = self.compute_df.Inflow[0]

        computation = {'Time': [initial_time],
                       'Level': [initial_level],
                       'Storage': [initial_storage],
                       'Spillway': [initial_spill],
                       'Intact Flow': [initial_intact],
                       'Breach Main Flow': [0.0],
                       'Breach Lateral Flow': [0.0],
                       'Breach Intact Flow': [0.0],
                       'Total Breach Flow': [0.0],
                       'Total Outflow': [initial_outflow],
                       'Mass Error (ML)': [0.0]}
        initial_delta_flow = initial_inflow - initial_outflow
        # Do the main computation
        prior_delta_storage = 0.0
        flow_volume = 0.0
        for time, step in self.compute_df.iloc[1:].iterrows():

            # 1. get the timestep... strictly, this should be the previous
            #    timestep, but the timestep is uniform so no matter
            delta_time = (time - initial_time) * 3600  # converted from hours to seconds

            # 2. establish the inflows for the current timestep
            inflow = step.Inflow
            average_inflow = (initial_inflow + inflow) / 2

            # 3. get the first estimate the lake volume and level based on the
            #    change in inflow and outflow at the previous timestep
            delta_storage = (initial_inflow - initial_outflow) * delta_time / 1000
            volume = initial_storage + delta_storage
            # lake_level_0 = self.get_lake_level(initial_storage + delta_storage / 2)
            level_tolerance = 10
            lake_level_1 = self.get_lake_level(initial_storage + delta_storage * level_tolerance)
            if (initial_level - lake_level_1)**2 > 0.001 and self.solution_method == 'implicit':
                try:
                    # print('initial lake level: {} | upper bound {}'.format(initial_level, lake_level_1))
                    root = optimize.root_scalar(self.lake_level_optimisation,
                                                x0=initial_level,
                                                x1=lake_level_1,
                                                args=(time, average_inflow, initial_outflow,
                                                      initial_storage, delta_time))
                    lake_level = root.root
                except ValueError:
                    print('Convergence issues at time: {} hours'.format(time))
                    lake_level = self.get_lake_level(volume)
                    print('Adopting lake level of: {} m/s'.format(lake_level))
            else:
                lake_level = self.get_lake_level(volume)

            # 5. get the final outflow estimate for the current timestep
            spillway = self.spillway.flow(lake_level, time)
            intact_flow = self.intact_overflow(lake_level, time)
            breach_main_flow, breach_lateral_flow, breach_intact_flow = self.failure_flow(lake_level, time)
            total_breach_flow = breach_main_flow + breach_lateral_flow + breach_intact_flow
            outflow = spillway + intact_flow + total_breach_flow

            # 6. store the mass balance (ML)
            delta_storage = (average_inflow - 0.5 * (initial_outflow + outflow)) * delta_time / 1000
            volume = initial_storage + delta_storage
            volume_2 = self.get_storage(lake_level)
            mass_error = volume_2 - volume

            # 7. store the results for the current timestep
            computation['Time'].append(time)
            computation['Level'].append(lake_level)
            computation['Storage'].append(volume)
            computation['Spillway'].append(spillway)
            self.outflow_df.loc[initial_time, 'Spillway'] = spillway
            computation['Intact Flow'].append(intact_flow)
            computation['Breach Main Flow'].append(breach_main_flow)
            computation['Breach Lateral Flow'].append(breach_lateral_flow)
            computation['Breach Intact Flow'].append(breach_intact_flow)
            computation['Total Breach Flow'].append(total_breach_flow)
            computation['Total Outflow'].append(outflow)
            computation['Mass Error (ML)'].append(mass_error)

            # Get the volume that has flowed through the dam
            flow_volume = flow_volume + 0.5 * (initial_outflow + outflow) * self.timestep * 3600 / 1000  # Megalitres

            # 5. Store values to use in next timestep
            initial_time = time
            initial_storage = volume
            initial_outflow = outflow
            initial_inflow = inflow
            initial_level = lake_level

            if lake_level > self.max_level:
                self.max_level = lake_level
            if outflow > self.max_flow:
                self.max_flow = outflow
                self.max_flow_volume = flow_volume

        # Store the results
        computation_df = pd.DataFrame(computation).set_index('Time')
        self.compute_df = pd.concat([self.compute_df, computation_df], axis=1)

    def get_outflow(self, lake_level, time):
        spillway = self.spillway.flow(lake_level, time)
        intact_flow = self.intact_overflow(lake_level, time)
        breach_main_flow, breach_lateral_flow, breach_intact_flow = self.failure_flow(lake_level, time)
        total_breach_flow = breach_main_flow + breach_lateral_flow + breach_intact_flow
        outflow = spillway + intact_flow + total_breach_flow
        return outflow

    def lake_level_optimisation(self, lake_level, time, average_inflow, initial_outflow, initial_storage, delta_time):
        outflow_1 = self.get_outflow(lake_level, time)
        storage_2 = self.get_storage(lake_level)
        outflow_2 = (2/delta_time) * (initial_storage*1000-storage_2*1000) + 2 * average_inflow - initial_outflow
        # print('test lake level: {}'.format(lake_level))
        # print('outflow 1: {} | outflow 2: {}'.format(outflow_1, outflow_2))
        return outflow_2 - outflow_1

    def write_to_csv(self, filepath=''):
        if filepath == '':
            filepath = self.event_properties['output_path']
        print('\nWriting results to file:', end='\n\t')
        print(filepath)
        self.compute_df['Mass Error (%)'] = self.compute_df['Mass Error (ML)']
        self.compute_df.to_csv(filepath)
        filepath = filepath.replace('.csv', '_banks.csv')
        self.outflow_df.to_csv(filepath)

    def create_plot(self, filepath='', show_plot=True, save_plot=False):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        if self.failure_type == 'none':
            flow_df = self.compute_df[['Inflow', 'Total Outflow']].copy()
            flow_df.plot(ax=axs[0], color=['royalblue', 'seagreen'])
        else:
            flow_df = self.compute_df[['Inflow', 'Total Breach Flow', 'Total Outflow']].copy()
            flow_df.plot(ax=axs[0], color=['royalblue', 'red', 'seagreen'])

        level_df = self.compute_df[['Level']].copy()
        level_df['Level'] = level_df['Level'].astype(float)

        level_df.plot(ax=axs[1], color=['black'], legend=None)
        axs[0].set_ylabel('Flow (m$^3$/s)')
        axs[0].set_xlabel('Time (hours)')
        axs[1].set_ylabel('Lake Level (m AHD)')
        axs[1].set_xlabel('Time (hours)')
        fig.suptitle(self.name)
        fig.tight_layout()

        if save_plot:
            if filepath == '':
                filepath = self.event_properties['output_path'].replace('.csv', '.png')
            print('Writing the plot to file:', end='\n\t')
            print(filepath)
            fig.savefig(filepath)

        if show_plot:
            plt.show()


class Embankment:
    def __init__(self, name='', has_breach=False, has_piping=False, number_of_sides=0):
        self.name = name
        self.properties = {}
        self.has_breach = has_breach
        self.has_piping = has_piping
        self.sidewalls = number_of_sides
        self.initiate_breach_level = 0.0
        self.is_breached = False
        self.is_piping = False
        self.is_lateral_breach = False
        self.time_of_main_breach = 0.0
        self.time_of_lateral_breach = 0.0
        self.breach_elevation = 0.0
        self.main_breach_width = 0.0
        self.lateral_breach_width = 0.0
        self.lateral_breach_ceased = False
        self.previous_time = 0.0
        self.prior_time = 0.0  # used in piping failure formation with implicit method
        self.has_shift = False
        self.timestep = 0.0
        self.piping_soffit = 0.0
        self.piping_invert = 0.0
        self.min_lake_level = 0.0
        if has_breach:
            print('Found a breach failure mechanism for {}'.format(self.name))
        if has_piping:
            print('Found a piping failure mechanism for {}'.format(self.name))

    def import_dict(self, embankment_dict, apply_shift=False):
        self.properties = embankment_dict
        print(self.properties)

    def apply_shift(self):
        self.has_shift = 'lake_hydraulic_grade_shift' in self.properties
        if self.has_shift:
            print("\n!!!\nLowering embankment {} by {} m to account for lake hydrodynamics (hydraulic gradient)\n!!!\n".
                  format(self.name, self.properties['lake_hydraulic_grade_shift']))
            self.properties['crest_elevation'] -= self.properties['lake_hydraulic_grade_shift']
            print(self.properties)

    def intact_overflow(self, lake_level, width=0):
        Cd = self.properties['discharge_coefficient']
        if width > 0:
            L = width
        else:
            L = self.properties['crest_length']
        H = lake_level - self.properties['crest_elevation']
        if H > 0.001:
            return Cd * L * math.pow(H, 1.5)
        else:
            return 0.0

    def failure_flow(self, lake_level, time):
        # set initial intact and breach flows assuming no failure
        # Note that these get overwritten if there is a failure
        intact_flow = self.intact_overflow(lake_level)
        main_flow, lateral_flow = [0.0, 0.0]

        # get the flow through failure modes
        self.check_failed(lake_level, time)
        if self.has_breach and self.is_breached:
            # Get flow through the main breach
            main_flow = self.main_breach_flow(lake_level, time)
            # Get flow through the lateral part of the breach
            if self.is_lateral_breach and self.properties['include_lateral_breach']:
                lateral_flow = self.lateral_breach_flow(lake_level, time)
            else:
                lateral_flow = 0.0
            # Get flow over the intact part of the breach
            breach_depth = self.properties['crest_elevation'] - self.breach_elevation
            top_side_width = breach_depth * self.properties['side_slope_H_in_1V']
            top_main_breach_width = self.main_breach_width + 2 * top_side_width
            intact_width = self.properties['crest_length'] - top_main_breach_width
            if self.properties['include_lateral_breach']:
                intact_width -= self.lateral_breach_width
            if intact_width > 0:
                intact_flow = self.intact_overflow(lake_level, intact_width)
            else:
                intact_flow = 0.0

        # compute the piping flow
        elif self.has_piping and self.is_piping:
            main_flow = self.piping_flow(lake_level, time)
            # print('Piping flow: {} m³/s | Piping soffit: {} m AHD | Crest: {} m AHD'.format(
            #     np.around(main_flow, decimals=2),
            #     np.around(self.piping_soffit, decimals=2),
            #     self.properties['crest_elevation']))
            if self.is_lateral_breach and self.properties['include_lateral_breach']:
                lateral_flow = self.lateral_breach_flow(lake_level, time)

        return [main_flow, lateral_flow, intact_flow]

    def piping_flow(self, lake_level, time):
        # get the base width of the breach
        time_since_main_breach = time - self.time_of_main_breach
        time_scaling = time_since_main_breach / self.properties['failure_period']
        breach_width = self.properties['breach_base_width'] * time_scaling
        if breach_width >= self.properties['breach_base_width']:
            breach_width = self.properties['breach_base_width']

        # get the soffit of the breach
        breach_depth = self.properties['breach_depth'] * time_scaling
        # print('time: {} | previous time: {}'.format(time, self.prior_time))
        if time > self.prior_time:  # this skips iterations done on the same timestep
            delta_depth = (self.timestep / self.properties['failure_period']) * self.properties['breach_depth']
        else:
            delta_depth = 0.0
        # assume that the soffit erodes at the same rate as the invert
        # i.e. vertical expansion is twice as fast as horizontal expansion
        piping_soffit = self.piping_soffit + delta_depth  # /2 if vertical = horizontal rate
        # check that the lake level is above pipe soffit and can erode the pipe ceiling
        if lake_level >= piping_soffit:
            self.piping_soffit = piping_soffit
        # Else check if the lake level is higher than the existing pipe ceiling and can partially erode the pipe ceiling
        elif lake_level > self.piping_soffit:
            self.piping_soffit = lake_level
        # else the lake level is too low to erode the pipe ceiling and the pipe soffit will not change

        # get the invert of the breach
        piping_invert = self.piping_invert - delta_depth
        piping_invert_limit = self.properties['crest_elevation'] - self.properties['breach_depth']
        if piping_invert <= piping_invert_limit:
            self.piping_invert = piping_invert_limit
            # initiate the lateral breach if the pipe invert hits the bottom
            lateral_breach_invert = self.properties['crest_elevation'] - self.properties['lateral_breach_depth']
            lateral_check = piping_soffit >= lateral_breach_invert
            if not self.is_lateral_breach and self.properties['include_lateral_breach'] and lateral_check:
                print('The pipe invert has reached the full breach depth at {} m AHD'.format(self.piping_invert))
                print('Initiating the lateral breach mechanism at time {}'.format(time))
                self.time_of_lateral_breach = time
                self.is_lateral_breach = True
        else:
            self.piping_invert = piping_invert

        breach_depth = self.piping_soffit - self.piping_invert
        breach_centre = self.piping_invert + breach_depth / 2
        crest_check = (piping_soffit - self.properties['crest_elevation'])
        # lake_check = (lake_level < piping_soffit)
        if crest_check > -0.001:
            print('\n!!!!\nPiping failure {} has fully formed creating a breach at a lake level of {} m AHD at {} hours'.
                  format(self.name, np.around(lake_level, decimals=2), np.around(time, decimals=2)))
            print('piping soffit is {} m AHD and the embankment crest is {} m AHD'.format(self.piping_soffit,
                                                                                          self.properties['crest_elevation']))
            print('piping invert is {} m AHD and the lake level is {} m AHD'.format(self.piping_invert, lake_level))
            print('piping depth: {}'.format(breach_depth))
            print('piping elapsed time: {} hours'.format(time_since_main_breach))
            self.is_breached = True
            self.has_breach = True
            self.is_piping = False
            print('\n!!!!\nMain breach {} initiated at a level of {} m AHD at {} hours'.
                  format(self.name, np.around(lake_level, decimals=2), np.around(time, decimals=2)))
            main_flow = self.main_breach_flow(lake_level, time)
        else:
            # print('Time: {} hours'.format(np.around(time, decimals=2)))
            # !!!!!!!!!!!!!!!!!!!
            # insert the piping flow here
            centre_area = breach_depth * breach_width
            side_area = 0.5 * self.properties['side_slope_H_in_1V'] * breach_depth * breach_depth
            breach_area = centre_area + side_area * 2  # assuming trapezoidal pipe shape similar to overtopping breach
            if lake_level > self.piping_soffit:
                if time > self.prior_time:
                    print('Time: {} | Orifice flow | Lake level: {} m AHD | Pipe invert: {} m AHD'.format(
                        time,
                        np.around(lake_level, decimals=2),
                        np.around(self.piping_invert, decimals=2)))
                main_flow = self.orifice_flow(area=breach_area, head=(lake_level - breach_centre))
            elif lake_level > self.piping_invert:
                if time > self.prior_time:
                    print('Time: {} | Weir flow | Lake level: {} m AHD | Pipe invert: {} m AHD'.format(
                        time,
                        np.around(lake_level, decimals=2),
                        np.around(self.piping_invert, decimals=2)))
                main_flow = self.trapezoid_weir_flow(width=breach_width, head=(lake_level - self.piping_invert))
            else:
                main_flow = 0.0
            # !!!!!!!!!!!!!!!!!!!!!
            self.prior_time = time
        return main_flow

    def orifice_flow(self, area, head):
        g = 9.81  # gravitational acceleration
        Cd = 4.8 / 1.811 / math.sqrt(2 * g)  # discharge coefficient is ~0.6
        return Cd * area * math.sqrt(2 * g * head)

    def trapezoid_weir_flow(self, width, head, Cd_base=-1.0, Cd_side=-1.0):
        if Cd_base < 0.0:
            Cd_base = 3.1 / 1.811  # imperial to metric conversion
        if Cd_side < 0.0:
            Cd_side = 2.45 / 1.811  # imperial to metric conversion
            Cd_side = Cd_side / 2  # convert from being for two sides to one side
        side = self.properties['side_slope_H_in_1V']
        if head > 0.001:
            base_flow = Cd_base * width * math.pow(head, 1.5)
            side_flow = Cd_side * side * math.pow(head, 2.5)
            return base_flow + 2 * side_flow
        else:
            return 0.0

    def main_breach_flow(self, lake_level, time, Cd_base=-1.0, Cd_side=-1.0):
        # check if the main breach is fully developed
        if not self.is_lateral_breach:
            # if not, compute the main breach dimensions
            time_since_main_breach = time - self.time_of_main_breach
            time_scaling = time_since_main_breach / self.properties['failure_period']
            breach_depth = self.properties['breach_depth'] * time_scaling
            if self.has_piping:
                breach_elevation = self.initiate_breach_level - breach_depth
            else:
                breach_elevation = self.properties['crest_elevation'] - breach_depth
            breach_floor = self.properties['crest_elevation'] - self.properties['breach_depth']
            breach_width = self.properties['breach_base_width'] * time_scaling
            check_depth = (breach_elevation <= breach_floor)
            check_width = (breach_width >= self.properties['breach_base_width'])
            # print('Depth check: {} | Width check: {}'.format(check_depth, check_width))
            if check_depth or check_width:
                # now main breach is fully developed
                print('Main breach {} fully formed at a level of {} m AHD at {} hours'.
                      format(self.name, np.around(lake_level, decimals=2), np.around(time, decimals=2)))
                self.breach_elevation = breach_floor
                self.main_breach_width = self.properties['breach_base_width']
                self.is_lateral_breach = True
                self.time_of_lateral_breach = time
                print('Main breach {} final elevation of {} m AHD and width of {} m\n!!!!'.format(
                    self.name, self.breach_elevation, self.main_breach_width))
                if self.is_lateral_breach and self.properties['include_lateral_breach']:
                    print('Starting lateral breach for {} at time of {} hours'.format(
                        self.name, np.around(time, decimals=2)))
                    print('Lateral breach direction is: {}'.format(self.properties['lateral_breach_direction']))
            # else if main breach is already fully developed
            else:
                if self.has_piping:
                    self.breach_elevation = self.initiate_breach_level - breach_depth
                else:
                    self.breach_elevation = self.properties['crest_elevation'] - breach_depth
                self.main_breach_width = breach_width

        L = self.main_breach_width
        H = lake_level - self.breach_elevation

        return self.trapezoid_weir_flow(width=L, head=H)

    def check_failed(self, lake_level, time):
        if not self.is_breached and self.has_breach:
            if lake_level >= self.initiate_breach_level:
                self.is_breached = True
                self.time_of_main_breach = time
                print('\n!!!!\nMain breach {} initiated at a level of {} m AHD at {} hours'.
                      format(self.name, np.around(lake_level, decimals=2), np.around(time, decimals=2)))

        elif not self.has_breach and self.has_piping and not self.is_piping:
            if lake_level >= self.initiate_breach_level or (lake_level - self.initiate_breach_level)**2 < 0.001:
                self.is_piping = True
                self.time_of_main_breach = self.previous_time
                print('\n!!!!\nPiping failure initiation level: {} m AHD'.format(self.initiate_breach_level))
                print('Piping failure {} initiated at a level of {} m AHD at {} hours'.format(
                    self.name, np.around(lake_level, decimals=2), np.around(self.time_of_main_breach, decimals=2)))
        self.previous_time = time

    def lateral_breach_flow_1(self, lake_level, time, Cd=-1.0):
        Cd_base = Cd
        breach_invert = self.properties['crest_elevation'] - self.properties['lateral_breach_depth']

        if Cd_base < 0.0:
            Cd_base = 3.1 / 1.811  # imperial to metric conversion

        if not self.lateral_breach_ceased:
            erosion = self.timestep * self.properties['lateral_breach_erosion_rate']  # in m/hr
            if self.properties['lateral_breach_direction'] == 'bi':
                erosion = erosion * 2

            if lake_level > breach_invert:
                self.lateral_breach_width += erosion

            if self.lateral_breach_width >= self.properties['lateral_breach_base_width']:
                self.lateral_breach_width = self.properties['lateral_breach_base_width']
                print('Lateral erosion for {} has reached the maximum width of {} m at {} hours\n!!!!'.
                       format(self.name, self.lateral_breach_width, np.around(time, decimals=2)))
                self.lateral_breach_ceased = True

        L = self.lateral_breach_width
        H = lake_level - breach_invert

        if H > 0.001 and L > 0.001:
            return Cd_base * L * math.pow(H, 1.5)
        else:
            return 0.0

    def lateral_breach_flow(self, lake_level, time, Cd=-1.0):
        # Strickly, lateral erosion should only occur when lake level is
        # above the breach invert level. Tried doing this in lateral_breach_flow_1
        # but was overestimating for some reason.
        Cd_base = Cd
        if Cd_base < 0.0:
            Cd_base = 3.1 / 1.811  # imperial to metric conversion

        if not self.lateral_breach_ceased:
            time_since_lateral_breach = time - self.time_of_lateral_breach
            erosion_rate = self.properties['lateral_breach_erosion_rate']  # in m/hr
            if self.properties['lateral_breach_direction'] == 'mono':
                lateral_erosion = erosion_rate * time_since_lateral_breach
            else:
                lateral_erosion = 2 * erosion_rate * time_since_lateral_breach
            if lateral_erosion >= self.properties['lateral_breach_base_width']:
                self.lateral_breach_width = self.properties['lateral_breach_base_width']
                print('Lateral erosion for {} has reached the maximum width of {} m at {} hours\n!!!!'.
                       format(self.name, self.lateral_breach_width, np.around(time, decimals=2)))
                self.lateral_breach_ceased = True
            else:
                self.lateral_breach_width = lateral_erosion

        L = self.lateral_breach_width
        H = self.properties['lateral_breach_depth'] - (self.properties['crest_elevation'] - lake_level)

        if H > 0.001 and L > 0.001:
            return Cd_base * L * math.pow(H, 1.5)
        else:
            return 0.0


class Monolith(Embankment):
    def __init__(self, name='', has_breach=False):
        self.flow_reduction_factor = 1.0
        self.has_failure = False
        super(Monolith, self).__init__(name, has_breach=has_breach, has_piping=False, number_of_sides=0)

    def failure_flow(self, lake_level, time):
        # set initial intact and breach flows assuming no failure
        # Note that these get overwritten if there is a failure
        intact_flow = self.intact_overflow(lake_level)
        main_flow, lateral_flow = [0.0, 0.0]

        # get the flow through failure modes
        self.check_failed(lake_level, time)
        if self.is_breached:
            intact_flow = intact_flow * self.flow_reduction_factor
            invert_level = self.properties['crest_elevation'] - self.properties['breach_depth']
            head = lake_level - invert_level
            print('Breach invert is {} m AHD | Lake level is {} m AHD | Effective head: {} m'.format(
                invert_level, np.around(lake_level, decimals=2), np.around(head, decimals=2)))
            if head > 0.001:
                main_flow = self.trapezoid_weir_flow(width=self.properties['breach_base_width'],
                                                     head=head,
                                                     Cd_base=self.properties['discharge_coefficient'],
                                                     Cd_side=self.properties['discharge_coefficient'])
            else:
                main_flow = 0.0
        return [main_flow, lateral_flow, intact_flow]

    def check_failed(self, lake_level, time):
        if not self.is_breached and self.has_breach:
            if lake_level > self.initiate_breach_level:
                self.is_breached = True
                self.time_of_main_breach = time
                self.flow_reduction_factor = (self.properties['crest_length'] -
                                              self.properties['breach_base_width']) / self.properties['crest_length']
                print('\n!!!!\nMonolith failure {} initiated at a level of {} m AHD at {} hours'.
                      format(self.name, np.around(lake_level, decimals=2), np.around(time, decimals=2)))
                print('Breach initiation level is {} m AHD'.format(self.initiate_breach_level))


class Spillway(Monolith):
    def __init__(self, name='', has_breach=False):
        self.rating_df = pd.DataFrame()
        super(Spillway, self).__init__(name, has_breach=has_breach)

    def import_dict(self, spillway_dict):
        print('Setting up the spillway...')
        self.properties = spillway_dict
        # Get rating: first column is elevation in m AHD and second volume is flow in cumecs
        print('Importing spillway rating file:', end='\n\t')
        rating_file = self.properties['spillway_rating_file']
        print(rating_file)
        self.rating_df = pd.read_csv(rating_file, index_col=0)

    def intact_overflow(self, lake_level):
        if lake_level > self.properties['crest_elevation']:
            x = self.rating_df.index.to_numpy()
            y = self.rating_df.iloc[:, 0].to_numpy()
            f = interpolate.interp1d(x, y)
            return f(lake_level) * self.flow_reduction_factor
        else:
            return 0.0

    def flow(self, lake_level, time):
        if self.has_breach:
            main_flow, lateral_flow, intact_flow = self.failure_flow(lake_level, time)
            flow = main_flow + intact_flow
        else:
            flow = self.intact_overflow(lake_level)
        return flow
