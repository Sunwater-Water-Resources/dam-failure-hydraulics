import os
import json
from DamFailure import FailureEvent
import pandas as pd
import sys


def main():
    # Get the list of model runs
    dam_failure_runs_file = sys.argv[1]
    print('Found dam failure run file:')
    print(dam_failure_runs_file)
    with open(dam_failure_runs_file, 'r') as jsonfile:
        jsondata = ''.join(line for line in jsonfile if not line.startswith('//'))
        dam_failure_runs = json.loads(jsondata)
    print(dam_failure_runs)

    # Do the model runs
    for dam_failure_run in dam_failure_runs['Simulations']:
        if dam_failure_run['include']:
            print('Running simulation: {}'.format(dam_failure_run['name']))
            run_dam_failure_model(dam_failure_run)


def run_dam_failure_model(dam_failure_run):
    # get the list of simulations
    simulation_list = dam_failure_run['simulation_list']
    folder = dam_failure_run['working_folder']
    simulation_list = os.path.join(folder, simulation_list)
    print('Opening the simulation list:')
    print(simulation_list)
    if '.xlsx' in simulation_list or '.xlsm' in simulation_list:
        simulation_df = pd.read_excel(simulation_list, sheet_name='Simulation_List')
    else:
        simulation_df = pd.read_csv(simulation_list)

    # run each simulation
    all_max_flows_df = pd.DataFrame()
    for run_id, simulation in simulation_df.iterrows():
        if simulation['Include'] == 'yes':
            max_flow, max_level = run_failure_simulation(simulation, folder)
            # output a list of the maximum flows and lake levels for each simulation
            if dam_failure_run['write_max_flows']:
                max_flow_dict = {'Simulation': str(simulation['Simulation_name']),
                                 'Failure Elevation': str(simulation['Failure_elevation']),
                                 'Maximum Flow': max_flow,
                                 'Maximum Level': max_level}
                max_flows_df = pd.DataFrame.from_dict([max_flow_dict])
                all_max_flows_df = pd.concat([all_max_flows_df, max_flows_df], axis=0)
                max_flow_filename = os.path.join(dam_failure_run['working_folder'], dam_failure_run['max_flow_filename'])
                all_max_flows_df.to_csv(max_flow_filename)


def run_failure_simulation(simulation, folder):
    # Create a new dam failure event
    log_filename = str(simulation['Output_name']).replace('.csv', '.txt')
    log_filename = os.path.join(folder, log_filename)
    dam_failure = FailureEvent(name=simulation['Simulation_name'],
                               log_filename=log_filename)
    dam_failure.log_message('\n!!!!!\nRunning simulation for: {}\n!!!!!\n'.format(simulation['Simulation_name']))
    dam_failure.log_message(simulation)

    # Set up the structure of the dam
    filename = os.path.join(folder, simulation['Dam_structure_file'])
    dam_failure.import_dam_structure(filename)

    # Set up the dam failure event properties
    filename = os.path.join(folder, simulation['Event_file'])
    dam_failure.import_event_properties(filename)

    # Get the inflows
    filename = os.path.join(folder, simulation['Inflow_file'])
    dam_failure.import_inflow(inflow_file=filename,
                              column_name=simulation['Inflow_id'])

    # Set the part of the dam that will fail
    if not pd.isnull(simulation['Failure_elevation']):
        failure_elevation = simulation['Failure_elevation']
    else:
        failure_elevation = -999
    failure_walls = simulation['Failure_structure'].split(',')
    for wall in failure_walls:
        dam_failure.set_failure(wall_names=wall,
                                type=simulation['Failure_type'],
                                situation=simulation['Situation'],
                                failure_elevation=failure_elevation)

    # Do the dam failure analysis
    dam_failure.compute_event()

    # Save the results
    output_filename = os.path.join(folder, simulation['Output_name'])
    dam_failure.write_to_csv(output_filename)
    output_filename = output_filename.replace('.csv', '.png')
    if simulation['Show_plot'] == 'yes':
        show_plot = True
    else:
        show_plot = False
    dam_failure.create_plot(filepath=output_filename,
                            show_plot=show_plot,
                            save_plot=True)
    max_flow = dam_failure.max_flow
    max_level = dam_failure.max_level
    del dam_failure
    return max_flow, max_level


if __name__ == "__main__":
    main()
