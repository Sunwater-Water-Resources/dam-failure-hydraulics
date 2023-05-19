import os.path

from DamFailure import FailureEvent
import pandas as pd


def main():
    # Set up the simulations
    simulation_list = 'Simulation_List.xlsx'
    list_of_max_flows = {'Create': True, 'Filename': 'results/max_flows.csv'}

    # get the data
    if '.xlsx' in simulation_list:
        simulation_df = pd.read_excel(simulation_list, sheet_name='Simulation_List')
    else:
        simulation_df = pd.read_csv(simulation_list)
    # simulation_df = simulation_df.set_index('Run_ID')  # set the index column
    all_max_flows_df = pd.DataFrame()
    for run_id, simulation in simulation_df.iterrows():
        if simulation['Include'] == 'yes':
            max_flow = run_failure_simulation(simulation)
            if list_of_max_flows['Create']:
                max_flow_dict = {'Simulation': str(simulation['Simulation_name']),
                                 'Failure Elevation': str(simulation['Failure_elevation']),
                                 'Maximum Flow': max_flow}
                max_flows_df = pd.DataFrame.from_dict([max_flow_dict])
                all_max_flows_df = pd.concat([all_max_flows_df, max_flows_df], axis=0)
                all_max_flows_df.to_csv(list_of_max_flows['Filename'])


def run_failure_simulation(simulation):

    # Create a new dam failure event
    log_filename = str(simulation['Output_name']).replace('.csv', '.txt')
    dam_failure = FailureEvent(name=simulation['Simulation_name'],
                               log_filename=log_filename)
    dam_failure.log_message('\n!!!!!\nRunning simulation for: {}\n!!!!!\n'.format(simulation['Simulation_name']))
    dam_failure.log_message(simulation)

    # Set up the structure of the dam
    dam_failure.import_dam_structure(simulation['Dam_structure_file'])

    # Set up the dam failure event properties
    dam_failure.import_event_properties(simulation['Event_file'])

    # Get the inflows
    dam_failure.import_inflow(inflow_file=simulation['Inflow_file'],
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
    # print(dam_failure)

    # Save the results
    output_filename = simulation['Output_name']
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
    del dam_failure
    return max_flow


if __name__ == "__main__":
    main()
