from preparator import Preparator
from config import preparation_config as config
import numpy as np


def left_right_task_data_preparation(feature_extraction=False, verbose=False):
    # We use the antisaccade dataset to extract the data for left and right benchmark task.
    saccade = config['saccade_trigger']
    fixation = config['fixation_trigger']
    cue = config['antisaccade']['cue_trigger']

    # if not feature_extraction:
    preparator = Preparator(
        load_file_pattern=config['ANTISACCADE_FILE_PATTERN'],
        save_file_name=config['output_name'], verbose=verbose)
    preparator.extract_data_at_events(extract_pattern=[cue, saccade, fixation], name_start_time='Beginning of cue', start_time=lambda events: events['latency'],
                                      name_length_time='Size blocks of 500', length_time=500,
                                      start_channel=1, end_channel=129, padding=False)

    # preparator = Preparator(load_directory=config['LOAD_ANTISACCADE_PATH'],
    #                         save_directory=config['SAVE_PATH'],
    #                         load_file_pattern=config['ANTISACCADE_FILE_PATTERN'],
    #                         save_file_name=config['output_name'], verbose=verbose)
    # preparator.extract_data_at_events(extract_pattern=[cue, saccade, fixation], name_start_time='Beginning of cue', start_time=lambda events: events['latency'],
    #                                   name_length_time='Size blocks of 500', length_time=500,
    #                                   start_channel=1, end_channel=129, padding=False)


    # else:
    #     preparator = Preparator(load_directory=config['LOAD_ANTISACCADE_PATH'],
    #                             save_directory=config['SAVE_PATH'],
    #                             load_file_pattern=config['ANTISACCADE_HILBERT_FILE_PATTERN'],
    #                             save_file_name=config['output_name'], verbose=verbose)
    #     preparator.extract_data_at_events(extract_pattern=[cue, saccade, fixation], name_start_time='At saccade on-set', start_time=lambda events: events['latency'].shift(-1),
    #                                                                                 name_length_time='Fixed blocks of 1', length_time=1,
    #                                                                                 start_channel=1, end_channel=258, padding=False)

    # take only blocks of pro-saccade
    preparator.blocks(on_blocks=['20'], off_blocks=['30'])
    preparator.addFilter(name='Keep right direction', f=lambda events: (events['type'].isin(['10']) & events['type'].shift(-1).isin(saccade) & (events['end_x'].shift(-1) < 400))
                         | (events['type'].isin(['11']) & events['type'].shift(-1).isin(saccade) & (events['end_x'].shift(-1) > 400)))
    preparator.addFilter(name='Keep saccade if it comes after a reasonable reaction time',
                         f=lambda events: events['latency'].shift(-1) - events['latency'] > 50)
    preparator.addFilter(name='Keep only the ones with big enough amplitude',
                         f=lambda events: events['amplitude'].shift(-1) > 2)
    preparator.addLabel(name='Giving label 0 for left and 1 for right',
                        f=lambda events: events['type'].apply(lambda x: 0 if x == '10' else 1))
    preparator.run()


def main():
    if config['task'] == 'LR_task':
        if config['dataset'] == 'antisaccade':
            left_right_task_data_preparation(config['feature_extraction'])
        else:
            raise ValueError(
                "This task cannot be prepared (is not implemented yet) with the given dataset.")


main()
