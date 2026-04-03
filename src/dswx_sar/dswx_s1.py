#!/usr/bin/env python

import copy
import logging
import time

from joblib import Parallel, delayed

from dswx_sar.sentinel1 import (masking_with_ancillary, mosaic_rtc_burst,
                                pre_processing, save_mgrs_tiles,
                                initial_threshold,
                                region_growing,
                                refine_with_bimodality,
                                )
from dswx_sar.common import _initial_threshold as initial_threshold
from dswx_sar.common import _fuzzy_value_computation as fuzzy_value_computation
from dswx_sar.common import _detect_inundated_vegetation as detect_inundated_vegetation
from dswx_sar.sentinel1.dswx_runconfig import (_get_parser,
                                     RunConfig,
                                     DSWX_S1_POL_DICT)
from dswx_sar.common import _generate_log

logger = logging.getLogger('dswx_sar')


def dswx_s1_workflow(cfg):

    t_all = time.time()
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_mode = processing_cfg.polarization_mode
    input_list = cfg.groups.input_file_group.input_file_path
    dswx_workflow = processing_cfg.dswx_workflow
    inundated_veg_cfg = processing_cfg.inundated_vegetation

    logger.info("")
    logger.info("Starting DSWx-S1 algorithm")
    logger.info(f"Number of RTC products: {len(input_list)}")
    logger.info(f"Polarizations : {pol_list}")

    # Create mosaic burst RTCs
    t_stage = time.time()
    mosaic_rtc_burst.run(cfg)
    logger.info(f'  [profile] mosaic_rtc_burst: {time.time() - t_stage:.1f}s')

    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['DV_POL'],
                        DSWX_S1_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['SV_POL'],
                        DSWX_S1_POL_DICT['SH_POL']]
    elif pol_mode == 'MIX_DUAL_H_SINGLE_V_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['DH_POL'],
                        DSWX_S1_POL_DICT['SV_POL']]
    elif pol_mode == 'MIX_DUAL_V_SINGLE_H_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['DV_POL'],
                        DSWX_S1_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]

    def _run_pol_set(cfg_orig, pol_set):
        """Run the full processing pipeline for one polarization set."""
        cfg_copy = copy.deepcopy(cfg_orig)
        cfg_copy.groups.processing.polarizations = pol_set

        t_stage = time.time()
        pre_processing.run(cfg_copy)
        logger.info(f'  [profile] pre_processing ({pol_set}): {time.time() - t_stage:.1f}s')

        t_stage = time.time()
        initial_threshold.run(cfg_copy)
        logger.info(f'  [profile] initial_threshold ({pol_set}): {time.time() - t_stage:.1f}s')

        t_stage = time.time()
        fuzzy_value_computation.run(cfg_copy)
        logger.info(f'  [profile] fuzzy_value_computation ({pol_set}): {time.time() - t_stage:.1f}s')

        t_stage = time.time()
        region_growing.run(cfg_copy)
        logger.info(f'  [profile] region_growing ({pol_set}): {time.time() - t_stage:.1f}s')

        if dswx_workflow == 'opera_dswx_s1':
            t_stage = time.time()
            masking_with_ancillary.run(cfg_copy)
            logger.info(f'  [profile] masking_with_ancillary ({pol_set}): {time.time() - t_stage:.1f}s')

            t_stage = time.time()
            refine_with_bimodality.run(cfg_copy)
            logger.info(f'  [profile] refine_with_bimodality ({pol_set}): {time.time() - t_stage:.1f}s')

            if ((inundated_veg_cfg.enabled == 'auto') and
               len(pol_set) >= 2) or \
               inundated_veg_cfg.enabled is True:

                t_stage = time.time()
                detect_inundated_vegetation.run(cfg_copy)
                logger.info(f'  [profile] detect_inundated_vegetation ({pol_set}): {time.time() - t_stage:.1f}s')

    if len(proc_pol_set) > 1:
        Parallel(n_jobs=len(proc_pol_set))(
            delayed(_run_pol_set)(cfg, pol_set)
            for pol_set in proc_pol_set
        )
    else:
        _run_pol_set(cfg, proc_pol_set[0])

    processing_cfg.polarizations = pol_list
    # save product as mgrs tiles.
    t_stage = time.time()
    save_mgrs_tiles.run(cfg)
    logger.info(f'  [profile] save_mgrs_tiles: {time.time() - t_stage:.1f}s')

    t_time_end = time.time()
    logger.info(f'total processing time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()
    args = parser.parse_args()
    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)
    _generate_log.configure_log_file(cfg.groups.log_file)

    dswx_s1_workflow(cfg)


if __name__ == '__main__':
    '''run dswx_s1 from command line'''
    main()
