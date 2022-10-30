import copy
import util
import src.cdt.emap as emap
import fourier
import evfilter

dt_conds = ['30s']
# threshold for continuous distribution dt filter
count_linear = 10
binsize_linear = "10s"
threshold_linear = 0.5

def filter_edict(edict, evmap, top_dt, end_dt, d_stat):
    # replace
    edict, evmap = replace_edict(edict, evmap, d_stat, top_dt, end_dt)
    # linear
    edict, evmap = filter_linear(edict, evmap, top_dt, end_dt)
    return edict, evmap

def filter_linear(edict: dict, evmap: emap.EventDefinitionMap, top_dt, end_dt):
    ret_edict = copy.deepcopy(edict)
    ret_evmap = emap._copy_evmap(evmap)

    for eid, l_dt in edict.items():
        if len(l_dt) < count_linear:
            continue
        ret = evfilter.remove_dist(l_dt, top_dt, end_dt, util.str2dur(binsize_linear), threshold_linear)
        if ret:
            ret_edict.pop(eid)
            ret_evmap.pop(eid)
    return _remap_eid(ret_edict, ret_evmap)

def replace_edict(edict: dict, evmap: emap.EventDefinitionMap, d_stat: dict, top_dt, end_dt):

    def revert_event(data, top_dt, end_dt, binsize):
        assert top_dt + len(data) * binsize == end_dt
        return [top_dt + i * binsize for i, val in enumerate(data) if val > 0]

    ret_edict = copy.deepcopy(edict)
    ret_evmap = emap._copy_evmap(evmap)
    s_eid_periodic = set()

    for dt_cond in dt_conds: 
        binsize = util.str2dur(dt_cond)
        
        for eid, l_stat in d_stat.items():
            if eid in s_eid_periodic or not fourier.pretest(l_stat, binsize):
                pass
            else:
                flag, remain_data, interval = fourier.replace(l_stat, binsize)
                if flag:
                    s_eid_periodic.add(eid)
                    if sum(remain_data) == 0:
                        ret_edict.pop(eid)
                        ret_evmap.pop(eid)
                    else:
                        ret_edict[eid] = revert_event(remain_data,
                                top_dt, end_dt, binsize)
                        ret_evmap.update_event(eid, evmap.info(eid),
                                evmap.EventDefinitionMap.type_periodic_remainder,
                                int(interval.total_seconds()))
                else:
                    pass

    return _remap_eid(ret_edict, ret_evmap)

def _remap_eid(edict: dict, evmap: emap.EventDefinitionMap):
    new_eid = 0
    for old_eid in edict.keys():
        if old_eid == new_eid:
            new_eid += 1
        else:
            temp = edict[old_eid]
            edict.pop(old_eid)
            edict[new_eid] = temp
            evmap.move_eid(old_eid, new_eid)
            new_eid += 1

    return edict, evmap