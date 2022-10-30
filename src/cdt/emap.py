from collections import namedtuple
import copy

EvDef = namedtuple('EvDef', ['type', 'value'])
class EventDefinitionMap: # eid -> evdef
    def __init__(self, top_dt, end_dt):
        self.top_dt = top_dt
        self.end_dt = end_dt
        self._emap = {} # key : eid, val : evdef
        self._ermap = {} # key : evdef, val : eid

    def __len__(self):
        return len(self._emap)

    def _eids(self):
        return self._emap.keys()

    def _next_eid(self):
        eid = len(self._emap)
        while eid in self._emap:
            eid += 1
        else:
            return eid

    def get_evdef(self, eid) -> EvDef:
        return self._emap[eid]

    def get_eid(self, evdef):
        return self._ermap[evdef]

    def pop(self, eid):
        evdef = self._emap.pop(eid)
        self._ermap.pop(evdef)
        return evdef

    def info(self, eid):
        return self._emap[eid]

    def move_eid(self, old_eid, new_eid):
        evdef = self.pop(old_eid)
        self._emap[new_eid] = evdef
        self._ermap[evdef] = new_eid
    
    def update_event(self, eid, type, value):
        d = {"type" : type,
             "value" : value,
            }
        evdef = EvDef(**d)

        old_evdef = self._emap[eid]
        self._ermap.pop(old_evdef)

        self._emap[eid] = evdef
        self._ermap[evdef] = eid
        return eid

    def process_row(self, columns, row):
        row_eids = []
        for column in columns:
                if column == '@timestamp':
                    continue
                value = row[column]
                if value == "":
                    continue
                d = {
                    "type": column,
                    "value": row[column],
                }

                evdef = EvDef(**d)

                if evdef in self._ermap:
                    row_eids.append(self._ermap[evdef])
                else:
                    eid = self._next_eid()
                    self._emap[eid] = evdef
                    self._ermap[evdef] = eid
                    row_eids.append(eid)
        return row_eids

def _copy_evmap(evmap):
    new_evmap = EventDefinitionMap(evmap.top_dt, evmap.end_dt,
            evmap.gid_name)
    new_evmap._emap = copy.deepcopy(evmap._emap)
    new_evmap._ermap = copy.deepcopy(evmap._ermap)
    return new_evmap
