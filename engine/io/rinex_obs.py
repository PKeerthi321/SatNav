import gzip
import pandas as pd
from datetime import datetime

# -----------------------
# RINEX OBS reader (fallback, gz-compatible)
# -----------------------
def _open_text_file(path):
    if path.lower().endswith('.gz'):
        return gzip.open(path, 'rt', errors='ignore')
    else:
        return open(path, 'r', errors='ignore')

def read_rinex_obs_fallback(obsfile):
    rows = []
    try:
        with _open_text_file(obsfile) as f:
            header = []
            # read header
            while True:
                line = f.readline()
                
                if not line:
                    break
                header.append(line.rstrip('\n'))
                if 'END OF HEADER' in line:
                    break
            # parse SYS / # / OBS TYPES lines
            sys_obs_types = {}
            for ln in header:
                if len(ln) >= 60 and 'SYS / # / OBS TYPES' in ln:
                    sys = ln[0]
                    types_line = ln[7:60]
                    types = [t for t in types_line.split() if t.strip()!='']
                    if sys in sys_obs_types:
                        sys_obs_types[sys].extend(types)
                    else:
                        sys_obs_types[sys] = types
            # read epochs
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.startswith('>'):
                    continue
                parts = line[1:29].split()
                if len(parts) < 6:
                    continue
                try:
                    year = int(parts[0]); month=int(parts[1]); day=int(parts[2])
                    hour=int(parts[3]); minute=int(parts[4]); sec=float(parts[5])
                except Exception:
                    continue
                epoch_time = datetime(year,month,day,hour,minute,int(sec))
                # num sats
                num_sats = 0
                if len(line) >= 35:
                    try:
                        num_sats = int(line[32:35])
                    except:
                        num_sats = 0
                for s in range(num_sats):
                    satline = f.readline()
                    if not satline:
                        break
                    if len(satline.strip()) == 0:
                        continue
                    sat_id = satline[0:3].strip()
                    sys = sat_id[0] if len(sat_id)>0 else '?'
                    obs_types = sys_obs_types.get(sys, None)
                    data_block = satline[3:].rstrip('\n')
                    if obs_types:
                        expected = len(obs_types)
                        while len(data_block) < expected*16:
                            nxt = f.readline()
                            if not nxt:
                                break
                            data_block += nxt.rstrip('\n')
                        for idx, obs in enumerate(obs_types):
                            start = idx*16
                            raw = data_block[start:start+16].strip() if start < len(data_block) else ''
                            if raw == '':
                                val = float('nan')
                            else:
                                raw_s = raw.replace('D','E')
                                try:
                                    val = float(raw_s)
                                except:
                                    val = float('nan')
                            rows.append((epoch_time, sat_id, sys, obs, val))
                    else:
                        # fallback parse upto 5 fields
                        for idx in range(5):
                            start = idx*16
                            raw = data_block[start:start+16].strip() if start < len(data_block) else ''
                            try:
                                val = float(raw.replace('D','E'))
                            except:
                                val = float('nan')
                            rows.append((epoch_time, sat_id, sys, f'OBS{idx+1}', val))
    except Exception as e:
        raise IOError(f"OBS read failed: {e}")
    df = pd.DataFrame(rows, columns=['UTC_Time','Sat','Sys','ObsType','Value'])
    return df

def read_rinex_obs(obsfile):
    # keep dependency-free fallback reader
    return read_rinex_obs_fallback(obsfile)
