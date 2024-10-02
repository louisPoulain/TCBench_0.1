import numpy as np
import argparse



def main(date, time, lead_time):
    start_nc = np.datetime64(f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{time[0:2]}")
    end_nc = np.datetime64(start_nc + np.timedelta64(lead_time, 'h'))
    s = f"{start_nc}_to_{end_nc}_ldt_{lead_time}"
    return s

parser = argparse.ArgumentParser()

parser.add_argument("--date", type=str, help="date", default="20000101")
parser.add_argument("--time", type=str, help="time", default="0000")
parser.add_argument("--lead_time", type=int, help="lead time", default=6)

args = parser.parse_args()

date, time, lead_time = args.date, args.time, args.lead_time

print(main(date, time, lead_time), flush=True)
    

