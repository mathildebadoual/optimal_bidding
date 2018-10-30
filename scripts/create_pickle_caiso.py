"""
Store CAISO requests into a dataframe stored into a pickle file
"""
from pyiso import client_factory
import pandas as pd
import datetime
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default="20150601")
    parser.add_argument('--end_date', type=str, default="20181101")
    parser.add_argument('--which', type=str, default='all',
                        help="choose between 'all', 'gen' and 'load'")
    args = parser.parse_args()

    caiso = client_factory('CAISO')

    start_date = datetime.datetime(int(args.start_date[0:4]),
                                   int(args.start_date[4:6]),
                                   int(args.start_date[6:8]))
    end_date = datetime.datetime(int(args.end_date[0:4]),
                                 int(args.end_date[4:6]),
                                 int(args.end_date[6:8]))

    if args.which == 'all' or args.which == 'gen':
        all_data = []
        date_0 = start_date
        date_1 = start_date + datetime.timedelta(days=30)
        pbar = tqdm(total=(end_date - start_date).days/30)
        while date_1 <= end_date:
            all_data += caiso.get_generation(start_at=date_0, end_at=date_1)
            date_0, date_1 = date_1, date_1 + datetime.timedelta(days=30)
            pbar.update(1)
        pbar.close()

        df = pd.DataFrame(all_data)
        df.to_pickle("gen_caiso.pkl")

    if args.which == 'all' or args.which == 'load':
        all_data = []
        date_0 = start_date
        date_1 = start_date + datetime.timedelta(days=30)
        pbar = tqdm(total=(end_date - start_date).days / 30)
        while date_1 <= end_date:
            all_data += caiso.get_load(start_at=date_0, end_at=date_1)
            date_0, date_1 = date_1, date_1 + datetime.timedelta(days=30)
            pbar.update(1)
        pbar.close()

        df = pd.DataFrame(all_data)
        df.to_pickle("dem_caiso.pkl")


if __name__ == "__main__":
    main()
