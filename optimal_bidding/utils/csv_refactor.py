import os
import pandas as pd
import csv
from os.path import isfile, join, abspath, dirname
import warnings


def consolidate_csvs(energy_data_path, fcas_data_path, output_folder, output_prefix, price_column_name = "RRP", region="SA1"):
    """
    Takes a folder with MMS csvs and create a new csv
    with just the demand and energy data.
    Assumes that all csvs in the directory are
    to be used and follow the MMS format.
    Warns the user if there are missing datetimes.

    Args:
      data_path: string
          Absolute path to directory containing csvs with MMS data.
      output_folder: string
          Absolute path to directory where outputted csvs will be created.
      output_prefix: string
          Prefix for the filename of the outputted csvs.
      region: string
          RegionID for the desired region data.

    Returns:
      None
    """

    five_min_df = pd.DataFrame(
        columns=["Timestamp", "Region", "Energy_Price", "Energy_Demand", "5min_Raise_Demand", "5min_Lower_Demand"])
    thirty_min_df = pd.DataFrame(
        columns=["Timestamp", "Region", "Energy_Price", "Energy_Demand", "5min_Raise_Demand", "5min_Lower_Demand"])
    fcas_df = pd.DataFrame(
        columns = ["Timestamp", "5min_Raise_Price", "5min_Lower_Price"])

    # grab csvs from the specified energy data folder
    onlycsvs = [
        join(energy_data_path, f) for f in os.listdir(energy_data_path)
        if isfile(join(energy_data_path, f)) and f.lower().endswith(".csv")
    ]
    for csv_name in onlycsvs:
        print("Reading {}".format(csv_name.split("/")[-1]))
        with open(csv_name) as csvfile:
            reader = csv.reader(csvfile)

            demand_index = None
            price_index = None

            raise_demand_index = None
            lower_demand_index = None

            timestamp_index = None
            region_index = None

            freq = None

            for row in reader:
                if row[0] == "C":
                    # logging rows are useless
                    pass
                elif row[0] == "I":
                    # header row (sometimes the format of the csv changes
                    # in the middle so there can be multiple header rows)
                    demand_index = row.index("TOTALDEMAND")
                    raise_demand_index = row.index("RAISE5MINLOCALDISPATCH")
                    lower_demand_index = row.index("LOWER5MINLOCALDISPATCH")


                    price_index = row.index(price_column_name)
                    timestamp_index = row.index("SETTLEMENTDATE")
                    region_index = row.index("REGIONID")
                    if row[1] == "DREGION":
                        freq = 5
                    elif row[1] == "TREGION":
                        freq = 30
                    else:
                        freq = None
                elif row[0] == "D":
                    # data row
                    data = {}
                    data["Timestamp"] = pd.to_datetime(row[timestamp_index])
                    data["Region"] = row[region_index]
                    data["Energy_Price"] = row[price_index]
                    data["Energy_Demand"] = row[demand_index]
                    data["5min_Raise_Demand"] = row[raise_demand_index]
                    data["5min_Lower_Demand"] = row[lower_demand_index]
                    if row[region_index] == region:
                        if freq == 5:
                            five_min_df = five_min_df.append(data,
                                                             ignore_index=True)
                        elif freq == 30:
                            thirty_min_df = thirty_min_df.append(
                                data, ignore_index=True)
                        else:
                            warnings.warn(
                                "Unrecognized frequency in {}. Ignoring row.".
                                format(csv_name), UserWarning)

                else:
                    warnings.warn(
                        "Unrecognized row type in {}. Ignoring.".format(
                            csv_name), UserWarning)
    # drop duplicates
    five_min_df = five_min_df.drop_duplicates(subset="Timestamp")
    thirty_min_df = thirty_min_df.drop_duplicates(subset="Timestamp")

    # set index
    five_min_df = five_min_df.set_index("Timestamp")
    thirty_min_df = thirty_min_df.set_index("Timestamp")

    # sort by date
    five_min_df = five_min_df.sort_index()
    thirty_min_df = thirty_min_df.sort_index()




    # grab csvs from the specified FCAS folder
    onlycsvs = [
        join(fcas_data_path, f) for f in os.listdir(fcas_data_path)
        if isfile(join(fcas_data_path, f)) and f.lower().endswith(".csv")
    ]
    for csv_name in onlycsvs:
        print("Reading {}".format(csv_name.split("/")[-1]))
        with open(csv_name) as csvfile:
            reader = csv.reader(csvfile)

            raise_price_index = None
            lower_price_index = None
            timestamp_index = None
            region_index = None


            for row in reader:
                if row[0] == "C":
                    # logging rows are useless
                    pass
                elif row[0] == "I":
                    # header row (sometimes the format of the csv changes
                    # in the middle so there can be multiple header rows)

                    raise_price_index = row.index("RAISE5MINRRP")
                    lower_price_index = row.index("LOWER5MINRRP")
                    timestamp_index = row.index("SETTLEMENTDATE")
                    region_index = row.index("REGIONID")

                elif row[0] == "D":
                    # data row
                    data = {}
                    data["Timestamp"] = pd.to_datetime(row[timestamp_index])
                    data["5min_Raise_Price"] = row[raise_price_index]
                    data["5min_Lower_Price"] = row[lower_price_index]
                    if row[region_index] == region:
                        fcas_df = fcas_df.append(data, ignore_index=True)


                else:
                    warnings.warn(
                        "Unrecognized row type in {}. Ignoring.".format(
                            csv_name), UserWarning)



    # write to specified output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fcas_df = fcas_df.drop_duplicates(subset="Timestamp")
    fcas_df = fcas_df.set_index("Timestamp")
    fcas_df = fcas_df.sort_index()

    five_min_df = five_min_df.join(fcas_df, how="inner")
    thirty_min_df = thirty_min_df.join(fcas_df, how="inner")

    five_min_df.to_csv(join(output_folder, output_prefix + "_5min.csv"))
    thirty_min_df.to_csv(join(output_folder, output_prefix + "_30min.csv"))



if __name__ == '__main__':
    dir_path = dirname(dirname(abspath(__file__)))
    static_path = join(dir_path, 'static')
    output_folder = join(static_path, 'consolidated_data')
    energy_path = join(static_path, 'energy_csvs')
    fcas_path = join(static_path, "fcas_data")
    output_prefix = 'FiveMonths2018'

    consolidate_csvs(energy_path,
                     fcas_path,
                     output_folder,
                     output_prefix,
                     "RRP",
                     region="SA1")
