import pandas as pd
import numpy as np

def cleaning(fp):
    """
    pass in allegations raw data to clean and output cleaned csv
    """
    allegations = pd.read_csv(fp)
    # replace complainant ages less than 0 with nans, since this doesnt make sense
    allegations["complainant_age_incident"] = allegations["complainant_age_incident"].mask(
        allegations["complainant_age_incident"] < 0, np.nan)
    # replace complainant ages 8 or less with nans, since it is likely there are no kids
    # filing complainants; are likely misinputs
    allegations["complainant_age_incident"] = allegations["complainant_age_incident"].mask(
        allegations["complainant_age_incident"] <= 8, np.nan).astype("Int8")
    # there is no 0th or 1000th precinct in NY, fill with nan
    allegations["precinct"] = allegations["precinct"].replace(0, np.nan).replace(1000, np.nan).astype("Int16")
    # convert "Unknown"s to nans
    allegations["complainant_ethnicity"] = allegations["complainant_ethnicity"].replace("Unknown", np.nan)
    # create substantiated column (serves as boolean for prediction)
    allegations["Substantiated"] = allegations["board_disposition"].str.startswith("Sub")
    # drop columns unnecessary for application of fairness intervention
    allegations = allegations.drop(columns = [
        "first_name", "last_name", "command_now", "month_received", "year_received", 
        "month_closed", "year_closed", "command_at_incident", "rank_abbrev_incident",
        "rank_abbrev_now", "rank_now", "rank_incident", "outcome_description", "board_disposition"])
    return allegations
    