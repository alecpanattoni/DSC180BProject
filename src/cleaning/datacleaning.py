import pandas as pd
import numpy as np

def cleaning(fp):
    """
    pass in allegations raw data to clean and output cleaned csv
    """
    allegations = pd.read_csv(fp)
    # replace complainant ages 8 or less with nans, since it is likely there are no kids
    # filing complainants; are likely misinputs
    allegations["complainant_age_incident"] = allegations["complainant_age_incident"].mask(
        allegations["complainant_age_incident"] <= 8, np.nan).astype(float)
    # convert "Unknown"s to nans
    allegations["complainant_ethnicity"] = allegations["complainant_ethnicity"].replace("Unknown", np.nan)
    # create substantiated column (serves as boolean for prediction)
    allegations["substantiated"] = allegations["board_disposition"].str.startswith("Sub")
    # drop columns unnecessary for application of fairness intervention
    allegations = allegations[['complaint_id', 'complainant_ethnicity', 'complainant_age_incident',
       'allegation', 'contact_reason', 'substantiated']]
    # start with no missing data
    allegations = allegations.dropna()
    return allegations
    
