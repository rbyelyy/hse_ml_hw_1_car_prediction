"""
Utility functions for prediction service.
"""
import numpy as np
import pandas as pd
import pickle as pi

from sklearn.preprocessing import OneHotEncoder

def split_torque(df):
    """
    Split torque column and return data set with two additional columns received from torque
    """
    # if we take a look on the possible torque values here
    # https://www.researchgate.net/figure/The-raw-data-set-for-20-automobile-engines-Torque-versus-RPM_fig1_264228339
    # we can notice that it is starting from 90+

    # lets find min and max torque to set a range
    trq = pd.to_numeric(df['torque'].str.extract(r"(?P<torque>\d{1,}.?\d+)[nNm]").torque)
    print(f"Torque nNm - Min: {trq.min()}  Max: {trq.max()}")

    # let's find min and max kgm to set a range
    trq = pd.to_numeric(df['torque'].str.extract(r"(?P<torque>\d{1,}.?\d+)[kgm]").torque)
    print(f"Torque kgm - Min: {trq.min()}  Max: {trq.max()}")

    # if torque < 51 we can * 9.8
    df['torque'] = df['torque'].str.replace(',', '')
    df['torque'] = df['torque'].str.replace('.', '')

    df['min_torque_rpm'] = df['torque'].str.extract(r"^(?P<torque_new>\d{1,}\.\d+|\d+).*$").torque_new
    df['rpm'] = df['torque'].str.extract(r"^.*(\@ |at )(?P<rpm>\d+).*$").rpm

    # let's move to numeric
    df['min_torque_rpm'] = pd.to_numeric(df['min_torque_rpm'])
    df['rpm'] = pd.to_numeric(df['rpm'])

    #  lets time all torque value < 51 to 9.8
    df['min_torque_rpm'] = df['min_torque_rpm'].apply(lambda _: _ * 9.8 if _ < 51 else _)

    return df

def data_clean(df):
    """
    Clean up 'mileage', 'engine', 'max_power' columns to be converted to numeric
    """
    # let's remove spaces make lower case etc.
    for _ in ['mileage', 'engine', 'max_power']:
        df[_] = df[_].str.lower()
        df[_] = df[_].str.strip()
        df[_] = df[_].str.replace(" ", "")

    # lets remove all possible string patterns
    df['mileage'] = df['mileage'].str.replace("kmpl", "")
    df['mileage'] = df['mileage'].str.replace("km/kg", "")
    df['max_power'] = df['max_power'].str.replace("bhp", "")
    df['engine'] = df['engine'].str.replace("cc", "")

    # let's cast to float
    for _ in ['mileage', 'engine', 'max_power']:
        df[_] = pd.to_numeric(df[_], errors='coerce')

    return df
def load_model(filename):
    """
    Load model for pickle file
    """
    with open(filename, 'rb') as f:
        model = pi.load(f)
    return model

def feature_adjust(df):
    """
    Adjust year and selling price (for year pow 2 and for selling price log10)
    """
    if df.shape[1] > 1:
        df['year'] = df['year']**2
    else:
        df['selling_price'] = np.log10(df['selling_price'])
    return df
def encode(df, features, load_encoder_file=False, save_encoder=True):
    """
    Perform one hot feature codding with the ability to save and upload results from pickle file
    """
    if not load_encoder_file:
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_encoder.fit(df[features])
        if save_encoder:
            with open('encoder.pickle', 'wb') as f:
                pi.dump(ohe_encoder, f)
    else:
        with open('encoder.pickle', 'rb') as f:
            ohe_encoder = pi.load(f)
    ohe_cols_test = pd.DataFrame(ohe_encoder.transform(df[features]))
    ohe_cols_test.columns = ohe_encoder.get_feature_names_out()
    return df.join(ohe_cols_test)
def data_preprocessing(df):
    """
    Split torque, drop unused features, ecode with OHE and adjust year and selling_price columns.
    """
    df = split_torque(df)
    df = df.drop(['name', 'selling_price', 'torque'], axis=1)
    df = data_clean(df=df)
    df = encode(df=df, features=['seats', 'fuel', 'owner', 'transmission', 'seller_type'],
                load_encoder_file=True, save_encoder=False)
    df = feature_adjust(df=df)
    return df

def run_model(model, df):
    """
    Run pre-trained model and return the original data frame with new predicted_price column.
    """
    prediction = model.predict(df.select_dtypes([np.number]))
    predict_price = np.round(10 ** prediction, 0)
    df['predicted_price'] = predict_price
    return df





