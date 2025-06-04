import csv
import datetime as dt
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytaf  # TAF decoder
import requests
from bs4 import BeautifulSoup
from keras.layers import (
    LSTM,
    Activation,
    Add,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    SimpleRNN,
)

# from tensorflow import _keras_module
from keras.models import Model, Sequential
from metar.Metar import Metar  # METAR decoder
from sklearn.ensemble import RandomForestClassifier as Bosque
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split as separar
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


###################################################################################
###################################################################################
###################################################################################
def df_filter(df, filter_column, filter):
    """
    This function is intended to filter a table by the specified column name with the chosen filter.

    Inputs:

        1. 'df' (DataFrame): Input DataFrame
        2. 'filter_column' (str): Filter column name
        3. 'filter' (str): Filter type inside the filter column name

    Outputs:

        1. 'df_filtered' (DataFrame): Output filtered DataFrame
    """
    if filter_column in df:
        df_filtered = df[df[filter_column] == filter]
    else:
        print("Error. No column exists in the table")
    return df_filtered


###################################################################################
###################################################################################
###################################################################################
def airports_NMOC_fun(df_airports):
    """
    This function computes the name of every airport with at least
    1 regulation in the selected time span

    Inputs:

        1. Regulations dataframe with repeated airport values

    Outputs:

        1. csv with airports list
        2. Tuple with all non-repeated airports within NMOC region
    """
    airports_list = df_airports["Protected Location Id"].drop_duplicates()
    airports_list.to_csv("./Data/Regulations/Airports_regulation_list.csv", index=False)


###################################################################################
###################################################################################
###################################################################################
def download_METARs(csv_file, time_init, time_end, output_format="txt", output_csv_file=False):
    """
    Fetch METAR data for ICAO location codes from a CSV file within a specified date range.

    Inputs:

        1. 'csv_file' (str): Path to the CSV file containing ICAO location codes.
        2. 'time_init' (datetime.datetime): Initial date.
        3. 'time_end' (datetime.datetime): Final date.
        4. 'output_format' (str): Output format for METAR data ('html' or 'txt').
        5. 'output_csv_file' (str): Path to the output CSV file (optional).

    Outputs:

        1. 'dict' (dict): A dictionary with ICAO location codes as keys and METAR data as values.
    """

    # Define the URL of the form submission page
    url = "http://ogimet.com/display_metars2.php"

    # Read ICAO location codes from the CSV file
    icao_codes = []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            icao_codes.append(row["Protected Location Id"])

    # Initialize a dictionary to store METAR data
    metar_data_dict = {}

    # Generate date intervals from start_year to end_year, considering leap years
    start_year = time_init.year
    end_year = time_end.year
    start_month = time_init.month
    end_month = time_end.month

    for year in range(start_year, end_year + 1):
        for month in range(start_month, end_month + 1):
            last_day = 31  # Default to 31 days for most months

            # Check for February in a leap year
            if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                last_day = 29
            elif month == 2:
                last_day = 28

            # Adjust last_day for months with 30 days
            if month in [4, 6, 9, 11]:
                last_day = 30

            # Define parameters for the request for the current month
            params = {
                "tipo": "SA",
                "ord": "DIR",
                "nil": "SI",
                "fmt": output_format,
                "ano": year,
                "mes": month,
                "day": "01",
                "hora": "00",  # You can customize the time if needed
                "anof": year,
                "mesf": month,
                "dayf": last_day,
                "horaf": "00",
            }

            # Loop through the ICAO codes and make requests for the current month
            for icao in icao_codes:
                params["lugar"] = icao

                # Send the POST request to obtain the METAR data
                response = requests.post(url, data=params, timeout=30)
                html_content = response.text

                # Parse the HTML response using BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")

                # Extract and store the METAR data in the dictionary
                metar_data = soup.find("pre")
                metar_text = metar_data.get_text() if metar_data else None

                metar_data_dict.setdefault(icao, []).append(metar_text)

    # If an output CSV file path is provided, write the data to the CSV file
    if output_csv_file:
        with open(output_csv_file, "w", newrow="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Protected Location Id", "Report"])
            for icao, metars in metar_data_dict.items():
                for metar in metars:
                    writer.writerow([icao, metar])


###################################################################################
###################################################################################
###################################################################################
def download_TAFs(Input_file, output_file=False):
    # Crea new empty string to store values
    new_taf_format = ""

    # Open input file in read mode
    with open(Input_file) as file:
        # Variable to look for: "=" is the end of TAF
        found_equalsign = False

        # Read each row
        for row in file:
            # Delete all blank spaces at beginning and end of row
            row = row.strip()

            # If I find "=" that means we are at the end of TAF
            if "=" in row:
                found_equalsign = True

            # Add first character and the rest except release date in the new format
            new_taf_format += f"{row}" if found_equalsign else f"{row} "

            # If I find the "=" sign, add a new row. Otherwise, continue in the same row
            if found_equalsign:
                # The "\n" sign is used for this purpose
                new_taf_format += "\n"

                # Reset variable
                found_equalsign = False

    # Replace "" with a blank space because we do not want "" in the midst of the TAF
    new_taf_format = new_taf_format.replace('""', " ")

    # Open the output file in write mode
    with open(output_file, "w") as file:
        # Write the new format
        file.write(new_taf_format)

    # Extract dates from the desired lines
    dates = []
    with open(output_file) as file:
        for row in file:
            # Extract the date part
            date_str = row[0:11]
            # Convert to the desired format
            date = dt.datetime.strptime(date_str, "%Y%m%d%H%M")
            # Add to the desired format to the list
            dates.append(date.strftime("%Y-%m-%d %H:%M"))

    # Reopen the file to remove the dates from the lines
    with open(output_file) as file:
        rows = file.readlines()

    # Open the file in write mode to remove the dates
    with open(output_file, "w") as file:
        for row in rows:
            # Remove the date part starting from character 13
            file.write(row[13:])

    # Confirm the new format has been saved
    print(f"Saved new format in {output_file}.")

    return dates


###################################################################################
###################################################################################
###################################################################################
def filter_METARs(data_path, data_path_out, time_step, download_type):
    """
    This function is intended to filter all METARs from a single file

    Inputs:

        1. 'data_path'(str): Name of the original csv document
        2. 'data_path_out'(str): Name of the original csv document
        3. 'time_step' (int): Time between measurements (multiple of 5)
        4. 'download_type' (int): Whether download is done via webpage (0) or via already downloaded file (1)
    Outputs:

        1. 'df_metar' (DataFrame): Table with columns [ICAO_AD_code, Date, [filter_names]]
    """

    if download_type == 1:
        # Read the csv file and convert it into a dataframe
        metar_data_init = pd.read_csv(data_path)

        metar_data_init["Date"] = pd.to_datetime(
            metar_data_init[["year", "month", "day", "hour", "minute"]]
        )

        # Store METAR words (because the beginning of the file contains non-wanted information)
        metar_data = pd.DataFrame(columns=["Date", "Report"])
        for i in range(len(metar_data_init)):
            if "METAR" in metar_data_init["Report"].iloc[i]:
                date_value = metar_data_init["Date"].iloc[i]
                report_value = metar_data_init["Report"].iloc[i]
                metar_data.loc[len(metar_data)] = [date_value, report_value]

        ## Obtain the information of the metar in dataframe
        icao_code = metar_data_init["ICAO code"].iloc[0]

        times = []
        for row in metar_data["Report"]:
            # Busca una cadena que coincida con el formato de tiempo ('HHMMZ')
            time_match = re.search(r"\d{6}Z", row)
            if time_match:
                # Si se encuentra una coincidencia, agrega el tiempo encontrado a la lista
                times.append(time_match.group())

        # Ahora, 'times' contiene los tiempos encontrados en el METAR
        # Aquí almacenaremos las líneas de METAR modificadas
        # Aquí almacenaremos las líneas de METAR modificadas
        new_metar_report = []
        new_metar_date = []

        for i in range(0, len(times) - 1):
            current_time = metar_data["Date"].iloc[i]
            next_time = metar_data["Date"].iloc[i + 1]
            time_difference = (next_time - current_time).total_seconds()

            # Comprueba si la diferencia de tiempo es mayor a 30 minutos
            if abs(time_difference / 60) > time_step:
                multiples = int((abs(time_difference) / 60) / time_step)

                # Copia la primera fila a la fila adicional
                new_metar_report.append(metar_data["Report"].iloc[i])
                new_metar_date.append(metar_data["Date"].iloc[i])
                for _ in range(multiples - 1):
                    new_metar_report.append(metar_data["Report"].iloc[i])
                    new_metar_date.append(
                        (metar_data["Date"].iloc[i] + dt.timedelta(minutes=time_step)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )
            else:
                # Si no hay una brecha de tiempo grande, simplemente agrega la línea actual
                new_metar_report.append(metar_data["Report"].iloc[i])
                new_metar_date.append(metar_data["Date"].iloc[i])

        new_metar_report.append(metar_data["Report"].iloc[-1])
        new_metar_date.append(metar_data["Date"].iloc[-1])

        # Rewrite the array
        metar_data = pd.DataFrame({"Date": new_metar_date, "Report": new_metar_report})

        # METAR DECODING
        # Obtain a list for the visibility values for all the METARs
        metar_visibility = []

        # Obtain a list for the wind speed and direction values for all the METARs
        metar_wind = []

        # Obtain a list for the temperature values for all the METARs
        metar_temperature = []

        # Obtain a list for the cloud base height values for all the METARs
        metar_coverage = []
        metar_ceiling = []

        m_clouds = []
        m_winds = []
        m_temps = []
        for i in range(len(metar_data)):
            example = Metar(icao_code, metar_data["Report"].iloc[i])
            m_vis = example.getAttribute("visibility")
            if m_vis is None:
                m_vis = 10000
            elif m_vis == 9999:
                m_vis = 10000
            else:
                m_vis = m_vis
            # if not 'VV///' in metar_data[i]:

            m_wind = example.getAttribute("wind")
            m_temp = example.getAttribute("temperatures")
            m_cloud = example.getAttribute("cloud")

            metar_visibility.append(m_vis)
            m_temps.append(m_temp)
            m_winds.append(m_wind)
            m_clouds.append(m_cloud)

        ###################################
        # List only for the loop. It stores the wind information for each of the METAR codes and its initialized to 0 every iteration
        metar_wind_info = []
        metar_temp_info = []
        metar_ceiling_info = []
        metar_coverage_info = []

        for i in range(len(m_winds)):
            # Define an extra variable to read each of the clouds decomposition
            extra_var1 = m_temps[i]
            extra_var2 = m_winds[i]
            extra_var3 = m_clouds[i]
            if extra_var2 is not None:
                direction = extra_var2.get("direction")
                metar_wind_info.extend([direction, extra_var2.get("speed")])
            else:
                metar_wind_info = ["No info", 0]

            if extra_var1 is not None:
                temperature = extra_var1.get("temperature")
                metar_temp_info.extend([temperature, extra_var1.get("dewpoint")])
            else:
                metar_temp_info = metar_temperature[i - 1]

            if extra_var3 is not None:
                for j in range(len(extra_var3)):
                    code = extra_var3[j].get("code")
                    if code == "FEW":
                        code = 1
                    elif code == "SCT":
                        code = 2
                    elif code == "BKN":
                        code = 3
                    elif code == "OVC":
                        code = 4

                    metar_coverage_info = code
                    metar_ceiling_info = extra_var3[j].get("altitude")
            else:
                metar_coverage_info = 0
                metar_ceiling_info = 5000

            metar_wind.append(metar_wind_info)
            metar_temperature.append(metar_temp_info)
            metar_ceiling.append(metar_ceiling_info)
            metar_coverage.append(metar_coverage_info)

            # Restart the list
            metar_wind_info = []
            metar_temp_info = []
            metar_ceiling_info = []
            metar_coverage_info = []

        ############################
        ############################

        # Create two separate columns to differentiate between speed and direction of wind
        metar_wind_direction = []
        metar_wind_speed = []
        metar_temperature_value = []
        metar_dewpoint_value = []

        for i in range(len(metar_wind)):
            metar_wind_direction.append(metar_wind[i].__getitem__(0))
            metar_wind_speed.append(metar_wind[i].__getitem__(1))
            metar_temperature_value.append(metar_temperature[i].__getitem__(0))
            metar_dewpoint_value.append(metar_temperature[i].__getitem__(1))

        ###################################

        ##TABLE WITH METAR INFORMATION: Create a table (with pandas) with all the information of the METAR codes
        for _ in range(len(metar_data)):
            data_metar = {
                "Complete date": metar_data["Date"],
                "Horizontal visibility": metar_visibility,
                "Ceiling coverage": metar_coverage,
                "Ceiling height": metar_ceiling,
                "Wind Speed": metar_wind_speed,
                "Wind Direction": metar_wind_direction,
                "Temperature": metar_temperature_value,
                "Dewpoint": metar_dewpoint_value,
            }

        df_metar = pd.DataFrame(data_metar)
        df_metar.to_csv(data_path_out)

        return df_metar


###################################################################################
###################################################################################
###################################################################################
def filter_by_dates(regulation_file, metar_file):
    # Lee el archivo de regulación
    regulation_data = pd.read_csv(regulation_file)
    # regulation_dates = regulation_data["Initial Time"]

    # Lee el archivo METAR del aeropuerto deseado
    metar_data = pd.read_csv(metar_file, parse_dates=["Complete date"])

    # Filtra las filas en el archivo METAR que coincidan con las fechas del archivo de regulación
    filtered_metar_data = metar_data[
        metar_data["Complete date"].isin(regulation_data["Initial Time"])
    ]

    # Elimina la columna "Unnamed: 0" si está presente
    if "Unnamed: 0" in filtered_metar_data.columns:
        filtered_metar_data = filtered_metar_data.drop(columns=["Unnamed: 0"])

    return filtered_metar_data


###################################################################################
###################################################################################
###################################################################################
def filter_TAFs(Input_file):
    """
    Here TAFs are filtered to get the most relevant features

    Inputs:

        1. 'Input_file' (txt): Input file
        2.

    Outputs:

        1. 'Output_file': Filtered file
    """
    with open(Input_file) as file:
        for row in file:
            _ = pytaf.TAF(row)
            # valid_from_date = taf._taf_header(["valid_from_date"])


###################################################################################
###################################################################################
###################################################################################
def reg_binarizer(time_init, time_end, time_delta, reg_path, out_path):
    """
    This function is intended to compute whether there is an active regulation
    at a certain time at a given airport.

    Inputs:

        1. 'time_init'(datetime): Initial time
        2. 'time_end'(datetime): Final time
        3. 'time_delta'(float): Time separation between measurements
        4. 'reg_path'(DataFrame): Regulation file path dataframe
        5. 'out_path'(str): Output csv path

    Outputs:

        1. 'reg_binary_df'(DataFrame): DataFrame with 0 or 1 for each measurement for active regulations at each airport
    """
    # Create the target list with 1 and 0 values
    reg_binary_list = []
    times_list = []

    # Call regulations dataframe and switch to datetimelike values
    reg_path["Regulation Start Time"] = pd.to_datetime(reg_path["Regulation Start Time"])
    reg_path["Regulation End Date"] = pd.to_datetime(reg_path["Regulation End Date"])

    # Collect regulations starting and ending times
    start_time = reg_path["Regulation Start Time"]
    end_time = reg_path["Regulation End Date"]

    # Create a list for timestamp values type
    start_time_lista = start_time.tolist()
    end_time_lista = end_time.tolist()
    while time_init <= time_end:
        # Select lower and upper times to focus and tell if there is a regulation
        time_lower = time_init
        time_upper = time_init + dt.timedelta(minutes=time_delta)

        # Initialize a flag to check if a regulation is found for the current time window
        regulation_found = False

        for i in range(0, len(reg_path)):
            if (time_upper > start_time_lista[i] and start_time_lista[i] >= time_lower) or (
                start_time_lista[i] <= time_lower < end_time_lista[i]
            ):
                regulation_found = True
                break  # No need to check further if a regulation is found

        # After checking all regulations, if no regulation was found, append 0; otherwise, append 1
        if regulation_found:
            reg_binary_list.append(1)
        else:
            reg_binary_list.append(0)

        # Include row with initial time
        times_list.append(time_init)
        time_init = time_init + dt.timedelta(minutes=time_delta)

    # Convert the list to a DataFrame
    reg_binary_df = pd.DataFrame({"Initial Time": times_list, "Regulation Binary": reg_binary_list})

    # Save the DataFrame to a CSV file
    # Set index=False to exclude the index column
    reg_binary_df.to_csv(out_path, index=False)

    return reg_binary_df


###################################################################################
###################################################################################
###################################################################################
def Reg_binarizer_undersampling(file_paths, ADs, output_file):
    # Lista para almacenar los DataFrames de los archivos
    dfs = []

    # Leer los archivos y cargar los DataFrames en la lista
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Crear un conjunto de fechas únicas de todos los DataFrames
    all_dates = set()
    for df in dfs:
        all_dates.update(df[df["Regulation Binary"] == 1]["Initial Time"])

    # Crear un DataFrame final con todas las fechas únicas y las columnas de aeropuertos
    final_df = pd.DataFrame({"Initial Time": sorted(all_dates)})
    for airport in ADs:
        final_df[airport] = 0

    # Iterar sobre cada DataFrame y actualizar el DataFrame final
    for i, _ in enumerate(dfs):
        for _, row in df.iterrows():
            if row["Regulation Binary"] == 1:
                date = row["Initial Time"]
                # Configurar el valor en la columna del aeropuerto correspondiente a 1
                final_df.loc[final_df["Initial Time"] == date, ADs[i]] = 1

    final_df.to_csv(output_file, index=False)

    return final_df


###################################################################################
###################################################################################
###################################################################################
def Reg_binarizer_oversampling(Input_file, Output_file):
    """ """

    # Lee el archivo CSV en un DataFrame
    df = pd.read_csv(Input_file)

    # Itera a través de la columna "Regulation Binary"
    change_flag = False  # Esta bandera indica si ya hemos cambiado un 0 a 1
    for i in range(1, len(df)):
        if (df.loc[i, "Regulation Binary"] == 0 and df.loc[i - 1, "Regulation Binary"] == 1) or (
            df.loc[i, "Regulation Binary"] == 1 and df.loc[i - 1, "Regulation Binary"] == 0
        ):
            if not change_flag:
                df.loc[i, "Regulation Binary"] = 1
                df.loc[i + 1, "Regulation Binary"] = 1
                df.loc[i - 1, "Regulation Binary"] = 1
                df.loc[i - 2, "Regulation Binary"] = 1
                df.loc[i + 2, "Regulation Binary"] = 1
                df.loc[i + 3, "Regulation Binary"] = 1
                df.loc[i - 3, "Regulation Binary"] = 1
                change_flag = True
            else:
                change_flag = False
    df.to_csv(Output_file, index=False)


###################################################################################
###################################################################################
###################################################################################
def RF_model(X, y, n_trees, criterion, AD, test_size, max_depth, random_state, Output_path):
    """
    Trains a Random Forest (RF) classification model and displays performance metrics.

    Inputs:

        X (DataFrame): Input DataFrame.
        y (array): Target vector (1s and 0s for regulation prediction).
        n_trees (int): Number of trees in the forest.
        criterion (str): Criterion for splitting trees.
        AD (str): Aerodrome.
        test_size (float): Test size percentage (0-1).
        random_state (int): Random seed for reproducibility.
        flag_plot (bool): If True, display plots. Otherwise, no plots are displayed.
        output_path (str): Output path for saving plots.

    Outputs:

        cm (array): Confusion matrix.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = separar(X, y, test_size=test_size, random_state=random_state)

    # Scale the data
    # (You might want to use a scaler here if needed)

    # Create and train the Random Forest classifier
    classifier = Bosque(
        n_estimators=n_trees, criterion=criterion, max_depth=max_depth, random_state=random_state
    )
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Compute the confusion matrix
    cm = CM(y_test, y_pred)

    # Display performance metrics
    precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    f1_score = 2 / (1 / precision + 1 / recall)

    # Definir etiquetas para las clases
    class_names = ["Inactive", "Active"]

    # Crear figura y ejes
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Anotar los valores en las celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Etiquetas de ejes y título
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.tight_layout()

    file_name1 = f"RF_{AD}_CM_{n_trees}trees_{criterion}C_{max_depth}MD.png"

    # Define la ruta completa del archivo de salida
    output_file_path1 = os.path.join(Output_path, file_name1)

    plt.savefig(output_file_path1)

    return precision, recall, accuracy, f1_score


###################################################################################
###################################################################################
###################################################################################
def plot_roc_curve(X, y, n_trees, criterion, AD, test_size, max_depth, random_state, Output_path):
    """
    Plots the ROC curve for a Random Forest classification model.

    Inputs:

        X (DataFrame): Input DataFrame.
        y (array): Target vector (1s and 0s for regulation prediction).
        n_trees (int): Number of trees in the forest.
        criterion (str): Criterion for splitting trees.
        AD (str): Aerodrome.
        test_size (float): Test size percentage (0-1).
        random_state (int): Random seed for reproducibility.
        Output_path (str): Output path for saving the ROC curve plot.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = separar(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = separar(
        X_train, y_train, test_size=test_size, random_state=random_state
    )

    # Create and train the Random Forest classifier
    classifier = Bosque(
        n_estimators=n_trees, criterion=criterion, max_depth=max_depth, random_state=random_state
    )
    classifier.fit(X_train, y_train)

    # Make predictions on the validation set
    y_prob = classifier.predict_proba(X_val)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)

    # Calculate the AUC (Area Under the Curve)
    auc = roc_auc_score(y_val, y_prob)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Random Forest ({n_trees} trees, {criterion} criterion)")
    plt.legend(loc="lower right")

    # Save the ROC curve plot
    file_name = f"RF_{AD}_ROC_{n_trees}trees_{criterion}C_{max_depth}MD.png"
    output_file_path = os.path.join(Output_path, file_name)
    plt.savefig(output_file_path)


###################################################################################
###################################################################################
###################################################################################
def graphs_weather(input_file, AD):
    """
    This function creates graphs related to weather

    Inputs:

        1. List of regulation csv file
        2. 'AD' (str): Selected aerodrome

    Outputs:

        1. Graph of weather vs all kinds of regulations counts in current aerodrome
        2. Graph of weather vs all kinds of regulations counts in all aerodromes
        3. Graph of different kinds of weather in current aerodrome
        4. Graph of different kinds of weather in all aerodromes
    """

    all_reg_matrix = pd.read_csv(input_file)

    # Filter for weather-related data
    weather_matrix = all_reg_matrix[all_reg_matrix["Regulation Reason Name"] == "W - Weather"]
    ICAO_matrix = weather_matrix[weather_matrix["Protected Location Type"] == "Aerodrome"]
    airport_matrix = ICAO_matrix[ICAO_matrix["Protected Location Id"] == AD]

    # Create a copy of the matrix I want to send back
    modified_airport_matrix = airport_matrix.copy()

    # Extract specific weather types from a column (adjust the column name accordingly)
    all_type = all_reg_matrix["Regulation Reason Name"].tolist()
    weather_type = ICAO_matrix["Regulation Description"].str.lower().tolist()
    weather_airport_type = airport_matrix["Regulation Description"].str.lower().tolist()

    # Replace all columns that at least contain CB by only the word CB
    for column in range(len(weather_type)):
        if not pd.isna(weather_type[column]):
            weather_type[column] = (
                "CB"
                if any(
                    keyword in weather_type[column]
                    for keyword in ["cb", "rain", "show", "thunderstorm"]
                )
                else (
                    "VISIBILITY/CEILING"
                    if any(
                        keyword in weather_type[column]
                        for keyword in ["visibility", "ceiling", "vis", "lvp", "l.v.p."]
                    )
                    else (
                        "WIND"
                        if "wind" in weather_type[column]
                        else (
                            "FOG"
                            if "fog" in weather_type[column]
                            else (
                                "SNOW/ICE"
                                if any(
                                    keyword in weather_type[column]
                                    for keyword in ["snow", "freez", "icing", "ice"]
                                )
                                else "OTHER"
                            )
                        )
                    )
                )
            )
        else:
            weather_type[column] = "OTHER"

    counter_weather1 = Counter(weather_type)
    total1 = sum(counter_weather1.values())
    words1 = list(counter_weather1.keys())
    counts1 = list(counter_weather1.values())
    percentages1 = [count / total1 * 100 for count in counts1]

    ## FIGURE 1 ##
    # Create a bar plot
    plt.figure(figsize=(5, 4))
    plt.bar(words1, percentages1)  # Usamos porcentajes en lugar de recuentos
    plt.xlabel("")
    plt.ylabel("Percentage of counts")
    plt.title("Weather types in Eurocontrol area")
    # Rotamos las etiquetas del eje X para mayor legibilidad
    plt.xticks(rotation=60)
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()

    #####################################################################
    for column in range(len(weather_airport_type)):
        if not pd.isna(weather_airport_type[column]):
            weather_airport_type[column] = (
                "CB"
                if any(
                    keyword in weather_airport_type[column]
                    for keyword in ["cb", "rain", "show", "thunderstorm"]
                )
                else (
                    "VISIBILITY/CEILING"
                    if any(
                        keyword in weather_airport_type[column]
                        for keyword in ["visibility", "ceiling", "vis", "lvp", "l.v.p."]
                    )
                    else (
                        "WIND"
                        if "wind" in weather_airport_type[column]
                        else (
                            "FOG"
                            if "fog" in weather_airport_type[column]
                            else (
                                "SNOW/ICE"
                                if any(
                                    keyword in weather_airport_type[column]
                                    for keyword in ["snow", "freez", "icing", "ice"]
                                )
                                else "OTHER"
                            )
                        )
                    )
                )
            )
        else:
            weather_airport_type[column] = "OTHER"

    counter_weather2 = Counter(weather_airport_type)
    total2 = sum(counter_weather2.values())
    words2 = list(counter_weather2.keys())
    counts2 = list(counter_weather2.values())
    percentages2 = [count / total2 * 100 for count in counts2]

    ## FIGURE 2 ##
    # Create a bar plot
    plt.figure(figsize=(5, 4))
    plt.bar(words2, percentages2)  # Usamos porcentajes en lugar de recuentos
    plt.xlabel("")
    plt.ylabel("Percentage of counts")
    plt.title(f"Weather types in {AD}")
    # Rotamos las etiquetas del eje X para mayor legibilidad
    plt.xticks(rotation=60)
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()

    counter_weather3 = Counter(all_type)
    total3 = sum(counter_weather3.values())
    words3 = list(counter_weather3.keys())
    counts3 = list(counter_weather3.values())
    percentages3 = [count / total3 * 100 for count in counts3]

    ## FIGURE 3 ##
    # Create a bar plot
    plt.figure(figsize=(5, 4))
    plt.bar(words3, percentages3)  # Usamos porcentajes en lugar de recuentos
    plt.xlabel("")
    plt.ylabel("Percentage of counts")
    plt.title("Regulations types in Eurocontrol area")
    # Rotamos las etiquetas del eje X para mayor legibilidad
    plt.xticks(rotation=60)
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()

    ###############################################################
    # Replace the 'Regulation Description' column in the copy with the modified one
    modified_airport_matrix["Regulation Description"] = weather_airport_type

    ADs_type = ICAO_matrix["Protected Location Id"].str.lower().tolist()

    counter_ADs1 = Counter(ADs_type)
    words1 = list(counter_ADs1.keys())
    counts1 = list(counter_ADs1.values())

    # Organiza los resultados por counts1 de manera ascendente y limita a los 15 primeros
    sorted_counts1, sorted_words1 = zip(
        *sorted(zip(counts1, words1, strict=False), reverse=True), strict=False
    )
    sorted_counts1 = sorted_counts1[:15]
    sorted_words1 = sorted_words1[:15]

    ## FIGURE 4 ##
    # Create a bar plot
    plt.figure(figsize=(5, 4))
    # Usamos porcentajes en lugar de recuentos
    plt.bar(sorted_words1, sorted_counts1)
    plt.xlabel("")
    plt.ylabel("Regulation counts")
    plt.title("Regulation counts in aerodromes in Eurocontrol area (Top 15)")
    # Rotamos las etiquetas del eje X para mayor legibilidad
    plt.xticks(rotation=60)
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()

    ###########################################################

    # Calculate the sum of "ATFM delays" for each airport
    atfm_sums = ICAO_matrix.groupby("Protected Location Id")["ATFM Delay (min)"].sum().reset_index()

    atfm_sums_sorted = atfm_sums.sort_values(by="ATFM Delay (min)", ascending=False)

    # Toma solo los primeros 15 aeropuertos con mayor valor
    top_15_airports = atfm_sums_sorted.head(15)

    ## FIGURE 5 ##
    # Create a bar plot
    plt.figure(figsize=(5, 4))
    plt.bar(top_15_airports["Protected Location Id"], top_15_airports["ATFM Delay (min)"])
    plt.xlabel("Aerodrome")
    plt.ylabel("Total ATFM Delay (min)")
    plt.title("Top 15 Aerodromes by Total ATFM Delay in Eurocontrol Area")
    plt.xticks(rotation=60)
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()

    # Return the filtered dataframe
    return modified_airport_matrix


###################################################################################
###################################################################################
###################################################################################
def graphs_TAF(visibility_file, ceiling_file, AD):
    """
    This function is intended to plot the results of TAF predictions done by Nicolas

    Inputs:

        1. 'visibility file' (csv): Visibility predictions
        2. 'ceiling file' (csv): Ceiling predictions
        3. 'AD' (str): Aerodrome of study

    Outputs:

        1. Plot of predicted vs real visibility
        2. Plot of predicted vs real ceiling height
    """
    # Read both files and convert them to DataFrame
    vis_file = pd.read_csv(visibility_file, skiprows=range(1, 18360))
    ceil_file = pd.read_csv(ceiling_file, skiprows=range(1, 18360))

    # Visibility values and absolute error
    vis_test = vis_file["y_test"].tolist()
    # vis_pred = vis_file["y_pred"].tolist()
    vis_error = [abs(x) for x in vis_file["residuals"].tolist()]
    vis_perc = []
    for i in range(len(vis_test)):
        if vis_test[i] != 0:
            vis_perc.append(vis_error[i] / vis_test[i] * 100)
        else:
            vis_perc.append(0)

    # Ceiling values and absolute error
    ceil_test = ceil_file["y_test"].tolist()
    # ceil_pred = ceil_file["y_pred"].tolist()
    ceil_error = [abs(x) for x in ceil_file["residuals"].tolist()]
    ceil_perc = []
    for i in range(len(ceil_test)):
        ceil_perc.append(ceil_error[i] / ceil_test[i] * 100)

    # Create an array for x values
    x_values_vis = list(range(1, len(vis_test) + 1))
    x_values_ceil = list(range(1, len(ceil_test) + 1))

    ## FIGURE 1 ##
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_vis, vis_error)
    plt.xlabel("Counts")
    plt.ylabel("absolute TAF error")
    plt.title(f"Visibility in {AD}")
    # Rotamos las etiquetas del eje X para mayor legibilidad
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()

    ## FIGURE 2 ##
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_ceil, ceil_error)
    plt.xlabel("Counts")
    plt.ylabel("absolute TAF error")
    plt.title(f"Ceiling in {AD}")
    # Rotamos las etiquetas del eje X para mayor legibilidad
    plt.tight_layout()

    # Muestra el gráfico.
    plt.show()


###################################################################################
###################################################################################
###################################################################################
def minmaxscaler_preprocess(file_path, columns_of_interest, Output_file):
    # Lee el archivo CSV en un DataFrame
    data = pd.read_csv(file_path)

    # Selecciona las columnas de interés
    selected_data = data[columns_of_interest]

    # Inicializa el MinMaxScaler
    scaler = MinMaxScaler()

    # Ajusta el escalador a los datos y transforma las columnas seleccionadas
    scaled_data = scaler.fit_transform(selected_data)

    # Crea un nuevo DataFrame con los datos escalados
    scaled_df = pd.DataFrame(scaled_data, columns=columns_of_interest)

    # Encuentra el valor máximo y mínimo en cada columna escalada
    max_values = scaled_df.max()
    min_values = scaled_df.min()
    scaled_df.to_csv(Output_file)

    return max_values, min_values, scaled_df


###################################################################################
###################################################################################
###################################################################################
def labelbinarizer_WD_preprocess(file_path, input_column, output_columns, Output_file):
    """
    Function to pre-process using LabelBinarizer the Wind Direction variable
    """
    # Lee el archivo CSV en un DataFrame
    data = pd.read_csv(file_path)

    # Encuentra la categoría más frecuente entre los valores numéricos
    most_frequent_category = pd.to_numeric(data[input_column], errors="coerce").mode()[0]
    most_frequent_category = int(most_frequent_category)

    # Reemplaza las cadenas "VRB" y "No info" con el valor más común en forma numérica
    data[input_column].replace(
        {"VRB": most_frequent_category, "No info": most_frequent_category}, inplace=True
    )

    # Convierte la columna "Wind Direction" a valores enteros
    data[input_column] = data[input_column].astype(int)

    # Define los intervalos y las etiquetas
    intervals = [0, 90, 180, 270, 359]
    labels = [1, 2, 3, 4]

    # Crea una nueva columna en el DataFrame original con las etiquetas correspondientes
    data["Wind Direction"] = pd.cut(data[input_column], bins=intervals, labels=labels)

    # Realiza la codificación one-hot para obtener las columnas de salida
    binarized_df = pd.get_dummies(
        data["Wind Direction"], columns=["Wind Direction"], prefix=output_columns
    )

    # Reemplaza True por 1 y False por 0 en el DataFrame resultante
    binarized_df = binarized_df.replace({True: 1, False: 0})

    final_df = pd.DataFrame(binarized_df)
    final_df.to_csv(Output_file, index=False)

    return final_df


###################################################################################
###################################################################################
###################################################################################
def labelbinarizer_CC_preprocess(file_path, output_columns, Output_file):
    """
    Function to pre-process using LabelBinarizer the Ceiling Coverage variable
    """
    # Lee el archivo CSV en un DataFrame
    data = pd.read_csv(file_path)

    # Realiza la codificación one-hot para obtener las columnas de salida
    binarized_df = pd.get_dummies(
        data["Ceiling coverage"], columns=["Ceiling coverage"], prefix=output_columns
    )

    # Reemplaza True por 1 y False por 0 en el DataFrame resultante
    binarized_df = binarized_df.replace({True: 1, False: 0})

    final_df = pd.DataFrame(binarized_df)
    final_df.to_csv(Output_file, index=False)

    return final_df


###################################################################################
###################################################################################
###################################################################################
def LSTM_model_1(
    X,
    Y,
    epochs,
    batch_size,
    AD,
    test_size,
    random_state,
    optimizer,
    activation,
    loss,
    dropout_rate,
    units,
    Output_path,
):
    """
    This function is intended to create a LSTM model for computing a binary classifier
    in the ATFM weather-related regulations problem

    Inputs:

        1. 'X' (DataFrame): Input file with
        2. 'Y' (list): Regulation binary list
        3. 'epochs' (int): Number of training epochs.
        4. 'batch_size' (int): Batch size.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    # Crear el modelo LSTM
    model = Sequential()

    ## MODEL 2 ##
    # Capa LSTM (primera capa)
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))  # Capa de Dropout

    # Capa LSTM (segunda capa)
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout_rate))  # Capa de Dropout

    # Capa LSTM (tercera capa)
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))  # Capa de Dropout

    # Capa totalmente conectada (Dense) para la clasificación binaria
    model.add(Dense(units=1, activation=activation))

    # Compilar el modelo
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Entrenar el modelo
    model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test)
    )

    # Representar gráficamente el rendimiento del modelo de precisión
    # plt.figure(figsize=(5, 4))
    # plt.plot(history.history['accuracy'], label='Training')
    # plt.plot(history.history['val_accuracy'], label='Testing')
    # plt.title('Model accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.tight_layout()

    # # Define el nombre del archivo para la imagen de precisión
    # file_name_precision = f'LSTM_{AD}_precision_{epochs}E_{batch_size}B_{units}U.png'
    # # Define la ruta completa del archivo de salida
    # output_file_path_precision = os.path.join(Output_path, file_name_precision)
    # # Guarda la figura en la ubicación especificada
    # plt.savefig(output_file_path_precision)

    # Calcular y mostrar la matriz de confusión
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred > 0.5
    cm = CM(Y_test, Y_pred)

    # Definir etiquetas para las clases
    class_names = ["Inactive", "Active"]

    # Crear figura y ejes para la matriz de confusión
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Anotar los valores en las celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Etiquetas de ejes y título
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.tight_layout()

    # Define el nombre del archivo para la imagen de matriz de confusión
    file_name_cm = f"LSTM_{AD}_CM_{epochs}E_{batch_size}B_{units}U.png"
    # Define la ruta completa del archivo de salida para la matriz de confusión
    output_file_path_cm = os.path.join(Output_path, file_name_cm)
    # Guarda la figura en la ubicación especificada
    plt.savefig(output_file_path_cm)


###################################################################################
###################################################################################
###################################################################################
def LSTM_model_2(
    X, Y, epochs, batch_size, AD, test_size, random_state, optimizer, activation, loss, Output_path
):
    """
    This function is intended to create a LSTM model for computing a binary classifier
    in the ATFM weather-related regulations problem

    Inputs:

        1. 'X' (DataFrame): Input file with
        2. 'Y' (list): Regulation binary list
        3. 'epochs' (int): Number of training epochs
        4. 'batch_size' (int): Batch size for training
        5. 'AD' (str): Identifier for the airport
        6. 'time_init' (str): Start time
        7. 'time_end' (str): End time
        8. 'test_size' (float): Size of the test dataset
        9. 'random_state' (int): Random seed for reproducibility
        10. 'optimizer' (str): Optimizer for model training
        11. 'activation' (str): Activation function for the output layer
        12. 'loss' (str): Loss function for model training
        13. 'Output_path' (str): Path to save the output files
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    # Crear el modelo LSTM
    model = Sequential()

    # Add an LSTM layer with 128 units
    model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))

    # Add a Dense layer for the output
    model.add(Dense(1, activation=activation))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Entrenar el modelo
    history = model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test)
    )

    # Representar gráficamente el rendimiento del modelo
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()

    # Define the file name
    file_name = f"LSTM_{AD}_{epochs}E_{batch_size}B_{optimizer}O_{activation}A_{loss}L.png"

    # Define la ruta completa del archivo de salida
    output_file_path = os.path.join(Output_path, file_name)

    # Guarda la figura en la ubicación especificada
    plt.savefig(output_file_path)

    # Calcular y mostrar la matriz de confusión
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred > 0.5
    cm = CM(Y_test, Y_pred)

    return cm


###################################################################################
###################################################################################
###################################################################################
def RNN_model(
    X, Y, epochs, batch_size, AD, test_size, random_state, optimizer, activation, loss, Output_path
):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)
    X_train, X_val, Y_train, Y_val = separar(
        X_train, Y_train, test_size=test_size, random_state=random_state
    )

    # Crear el modelo RNN
    model = Sequential()
    model.add(SimpleRNN(epochs, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(SimpleRNN(epochs, return_sequences=True))
    model.add(SimpleRNN(epochs, return_sequences=True))
    model.add(SimpleRNN(epochs))
    model.add(Dense(1, activation=activation))

    # Compilar el modelo
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Entrenar el modelo
    _ = model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test)
    )

    # Representar gráficamente el rendimiento del modelo
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training')
    # plt.plot(history.history['val_loss'], label='Validation')
    # plt.title('Model loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'], label='Training')
    # plt.plot(history.history['val_accuracy'], label='Validation')
    # plt.title('Model loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Precision')
    # plt.legend()

    # plt.tight_layout()

    # Define the file name
    # file_name = f'\RNN\RNN_{AD}_{epochs}E_{batch_size}B_{optimizer}O_{activation}A_{loss}L.png'

    # # Define la ruta completa del archivo de salida
    # output_file_path = os.path.join(Output_path, file_name)

    # # Guarda la figura en la ubicación especificada
    # plt.savefig(output_file_path)

    # Calcular y mostrar la matriz de confusión
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred > 0.5
    cm = CM(Y_test, Y_pred)

    # Display performance metrics
    precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    f1_score = 2 / (1 / precision + 1 / recall)

    return precision, recall, accuracy, f1_score


###################################################################################
###################################################################################
###################################################################################
def FNN_model(X, Y, hidden_layer_sizes, activation, solver, max_iter, test_size, random_state):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    # Crear el modelo de Red Neuronal Feedforward
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
    )

    # Entrenar el modelo
    model.fit(X_train, Y_train)

    # Calcular y mostrar la matriz de confusión
    Y_pred = model.predict(X_test)
    cm = CM(Y_test, Y_pred)

    # Representar gráficamente el rendimiento del modelo
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # # Plotear la pérdida del modelo (personalizable según tus necesidades)
    # plt.plot([i for i in range(max_iter)], [model.loss_curve_[i]
    #          for i in range(max_iter)], label='Training Loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # # Calcular y plotear la precisión después del entrenamiento en cada época
    # # Precisión en la época 0
    # train_accuracy = [np.mean(Y_train == model.predict(X_train))]
    # test_accuracy = [np.mean(Y_test == Y_pred)]  # Precisión en la época 0
    # for epoch in range(1, max_iter):
    #     model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
    #     train_accuracy.append(np.mean(Y_train == model.predict(X_train)))
    #     Y_pred = model.predict(X_test)
    #     test_accuracy.append(np.mean(Y_test == Y_pred))
    # plt.plot([i for i in range(max_iter)],
    #          train_accuracy, label='Training Accuracy')
    # plt.plot([i for i in range(max_iter)],
    #          test_accuracy, label='Test Accuracy')
    # plt.title('Model Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.tight_layout()

    # # Define el nombre del archivo
    # file_name = f'FNN{hidden_layer_sizes}{activation}{solver}{max_iter}E{test_size}T{random_state}.png'

    # # Define la ruta completa del archivo de salida
    # output_file_path = os.path.join(Output_path, file_name)

    # # Guarda la figura en la ubicación especificada
    # plt.savefig(output_file_path)

    # Display performance metrics
    precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    f1_score = 2 / (1 / precision + 1 / recall)

    return precision, recall, accuracy, f1_score


###################################################################################
###################################################################################
###################################################################################
def CNN_RNN_model(
    X,
    Y,
    epochs,
    batch_size,
    AD,
    time_init,
    time_end,
    test_size,
    random_state,
    activationDense,
    activationConv1D,
    optimizer,
    loss,
    kernel_size,
    pool_size,
    Conv1D_layer,
    Dense_layer,
    Output_path,
):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    # Escalar las características para mejorar el rendimiento de la red RNN
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear el modelo CNN
    cnn_model = Sequential()
    cnn_model.add(
        Conv1D(
            Conv1D_layer,
            kernel_size=kernel_size,
            activation=activationConv1D,
            input_shape=(X_train.shape[1], 1),
        )
    )
    cnn_model.add(MaxPooling1D(pool_size=pool_size))
    cnn_model.add(Flatten())

    # Crear el modelo RNN
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(epochs, input_shape=(X_train.shape[1], 1), return_sequences=True))
    rnn_model.add(SimpleRNN(epochs, return_sequences=True))
    rnn_model.add(SimpleRNN(epochs))

    # Combinar los modelos CNN y RNN
    combined_model = Sequential([cnn_model, rnn_model])
    combined_model.add(Dense(Dense_layer, activation=activationDense))

    # Compilar el modelo combinado
    combined_model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Entrenar el modelo
    history = combined_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test)
    )

    # Representar gráficamente el rendimiento del modelo
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()

    # Define the file name
    file_name = f'CNN_RNN_{AD}_{time_init.strftime("%Y%m%d")}_{time_end.strftime("%Y%m%d")}_{epochs}epochs_{batch_size}batch.jpg'

    # Define la ruta completa del archivo de salida
    output_file_path = os.path.join(Output_path, file_name)

    # Guarda la figura en la ubicación especificada
    plt.savefig(output_file_path)

    # Calcular y mostrar la matriz de confusión
    Y_pred = combined_model.predict(X_test)
    Y_pred = Y_pred > 0.5
    cm = CM(Y_test, Y_pred)

    return cm


###################################################################################
###################################################################################
###################################################################################
def CNN_model(
    X,
    Y,
    epochs,
    batch_size,
    AD,
    time_init,
    time_end,
    test_size,
    random_state,
    activationDense,
    activationConv1D,
    optimizer,
    loss,
    kernel_size,
    pool_size,
    Conv1D_layer,
    Dense_layer,
    Output_path,
):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    # Escalar las características para mejorar el rendimiento de la CNN
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear el modelo CNN
    model = Sequential()
    model.add(
        Conv1D(
            Conv1D_layer,
            kernel_size=kernel_size,
            activation=activationConv1D,
            input_shape=(X_train.shape[1], 1),
        )
    )
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(Dense_layer, activation=activationDense))

    # Compilar el modelo
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Entrenar el modelo
    history = model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test)
    )

    # Representar gráficamente el rendimiento del modelo
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Define the file name
    file_name = f'CNN{AD}{time_init.strftime("%Y%m%d")}{time_end.strftime("%Y%m%d")}{epochs}E{batch_size}B{optimizer}O{activationDense}AD{activationConv1D}AC{loss}L{kernel_size}K{pool_size}P{Conv1D_layer}CL{Dense_layer}DL.png'

    # Define la ruta completa del archivo de salida
    output_file_path = os.path.join(Output_path, file_name)

    # Guarda la figura en la ubicación especificada
    plt.savefig(output_file_path)

    return model


###################################################################################
###################################################################################
###################################################################################
def CNN_LSTM_model(
    X,
    Y,
    epochs,
    batch_size,
    AD,
    time_init,
    time_end,
    test_size,
    random_state,
    activationDense,
    activationConv1D,
    optimizer,
    loss,
    kernel_size,
    pool_size,
    Conv1D_layer,
    Dense_layer,
    Output_path,
):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    # Escalar las características para mejorar el rendimiento de la CNN
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define el número de timesteps (puedes ajustar esto según tu problema)
    timesteps = 1

    # Reformatea los datos para que tengan la forma adecuada
    X_train_reshaped = X_train.reshape(X_train.shape[0], timesteps, X_train.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], timesteps, X_test.shape[1])

    # Crear el modelo CNN
    cnn_model = Sequential()
    cnn_model.add(
        Conv1D(
            Conv1D_layer,
            kernel_size=kernel_size,
            activation=activationConv1D,
            input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
        )
    )
    cnn_model.add(MaxPooling1D(pool_size=pool_size))
    cnn_model.add(Flatten())

    # Crear el modelo LSTM
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(
            epochs,
            input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
            return_sequences=True,
        )
    )
    lstm_model.add(LSTM(epochs, return_sequences=True))
    lstm_model.add(LSTM(epochs))

    # Combinar los modelos CNN y LSTM
    combined_model = Sequential([cnn_model, lstm_model])
    combined_model.add(Dense(Dense_layer, activation=activationDense))

    # Compilar el modelo combinado
    combined_model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Entrenar el modelo
    history = combined_model.fit(
        X_train_reshaped,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_reshaped, Y_test),
    )

    # Representar gráficamente el rendimiento del modelo
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Define el nombre del archivo
    file_name = f'CNN_LSTM_{AD}_{time_init.strftime("%Y_%m_%d")}_{time_end.strftime("%Y_%m_%d")}_{epochs}epochs_{batch_size}batch.jpg'

    # Define la ruta completa del archivo de salida
    output_file_path = os.path.join(Output_path, file_name)

    # Guarda la figura en la ubicación especificada
    plt.savefig(output_file_path)

    # Calcular y mostrar la matriz de confusión
    Y_pred = combined_model.predict(X_test_reshaped)
    Y_pred = Y_pred > 0.5
    cm = CM(Y_test, Y_pred)

    return cm


###################################################################################
###################################################################################
###################################################################################
def WaveNet_model(
    X, Y, epochs, batch_size, AD, test_size, random_state, optimizer, activation, loss, Output_path
):
    """
    This function is intended to create a WaveNet model for computing a binary classifier
    in the ATFM weather-related regulations problem

    Inputs:

        1. 'X' (DataFrame): Input file with
        2. 'Y' (list): Regulation binary list
        3. 'epochs' (int): Number of training epochs
        4. 'batch_size' (int): Batch size for training
        5. 'AD' (str): Identifier for the airport
        6. 'test_size' (float): Size of the test dataset
        7. 'random_state' (int): Random seed for reproducibility
        8. 'optimizer' (str): Optimizer for model training
        9. 'activation' (str): Activation function for the output layer
        10. 'loss' (str): Loss function for model training
        11. 'Output_path' (str): Path to save the output files
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = separar(X, Y, test_size=test_size, random_state=random_state)

    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)

    # Define the input layer
    input_layer = Input(shape=(X_train.shape[1], 1))

    # WaveNet architecture
    residual = Conv1D(filters=128, kernel_size=2, padding="causal")(input_layer)
    skip_connections = []
    for _ in range(5):  # Adjust the number of layers as needed
        dilated_conv = Conv1D(
            filters=128, kernel_size=2, padding="causal", activation="relu", dilation_rate=2
        )(residual)
        skip_connections.append(dilated_conv)
        residual = Add()([residual, dilated_conv])
    out = Add()(skip_connections)
    out = Activation("relu")(out)
    out = Conv1D(filters=1, kernel_size=1)(out)
    out = Activation(activation)(out)

    # Create the model
    model = Model(inputs=input_layer, outputs=out)
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Print a summary of the model architecture
    model.summary()

    # Entrenar el modelo
    history = model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test)
    )

    # Representar gráficamente el rendimiento del modelo
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()

    # Define the file name
    file_name = f"WaveNet_{AD}_{epochs}E_{batch_size}B_{optimizer}O_{activation}A_{loss}L.png"

    # Define la ruta completa del archivo de salida
    output_file_path = os.path.join(Output_path, file_name)

    # Guarda la figura en la ubicación especificada
    plt.savefig(output_file_path)

    # Calcular y mostrar la matriz de confusión
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred > 0.5
    cm = CM(Y_test, Y_pred)

    return cm
