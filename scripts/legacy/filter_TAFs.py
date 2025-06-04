import pandas as pd
import pytaf
from functions import download_TAFs


def filter_eTAFs(Input_file, Visibility_Nicolas, Ceiling_Nicolas, Output_file):
    """
    Here TAFs are filtered to get the most relevant features

    Inputs:

        1. 'Input_file' (txt): Input file
    """
    Visibility_df = pd.read_csv(Visibility_Nicolas)
    visibility_list = Visibility_df.iloc[:, 4].to_list()
    Ceiling_df = pd.read_csv(Ceiling_Nicolas)
    ceiling_list = Ceiling_df.iloc[:, 4].to_list()
    max_wind_speed = 0  # Initialize maximum wind speed to 0
    max_wind_gust = 0  # Initialize maximum wind gust to 0
    wind_speed_list = []  # List to store wind speed
    wind_gust_list = []  # List to store wind gust
    visibility_list = []  # List to store visibility
    ceiling_list = []  # List to store ceiling

    with open(Input_file) as file:
        for row in file:
            taf = pytaf.TAF(row)
            # valid_from_date = taf._taf_header["valid_from_date"]
            # valid_from_hours = taf._taf_header["valid_from_hours"]
            # valid_till_date = taf._taf_header["valid_till_date"]
            # valid_till_hours = taf._taf_header["valid_till_hours"]
            for i in range(1, len(taf._weather_groups)):
                print(taf._weather_groups[i], "\n")

                if taf._weather_groups[i]["wind"] is not None:
                    wind_speed = int(taf._weather_groups[i]["wind"]["speed"])
                    wind_gust = int(taf._weather_groups[i]["wind"]["gust"])

                    # Update maximum wind speed if needed
                    if wind_speed > max_wind_speed:
                        max_wind_speed = wind_speed

                    # Update maximum wind gust if needed
                    if wind_gust > max_wind_gust:
                        max_wind_gust = wind_gust

                    wind_speed_list.append(max_wind_speed)
                    wind_gust_list.append(max_wind_gust)
                else:
                    wind_speed_list.append("No info")
                    wind_gust_list.append("No info")

            # Aquí puedes agregar código para obtener y guardar visibilidad y ceiling si están disponibles
            # Asumiré que puedes obtener estos valores de manera similar a cómo obtuviste el viento.

    # Crear un DataFrame con los datos
    df = pd.DataFrame(
        {
            "Release Date": valid_from_date,
            "Wind speed": wind_speed_list,
            "Gust speed": wind_gust_list,
            "Visibility": visibility_list,
            "Ceiling": ceiling_list,
        }
    )

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(Output_file, index=False)


dates = download_TAFs(
    Input_file=r".\Data\TAF\TAF_ejemplo.txt", output_file=r".\Data\TAF\TAF_ejemplo_formatted.txt"
)
valid_from_date = filter_eTAFs(
    Input_file=r".\Data\TAF\TAF_ejemplo_formatted.txt",
    Visibility_Nicolas=r".\Data\TAF\Visibility predictions TAF.csv",
    Ceiling_Nicolas=r".\Data\TAF\ceiling predictions TAF.csv",
    Output_file=r".\Data\TAF\TAF_output.csv",
)
