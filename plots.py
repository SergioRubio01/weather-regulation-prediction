from PIL import Image

# # Lee el archivo CSV
# AD = 'LFPG'
# df = pd.read_csv(f'.\Data\Regulations\Regulations binarizer_{AD}_30.csv')

# # Cuenta la cantidad de valores 0 y 1 en la columna "Regulation Binary"
# counts = df['Regulation Binary'].value_counts()

# # Crea un gráfico de pastel
# plt.figure(figsize=(6, 6))
# plt.pie(counts, labels=['0', '1'], autopct='%1.1f%%', startangle=140)
# plt.title(f'{AD} regulations')
# plt.axis('equal')  # Para que el gráfico de pastel sea un círculo
# plt.show()

# file1 = '.\Data\Regulations\Regulations binarizer_EGLL_30.csv'
# file2 = '.\Data\Regulations\Regulations binarizer_LFPG_30.csv'
# file3 = '.\Data\Regulations\Regulations binarizer_LOWW_30.csv'
# file4 = '.\Data\Regulations\Regulations binarizer_LSZH_30.csv'

# file_paths = [file1, file4]
# ADlist     = ['EGLL','LSZH']
# output_file = f'.\Data\Regulations\Regulation_binarizer_{len(ADlist)}_30.csv'

# Reg_binarizer_undersampling(file_paths, ADs=ADlist ,output_file=output_file)


# NAME = 'LFPG'

# Input_file = f'.\Data\Regulations\Regulations binarizer_{NAME}_30.csv'
# Output_file = f'.\Data\Regulations\Regulations binarizer_{NAME}_30_over.csv'

# Reg_binarizer_oversampling(Input_file, Output_file)
# # Cargar el archivo CSV en un DataFrame
# file_path = f".\Data\Regulations\Regulations binarizer_{NAME}_30_over.csv"
# df = pd.read_csv(file_path)

# # Nombre de la columna que contiene los valores 1
# column_name = "Regulation Binary"  # Reemplaza con el nombre real de tu columna

# # Filtrar los valores iguales a 1 en esa columna
# values_to_plot = df[column_name].tolist()
# values = Counter(values_to_plot)
# words1 = list(values.keys())
# counts1 = list(values.values())

# # Create a bar plot
# plt.figure(figsize=(5, 3))

# # Usamos una condición para asignar colores a las barras
# colors = ['blue' if label == 0 else 'orange' for label in words1]

# # Usamos porcentajes en lugar de recuentos
# plt.bar(words1, counts1, color=colors)
# plt.ylabel('Counts')
# plt.title(f'Active vs inactive regulations in {NAME}')

# # Establece las etiquetas del eje X con 'Inactive' en la posición 0 y 'Active' en la posición 1
# plt.xticks([0, 1], ['Inactive', 'Active'])

# plt.tight_layout()
# plt.show()
# AD = 'LSZH'
# x = pd.read_csv(os.path.join('.\Data\METAR', f'METAR_{AD}_filtered_final.csv'))
# X = x.iloc[:, 1:14].values

# # Compute target vector
# y = pd.read_csv(os.path.join('.\Data\Regulations', 'Regulation_binarizer_2_30.csv'))
# Y = y[f'{AD}'].values
# X_train, X_test, Y_train, Y_test = separar(
#     X, Y, test_size=0.2, random_state=42)


# X_train, X_val, Y_train, Y_val = separar(
#     X_train, Y_train, test_size=0.2, random_state=42)

# # Puedes contar los valores de la siguiente manera:
# unique_values, counts = np.unique(Y_val, return_counts=True)

# colors = ['blue' if label == 0 else 'orange' for label in unique_values]

# # Crear un histograma
# # Create a bar plot
# plt.figure(figsize=(5, 3))
# plt.bar(unique_values, counts, color=colors)

# # Personalizar el gráfico
# plt.title('Validation samples')
# plt.xlabel('')
# plt.ylabel('Counts')

# # Establece las etiquetas del eje X con 'Inactive' en la posición 0 y 'Active' en la posición 1
# plt.xticks([0, 1], ['Inactive', 'Active'])

# # Mostrar el gráfico
# plt.show()

# Crea una lista con las rutas de tus 9 imágenes

# Creating the feature importances plot
# AD = ['LSZH','EGLL']

# x = pd.read_csv(os.path.join('.\Data\METAR', f'METAR_{AD[0]}_filtered_final.csv'))
# X = x.iloc[:, 1:15].values
# print(X[1:5])

# # Compute target vector
# y = pd.read_csv(os.path.join('.\Data\Regulations', 'Regulation_binarizer_2_30.csv'))
# Y = y[f'{AD[0]}'].values
# n_estimators = [50, 100, 200]
# max_depth = 50
# for i in range(len(n_estimators)):
#     visualizer = FeatureImportances(RandomForestClassifier(n_estimators=n_estimators[i],max_depth=max_depth),
#                                 relative=True)

#     visualizer.fit(X, Y)

#     # Saving plot in PNG format
#     visualizer.show(outpath=f".\Output\RF\RF_{AD[0]}_ImpPlot_{n_estimators[i]}.png")

# visualizer = ValidationCurve(RandomForestClassifier(n_estimators=n_estimators[0]),
#                             param_name="max_depth", n_jobs=-1,
#                             param_range=np.arange(1, 10),
#                             cv=15, scoring="accuracy")

# classes = list(set(Y))
# visualizer = PCA(scale=True, projection=2,
#                  classes=classes)

# visualizer.fit(X, Y)

# # Saving plot in PNG format
# visualizer.show(outpath=f".\Output\RF\RF_{AD[0]}_PCA.png")

units = [50, 100, 200]
batch_size = [32, 64, 128]
epochs = [30, 60, 120]
image_paths = []
for i in range(len(units)):
    for j in range(len(epochs)):
        image_paths.append(rf".\Output\LSTM\LSTM_LSZH_CM_{epochs[j]}E_32B_{units[i]}U.png")
# Abre las imágenes y las almacena en una lista
images = [Image.open(path) for path in image_paths]

# Asegúrate de que todas las imágenes tengan el mismo tamaño
width, height = images[0].size
for img in images:
    if img.size != (width, height):
        raise ValueError("Todas las imágenes deben tener el mismo tamaño.")

# Crea una imagen en blanco con el tamaño de la matriz 3x3
combined_image = Image.new("RGB", (3 * width, 3 * height))

# Combina las imágenes en la matriz 3x3
for i in range(3):
    for j in range(3):
        index = i * 3 + j
        combined_image.paste(images[index], (j * width, i * height))

# Guarda la imagen combinada
combined_image.save(r".\Output\LSTM\LSTM_LSZH_CM_combined.png")

# Cierra las imágenes originales
for img in images:
    img.close()
