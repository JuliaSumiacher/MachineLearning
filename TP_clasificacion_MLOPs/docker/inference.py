import joblib
import pandas as pd
from preprocesador import preprocesamiento

import logging
from sys import stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

model = joblib.load('model.pkl')

logger.info('> Modelo cargado')

input = pd.read_csv('/files/input.csv') 

logger.info('> Entrada cargada')

input = preprocesamiento(input)

logger.info('> Preprocesamiento de los datos listo')

output = model.predict(input)

logger.info('> Predicciones realizadas')

pd.DataFrame(output, columns=['RainTomorrow']).to_csv('/files/output.csv', index=False)

logger.info('> Salida guardada')