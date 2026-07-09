# Reporte de análisis

La aplicación clasifica patrones sintéticos 2D mediante una CNN mínima: `Conv2D -> MaxPooling2D -> Flatten -> Dense`.

El flujo probado incluye carga de CSV, entrenamiento, inferencia, persistencia, monitoreo y exportación de reportes. Los tests automatizados cubren escenarios correctos y escenarios de error para ambos epics.
