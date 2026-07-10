# Proyecto Final — Neural Network Application

Curso: **CS2023 - Programación III**
Proyecto: **Neural Network Application en C++23**
Caso de uso: **Clasificación de patrones sintéticos en rejillas 2D usando una CNN mínima**

## Integrantes

| Integrante   | Responsabilidad principal                                               |
| ------------ | ----------------------------------------------------------------------- |
| Luana Yolanda Meniz Cueva | Backend tensorial, `Shape`, `Tensor`, operaciones numéricas y Eigen     |
| Ximena Gamero Goyzueta | Capas base: `Dense`, `Flatten`, activaciones y pérdida                  |
| Luis Enrique Nieva Carbajal | `Conv2D`, `MaxPooling2D`, grafo computacional y backpropagation         |
| Enrique Zheng Chan | `Sequential`, optimizador, entrenamiento, persistencia y CMake          |
| Luis Leonardo Alca Zarate | Aplicación práctica, carga de dataset, monitoreo, reportes y evidencias |


---

## 1. Resumen del proyecto

Este proyecto implementa una biblioteca básica de redes neuronales en **C++23**, con una API pública inspirada en TensorFlow/Keras. El sistema permite crear tensores, definir una red secuencial, ejecutar inferencia, calcular pérdida, entrenar mediante backpropagation, actualizar parámetros con SGD, guardar/cargar modelos y aplicar la red a un problema práctico de clasificación.

El caso de uso elegido es la **clasificación de patrones sintéticos en imágenes pequeñas**. Cada muestra es una rejilla 2D de baja resolución, por ejemplo `4x4x1` o `6x6x1`, y su etiqueta se representa como un vector *one-hot*. La arquitectura aprovecha capas convolucionales para detectar patrones locales en las rejillas.

El proyecto cubre los dos epics solicitados:

### Epic 1 — Biblioteca numérica, red neuronal básica y grafo computacional

Se implementa:

* `Shape`
* `Tensor<T>`
* Operaciones numéricas: `matmul`, `conv2d`, operaciones elemento a elemento y `reshape`
* Capas: `Input`, `Dense`, `Conv2D`, `MaxPooling2D`, `Flatten`
* Activaciones: `Linear`, `Relu`, `Softmax`
* Pérdida: `CategoricalCrossentropy`
* Optimizador: `SGD`
* `Sequential`
* `predict`
* `evaluate`
* `fit`
* `backward`
* `last_gradients`
* `parameters`
* `save`
* `load_model`

### Epic 2 — Monitoreo, aplicación práctica y documentación

Se implementa:

* `TrainingMonitor`
* `MetricsLogger`
* `DatasetLoader`
* `PatternClassifier`
* `ReportExporter`
* Exportación de métricas en CSV
* Reportes en Markdown y CSV
* Persistencia del modelo entrenado
* Demo funcional end-to-end

---

## 2. Requisitos

Para compilar y ejecutar el proyecto se necesita:

* CMake `3.20` o superior
* Compilador compatible con C++23
* CLion, Visual Studio Code o terminal
* Eigen (incluido en el repositorio, es el backend numérico principal — no opcional)
  
### Sobre Eigen

El proyecto usa Eigen como backend numérico principal para el `Tensor` y las operaciones
matriciales (`matmul`). Eigen viene incluido directamente en el repositorio en:

third_party/eigen-3.4.0/

No requiere instalación aparte ni vcpkg: CMake apunta directo a esa carpeta.

---

## 3. Estructura del proyecto

```txt
utec_ai/
├── CMakeLists.txt
├── README.md
├── main.cpp
├── include/
│   └── utec/
│       ├── algebra/
│       │   ├── shape.h
│       │   ├── tensor_backend.h
│       │   └── tensor_ops.h
│       │
│       ├── nn/
│       │   ├── nn_interfaces.h
│       │   ├── nn_activation.h
│       │   ├── nn_dense.h
│       │   ├── nn_convolution.h
│       │   ├── nn_pooling.h
│       │   ├── nn_flatten.h
│       │   ├── nn_loss.h
│       │   ├── nn_optimizer.h
│       │   ├── nn_graph.h
│       │   └── neural_network.h
│       │
│       ├── io/
│       │   ├── dataset_loader.h
│       │   └── model_serializer.h
│       │
│       ├── monitoring/
│       │   ├── training_monitor.h
│       │   ├── metrics_logger.h
│       │   └── report_exporter.h
│       │
│       └── apps/
│           └── PatternClassifier.h
│
├── src/
│   └── placeholder.cpp
│
├── tests/
│   ├── epic1/
│   │   ├── feature1_eigen_backend_test.cpp
│   │   ├── feature2_numeric_ops_test.cpp
│   │   ├── feature3_forward_layers_test.cpp
│   │   └── feature4_computational_graph_test.cpp
│   │
│   └── epic2/
│       ├── feature1_monitoring_test.cpp
│       ├── feature2_cnn_application_test.cpp
│       ├── feature3_serialization_test.cpp
│       └── feature4_reporting_test.cpp
│
├── docs/
│   ├── REPORTE_ANALISIS.md
│   ├── reporte_feature4.md
│   └── reporte_feature4.csv
│
└── artifacts/
    ├── train_x.csv
    ├── train_y.csv
    └── demo_model.bin
```

---

## 4. Compilación y ejecución

### 4.1. Compilar desde terminal

Desde la raíz del proyecto:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```


> Nota: Si el proyecto se abre en CLion, el IDE crea automáticamente una carpeta de compilación como cmake-build-debug o cmake-build-release. En ese caso, los mismos comandos (ctest, ejecución del ejecutable, etc.) deben ejecutarse desde esa carpeta generada por CLion.

---

### 4.2. Ejecutar la demo principal

Desde la carpeta `build`:

```bash
./utec_ai_demo
```

En Windows puede ser:

```bash
utec_ai_demo.exe
```

Salida esperada aproximada:

```txt
UTEC AI demo OK
Loss antes: 1.18032
Loss despues: 1.14044
Epocas: 20
Prediccion restaurada shape: {1,3}
```

Los valores exactos pueden variar ligeramente, pero se espera que:

```txt
Loss despues < Loss antes
```

Esto confirma que el entrenamiento modifica los parámetros y reduce la pérdida.

---

## 5. Ejecución de pruebas automatizadas

El proyecto incluye pruebas automatizadas por feature para Epic 1 y Epic 2.

Desde la carpeta `build`:

```bash
ctest --output-on-failure
```

Salida esperada:

```txt
100% tests passed, 0 tests failed out of 8
```

También se puede ejecutar solo un grupo de pruebas:

```bash
ctest -R epic1 --output-on-failure
```

```bash
ctest -R epic2 --output-on-failure
```

---

## 6. Tests incluidos

### Epic 1

| Test                                    | Qué valida                                                                                                           |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `feature1_eigen_backend_test.cpp`       | `Shape`, `Tensor`, `zeros`, `ones`, `from_data`, `rank`, `shape`, `reshape`, acceso por índice y errores controlados |
| `feature2_numeric_ops_test.cpp`         | `matmul`, `conv2d`, `Strides`, `Padding`, validación de canales incompatibles y stride inválido                      |
| `feature3_forward_layers_test.cpp`      | `Sequential`, `Input`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Softmax`, `predict` y parámetros visibles      |
| `feature4_computational_graph_test.cpp` | `fit`, `evaluate`, `backward`, `last_gradients`, pérdida decreciente y actualización real de `weights` y `bias`      |

### Epic 2

| Test                                | Qué valida                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `feature1_monitoring_test.cpp`      | Registro de métricas por época, historial, exportación CSV y errores por `NaN` o época duplicada             |
| `feature2_cnn_application_test.cpp` | Carga de dataset desde CSV, entrenamiento con `PatternClassifier`, predicción y errores por dataset inválido |
| `feature3_serialization_test.cpp`   | Guardado/carga del modelo, restauración de predicciones, pesos, bias y metadata                              |
| `feature4_reporting_test.cpp`       | Exportación de reportes en Markdown y CSV, y validación de corridas inconsistentes                           |

---

## 7. Arquitectura general

El flujo general del proyecto es:

```txt
CSV / Datos sintéticos
        ↓
DatasetLoader
        ↓
Tensor<float>
        ↓
Sequential
        ↓
Conv2D → MaxPooling2D → Flatten → Dense
        ↓
Softmax
        ↓
CategoricalCrossentropy
        ↓
Backward propagation
        ↓
SGD
        ↓
Parámetros actualizados
        ↓
Métricas, reportes y modelo serializado
```

---

## 8. Módulos principales

### 8.1. `utec::tf::Shape`

Archivo:

```txt
include/utec/algebra/shape.h
```

Responsabilidades:

* Representar dimensiones dinámicas.
* Validar que las dimensiones sean positivas.
* Calcular `rank`.
* Calcular `total_size`.
* Permitir comparación entre shapes.
* Usarse como base para `Tensor<T>`.

Ejemplo:

```cpp
Shape s{2, 3};
auto r = s.rank();       // 2
auto n = s.total_size(); // 6
```

---

### 8.2. `utec::tf::Tensor<T>`

Archivo:

```txt
include/utec/algebra/tensor_backend.h
```

Responsabilidades:

* Almacenar datos en memoria contigua.
* Crear tensores con `zeros`, `ones` y `from_data`.
* Acceder a elementos con `operator()`.
* Acceder a elementos planos con `flat`.
* Validar índices fuera de rango.
* Cambiar forma con `reshape`.
* Ejecutar operaciones elemento a elemento.

Ejemplo:

```cpp
auto x = Tensor<float>::zeros(Shape{2, 3});
x(1, 2) = 7.0f;

auto y = Tensor<float>::ones(Shape{2, 3});
auto z = x + y;

auto reshaped = z.reshape(Shape{3, 2});
```

---

### 8.3. Operaciones numéricas

Archivo:

```txt
include/utec/algebra/tensor_ops.h
```

Operaciones implementadas:

* `matmul`
* `conv2d`
* `allclose`
* `Strides`
* `Padding`

La convolución usa:

* Activaciones en formato **NHWC**: `{batch, height, width, channels}`
* Kernels en formato **HWIO**: `{kernel_height, kernel_width, input_channels, output_channels}`

Ejemplo:

```cpp
auto out = ops::conv2d(input, kernel, ops::Strides{1, 1}, ops::Padding::Valid);
```

---

## 9. Red neuronal implementada

### 9.1. API pública estilo TensorFlow/Keras

Ejemplo de uso:

```cpp
using namespace utec::tf;

Sequential model;

model.add(layers::Input(Shape{6, 6, 1}));
model.add(layers::Conv2D(2, {3, 3}, Activation::Relu));
model.add(layers::MaxPooling2D({2, 2}));
model.add(layers::Flatten());
model.add(layers::Dense(2, Activation::Softmax));

model.compile(
    optimizers::SGD(0.05f),
    losses::CategoricalCrossentropy{}
);

auto history = model.fit(x, y, FitOptions{.epochs = 30, .batch_size = 2});
auto eval = model.evaluate(x, y);
auto pred = model.predict(x);
model.save("artifacts/model.bin");
```

---

### 9.2. Capas implementadas

| Capa           | Parámetros entrenables | Función                                               |
| -------------- | ---------------------: | ----------------------------------------------------- |
| `Input`        |                     No | Define la forma de entrada                            |
| `Dense`        |                     Sí | Calcula `xW + b` y aplica activación                  |
| `Conv2D`       |                     Sí | Extrae patrones locales con kernels convolucionales   |
| `MaxPooling2D` |                     No | Reduce dimensión espacial tomando máximos por ventana |
| `Flatten`      |                     No | Convierte tensores espaciales en vectores por muestra |

---

### 9.3. Activaciones implementadas

| Activación | Uso                       |
| ---------- | ------------------------- |
| `Linear`   | Salida sin transformación |
| `Relu`     | Capas intermedias         |
| `Softmax`  | Clasificación multiclase  |

---

### 9.4. Pérdida implementada

Se implementa:

```cpp
losses::CategoricalCrossentropy
```

Esta pérdida se usa para clasificación multiclase con etiquetas *one-hot*.

Durante backpropagation, cuando la salida final usa `Softmax` y la pérdida es `CategoricalCrossentropy`, se usa la simplificación:

```txt
dZ = y_pred - y_true
```

Esto permite un cálculo estable y adecuado para el alcance del proyecto.

---

### 9.5. Optimizador implementado

Se implementa:

```cpp
optimizers::SGD
```

Regla de actualización:

```txt
param = param - learning_rate * gradient
```

SGD mantiene como estado mínimo la tasa de aprendizaje.

---

## 10. Grafo computacional y backpropagation

El grafo computacional se modela como una red secuencial. Cada capa guarda la información necesaria del forward para poder ejecutar su backward.

### Flujo de entrenamiento

En `Sequential::fit` se realiza:

1. División del dataset en mini-batches.
2. Forward propagation.
3. Cálculo de pérdida.
4. Cálculo del gradiente inicial.
5. Backward propagation en orden inverso.
6. Cálculo de gradientes por capa.
7. Actualización de parámetros con SGD.
8. Registro de pérdida por época.

### Gradientes registrados

`last_gradients()` devuelve un mapa con nombres como:

```txt
conv2d_0/weights
conv2d_0/bias
dense_0/weights
dense_0/bias
```

`MaxPooling2D` y `Flatten` no aparecen en este mapa porque no tienen parámetros entrenables, pero sí participan en backward propagando el gradiente hacia la capa anterior.

---

## 11. Persistencia del modelo

El modelo puede guardarse y cargarse desde archivo binario.

### Guardar

```cpp
model.save("artifacts/demo_model.bin", SaveOptions{
    .metadata = {
        {"input_shape", "6x6x1"},
        {"num_classes", "3"}
    }
});
```

### Cargar

```cpp
auto restored = Sequential::load_model("artifacts/demo_model.bin");
auto pred = restored.predict(sample);
```

La persistencia guarda:

* Arquitectura
* Tipo de capas
* Configuración de capas
* Activaciones
* Tamaño de kernel
* Tamaño de pooling
* Strides
* Pesos
* Bias
* Metadata
* Learning rate del optimizador

Esto permite que el modelo restaurado conserve el comportamiento del modelo original.

---

## 12. Aplicación práctica: `PatternClassifier`

Archivo:

```txt
include/utec/apps/PatternClassifier.h
```

La aplicación entrena una CNN para clasificar patrones simples representados como rejillas 2D.

### Entrada

Los datos de entrada están en CSV. Cada fila representa una imagen aplanada.

Ejemplo para una imagen `4x4x1`:

```txt
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

### Etiquetas

Las etiquetas están en formato *one-hot*.

Ejemplo con 3 clases:

```txt
1,0,0
0,1,0
0,0,1
```

### Carga del dataset

```cpp
auto dataset = io::DatasetLoader::from_csv(
    "artifacts/train_x.csv",
    "artifacts/train_y.csv",
    Shape{4, 4, 1}
);
```

### Entrenamiento de la aplicación

```cpp
apps::PatternClassifier app;
app.load_dataset(dataset.inputs, dataset.targets);
app.build_default_model(Shape{4, 4, 1}, 3);

auto report = app.train(FitOptions{
    .epochs = 10,
    .batch_size = 4
});
```

---

## 13. Monitoreo y reportes

### 13.1. `TrainingMonitor`

Archivo:

```txt
include/utec/monitoring/training_monitor.h
```

Registra:

* Época
* Loss
* Accuracy
* Duración en milisegundos

Ejemplo:

```cpp
monitoring::TrainingMonitor monitor;

monitor.on_epoch_end(1, {
    .loss = 0.91f,
    .accuracy = 0.60f,
    .duration_ms = 12
});

auto csv = monitor.to_csv();
```

### 13.2. `MetricsLogger`

Archivo:

```txt
include/utec/monitoring/metrics_logger.h
```

Permite guardar métricas de entrenamiento en archivos analizables.

### 13.3. `ReportExporter`

Archivo:

```txt
include/utec/monitoring/report_exporter.h
```

Permite registrar corridas experimentales y exportarlas como:

```txt
docs/reporte_feature4.md
docs/reporte_feature4.csv
```

Ejemplo:

```cpp
monitoring::ReportExporter exporter;

exporter.add_run({
    .name = "cnn_baseline",
    .train_loss = {1.2f, 0.8f, 0.5f},
    .val_accuracy = {0.45f, 0.70f, 0.82f},
    .notes = "baseline con Conv2D + Dense"
});

exporter.write_markdown("docs/reporte_feature4.md");
exporter.write_csv("docs/reporte_feature4.csv");
```

---

## 14. Evidencias generadas

Al ejecutar la demo y los tests se generan evidencias como:

```txt
artifacts/demo_model.bin
artifacts/train_x.csv
artifacts/train_y.csv
docs/reporte_feature4.md
docs/reporte_feature4.csv
```

Estas evidencias demuestran:

* Dataset de entrada disponible.
* Entrenamiento ejecutado.
* Métricas exportadas.
* Reportes generados.
* Modelo serializado.
* Modelo restaurado con inferencia funcional.

---

## 15. Resultados esperados

En la demo principal se entrena un modelo CNN mínimo sobre patrones sintéticos.
La salida esperada debe mostrar que la pérdida disminuye:

```txt
Loss antes: 1.18032
Loss despues: 1.14044
```

El valor exacto puede variar, pero se espera:

```txt
Loss despues < Loss antes
```

Esto confirma que:

* `fit` ejecuta entrenamiento real.
* Backpropagation calcula gradientes.
* SGD actualiza parámetros.
* La red aprende sobre el problema pequeño.

Además, los tests deben finalizar con:

```txt
100% tests passed, 0 tests failed out of 8
```

---

## 16. Decisiones de diseño

### 16.1. API inspirada en TensorFlow/Keras

Se eligió una API parecida a TensorFlow/Keras para facilitar la lectura y el uso:

```cpp
Sequential model;
model.add(layers::Input(...));
model.add(layers::Conv2D(...));
model.add(layers::MaxPooling2D(...));
model.add(layers::Flatten());
model.add(layers::Dense(...));
model.compile(...);
model.fit(...);
model.predict(...);
```

Esto hace que el flujo sea familiar y claro.

---

### 16.2. Grafo secuencial

El proyecto usa un grafo computacional secuencial.
No se implementa un motor dinámico tipo PyTorch, porque para el alcance del curso basta con que:

* Cada capa guarde sus cachés.
* El forward avance capa por capa.
* El backward retroceda en orden inverso.
* Los gradientes se acumulen por parámetro.
* El optimizador actualice los parámetros entrenables.

---

### 16.3. Uso de NHWC y HWIO

Para imágenes se usa:

```txt
NHWC = {batch, height, width, channels}
```

Para kernels convolucionales se usa:

```txt
HWIO = {kernel_height, kernel_width, input_channels, output_channels}
```

Esta convención se mantiene en `Conv2D`, `MaxPooling2D` y los tests.

---

### 16.4. MaxPooling con índices del máximo

`MaxPooling2D` guarda la posición ganadora de cada ventana durante forward.
En backward, el gradiente vuelve únicamente a esa posición.

Esto es necesario porque `MaxPooling2D` no tiene pesos, pero sí debe participar correctamente en backpropagation.

---

### 16.5. Serialización completa de arquitectura

La persistencia guarda la configuración de capas y no solo los pesos.
Esto evita reconstrucciones forzadas al cargar el modelo.

Se guarda:

* Tipo de capa.
* Configuración.
* Activación.
* Pesos y bias cuando aplica.
* Metadata.
* Configuración de entrenamiento básica.

---

## 17. Limitaciones

El proyecto cumple el alcance académico solicitado, pero tiene limitaciones:

* La convolución implementada está pensada para datasets pequeños.
* No se implementan optimizadores avanzados como Adam o RMSProp.
* No se implementa `Dropout`, `BatchNorm` ni regularización.
* No se implementa entrenamiento distribuido.
* No se implementa soporte completo para padding diferente de `Valid`.
* El dataset usado en la demo es sintético y pequeño.
* La aplicación práctica busca validar el flujo completo, no competir con frameworks reales.

---

## 18. Posibles mejoras futuras

Como mejoras futuras se podrían implementar:

* Adam.
* RMSProp.
* Padding `Same`.
* Strides mayores a 1 en convolución.
* Métricas adicionales.
* Dataset más grande.
* Validación separada automática.
* Guardado del estado completo de optimizadores adaptativos.
* Visualización de curvas de entrenamiento.
* Soporte para más tipos de capas.

---

## 19. Cómo probar desde CLion

1. Abrir la carpeta raíz del proyecto en CLion.
2. Esperar a que CMake cargue.
3. Seleccionar el target:

```txt
utec_ai_demo
```

4. Ejecutar `Build`.
5. Ejecutar `Run`.
6. Abrir la terminal integrada de CLion.
7. Entrar a la carpeta de build:

```bash
cd cmake-build-debug
```

8. Ejecutar:

```bash
ctest --output-on-failure
```

Resultado esperado:

```txt
100% tests passed, 0 tests failed out of 8
```

---

## 20. Comandos rápidos

### Compilar

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Ejecutar demo

```bash
./utec_ai_demo
```

### Ejecutar todos los tests

```bash
ctest --output-on-failure
```

### Ejecutar solo Epic 1

```bash
ctest -R epic1 --output-on-failure
```

### Ejecutar solo Epic 2

```bash
ctest -R epic2 --output-on-failure
```

---

## 21. Fuentes consultadas

* Documentos de especificación del proyecto final de Programación III.
* Guías de implementación de Epic 1 Features 1, 2, 3 y 4.
* Guías de implementación de Epic 2 Features 1, 2, 3 y 4.
* Documentación de CMake.
* Documentación de Eigen.
* Conceptos base de redes neuronales: forward propagation, backpropagation, softmax, cross-entropy y SGD.

---

## 22. Conclusión

El proyecto implementa una red neuronal básica end-to-end en C++23.
La solución cubre desde el backend tensorial hasta una aplicación práctica con entrenamiento, inferencia, monitoreo, persistencia y reportes.

El sistema permite:

* Crear tensores.
* Definir una red secuencial.
* Ejecutar forward propagation.
* Calcular pérdida.
* Ejecutar backpropagation.
* Registrar gradientes.
* Actualizar parámetros con SGD.
* Guardar y cargar modelos.
* Leer datasets desde CSV.
* Exportar métricas y reportes.
* Verificar el comportamiento mediante pruebas automatizadas por feature.

La demo y los tests permiten comprobar que el proyecto compila, se ejecuta y cumple el flujo completo esperado para la entrega final.
