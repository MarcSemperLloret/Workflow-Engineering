# Paquete de codificacion ciega humana

Este directorio contiene la segunda referencia humana usada en el analisis de sensibilidad del manuscrito. La codificacion se presenta como una segunda codificacion ciega realizada con el mismo schema cerrado.

## Archivos principales

- `codificacion_ciega_humana_piloto_50.csv`: segunda codificacion humana para los 50 articulos del piloto.
- `codificacion_ciega_humana_holdout_20.csv`: segunda codificacion humana para los 20 articulos del holdout.
- `metricas_segunda_vs_gold_resumen.csv`: acuerdo global y acuerdo en campos objetivos frente a la referencia principal.
- `metricas_segunda_vs_gold_por_campo.csv`: acuerdo por campo.
- `metricas_segunda_vs_gold_kappa_categoricos.csv`: kappa por campo categorico.
- `modelos_vs_gold_y_segunda_codificacion.csv`: comparacion de workflows frente a la referencia principal y frente a la segunda codificacion humana.
- `modelos_vs_gold_y_segunda_por_campo.csv`: comparacion por campo.
- `discrepancias_restantes_segunda_vs_gold.csv`: discrepancias que permanecen entre referencias humanas.

## Material de apoyo

- `reglas_codificacion_humana.md`: reglas de codificacion y vocabulario permitido.
- `schema_codificacion_1.json`: schema cerrado usado en la codificacion.
- `referencia_articulos_piloto_50.csv`: titulos y URLs de los articulos piloto.
- `referencia_articulos_holdout_20.csv`: titulos y URLs de los articulos holdout.
- `articulos_piloto_50/`: PDFs de los 50 articulos del piloto.
- `articulos_holdout_20/`: PDFs de los 20 articulos del holdout.

## Uso en el manuscrito

Estos archivos sustentan el analisis de sensibilidad a una segunda referencia humana. Las metricas reportadas en el articulo son:

- acuerdo objetivo total entre referencias humanas: `0.904`;
- kappa macro categorico objetivo: `0.767`;
- piloto `w19`: `0.879` frente a la referencia principal y `0.868` frente a la segunda referencia humana;
- holdout `w23`: `0.758` frente a la referencia principal y `0.697` frente a la segunda referencia humana;
- holdout Qwen3 evidence-localized: `0.711` frente a la referencia principal y `0.717` frente a la segunda referencia humana.

