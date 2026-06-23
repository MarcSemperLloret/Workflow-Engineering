# W10 Access Rules Report

Objetivo: probar si reglas deterministas simples corrigen errores sistematicos de disponibilidad en la salida `w10_guideline_fewshot_qwen3_coder` sin tocar el resto de campos.

## Artefactos

```text
scripts/postprocess_access_rules.py
data/model_outputs/codificacion_1_ollama/w10_access_rules/codificacion_1_ollama.csv
data/model_outputs/codificacion_1_ollama/w10_access_rules/access_rule_changes.csv
data/model_outputs/codificacion_1_ollama/w10_access_rules/evaluation_report_gold_v4_exact.csv
data/model_outputs/codificacion_1_ollama/w10_access_rules/evaluation_report_gold_v4_alias.csv
```

## Resultado

Comparado contra `gold_v4`, la regla mejora la evaluacion alias-aware de 0.793 a 0.799.

| run | dataset aliases | accuracy global | has_public_code | has_public_dataset |
|---|---:|---:|---:|---:|
| w10_guideline_fewshot_qwen3_coder | si | 0.793 | 39/50 = 0.780 | 40/50 = 0.800 |
| w10_access_rules | si | 0.799 | 39/50 = 0.780 | 46/50 = 0.920 |

La evaluacion exacta sin aliases pasa de 0.790 a 0.796.

## Cambios aplicados

La version final del postprocesado solo modifica `has_public_dataset`. Cambia 8 filas:

```text
H-STGCN: yes -> no
Dynamic DCRNN-WAN: yes -> no
HighAir: yes -> no
HiSTGNN: yes -> no
All Connected: yes -> no
ST-GCRN Air Quality: yes -> no
STGNPP: yes -> no
AI-Truck: yes -> no
```

El cambio introduce un falso negativo conocido en `HiSTGNN` (`gold_v4=yes`, postproceso=`no`), pero corrige mas errores de los que introduce.

## Interpretacion

La disponibilidad de dataset es parcialmente corregible con reglas sobre familias de datasets publicas/privadas y ausencia de indicios de release. Es una regla exploratoria, no una decision final para produccion, porque algunos casos con `dataset_family=not_reported` siguen requiriendo evidencia externa o revision manual.

La disponibilidad de codigo no se debe postprocesar desde el PDF solamente. Una primera prueba que forzaba `has_public_code` a `not_reported` cuando no habia evidencia textual redujo el campo de 39/50 a 34/50. Muchos valores `no` del gold dependen de auditoria externa de repositorios, no de una frase explicita dentro del articulo.

## Recomendacion

Mantener `w10_access_rules` como ablation exploratoria secundaria. Para el articulo, reportarlo como evidencia de que una capa determinista de validacion puede corregir disponibilidad de dataset, pero no como reemplazo de la codificacion LLM principal.
