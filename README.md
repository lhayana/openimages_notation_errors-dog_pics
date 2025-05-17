# Detecção de Erros de Rotulagem em Imagens de Cães do Open Images

## Introdução

Esse projeto foca na identificação de possíveis erros de rotulagem em uma base de 1000 imagens originalmente classificadas como "Cachorro", obtidas do Open Images V6 e salvas localmente na pasta `dogs_images`. Para auxiliar nessa detecção, foi utilizado um conjunto de dados de contraste composto por 1000 imagens de "Não-Cães" (uma mistura de Raposas, Ursos e Gatos, também do Open Images, mas neste projeto lidas de uma pasta local `similar_animals_images` onde estavam todas juntas).

O objetivo principal é utilizar técnicas de aprendizado de máquina para sinalizar imagens dentro do conjunto de "Cachorros" que podem ter sido rotuladas incorretamente.

## Metodologia

O fluxo de trabalho adotado neste projeto pode ser resumido nas seguintes etapas:

1.  **Preparação dos Dados:**
    * As imagens de cães são lidas da pasta local `dogs_images`.
    * As imagens de "não-cães" (uma mistura de raposas, ursos e gatos) são lidas da pasta local `similar_animals_images`.
    * Para o treinamento do modelo auxiliar, as imagens de cães recebem o rótulo "Dog" (0) e as outras o rótulo "NotDog" (1).

2.  **Extração de Features:**
    * Para cada imagem nos dois conjuntos, vetores de características (features) são extraídos utilizando o modelo Vision Transformer (ViT) pré-treinado `google/vit-base-patch16-224-in21k` do Hugging Face.
    * As features extraídas são salvas localmente usando `joblib` para otimizar reexecuções.

3.  **Modelo Out-of-Sample (OOS) e Geração de Probabilidades:**
    * Um modelo classificador (`LogisticRegression` do scikit-learn) é treinado sobre as features combinadas dos conjuntos "Dog" e "NotDog".
    * O treinamento é realizado usando validação cruzada (`cross_val_predict`) para gerar probabilidades Out-of-Sample (OOS) para cada imagem. Essas probabilidades OOS refletem a confiança do modelo em classificar uma imagem como "Dog" ou "NotDog" sem que o modelo tenha visto aquela imagem específica durante o seu treinamento naquela "rodada" da validação cruzada.

4.  **Análise de Erros de Rotulagem:**
    * **Tentativa com `cleanlab`:** A biblioteca `cleanlab` foi utilizada com diferentes parâmetros (`filter_by` e `return_indices_ranked_by`) para tentar identificar automaticamente erros de rótulo no conjunto de imagens de cães, com base nas probabilidades OOS.
    * **Inspeção Manual Guiada:** Uma análise visual foi realizada, ordenando as imagens da pasta `dogs_images` pela menor confiança P(Dog) atribuída pelo modelo OOS. As imagens com as menores confianças foram plotadas para inspeção.

## Resultados e Observações

* O modelo Out-of-Sample (OOS) foi treinado com sucesso para distinguir entre as classes "Dog" e "NotDog", gerando probabilidades para cada imagem do conjunto original de cães.
* Curiosamente, mesmo experimentando com diferentes parâmetros, a função `cleanlab.filter.find_label_issues` não identificou automaticamente nenhum erro de rótulo no conjunto de 1000 imagens de "Cachorro".
* No entanto, a inspeção manual das imagens com a menor confiança P(Dog) (conforme indicado pelo modelo OOS) revelou várias instâncias que visualmente não pareciam ser cães, incluindo exemplos claros de gatos. Por exemplo, uma imagem originalmente rotulada como "Dog" recebeu uma confiança P(Dog) de apenas 0.76% do modelo OOS, enquanto a confiança P(NotDog) foi de 99.24%.
* Isso sugere que, para este dataset e a configuração binária "Dog" vs "NotDog" (onde "NotDog" é uma classe mista), a abordagem de inspecionar visualmente as previsões de menor confiança do modelo OOS foi mais eficaz para encontrar erros de rotulagem do que os métodos automáticos do `cleanlab` com suas configurações padrão ou testadas. O `cleanlab` pode ser mais conservador ou seus algoritmos podem se beneficiar de uma distinção de classes mais granular no conjunto de "não-cães".

## Estrutura do Código

O script principal (`openimages_notation_errors_dog_pics.py`) está organizado nas seguintes seções:

1.  **Setup:** Instalação e importação das bibliotecas.
2.  **Baixando as imagens (Originalmente):** Código para baixar imagens do Open Images V6 usando `fiftyone` (esta parte é informativa no script final, pois o script foi adaptado para usar pastas locais).
3.  **Preparando datasets e extraindo as features das imagens:**
    * Define os caminhos para as pastas locais `dogs_images` e `similar_animals_images`.
    * Define parâmetros como o número máximo de amostras.
    * Carrega os caminhos dos arquivos de imagem.
    * Define e executa a função de extração de features usando o ViT.
4.  **Treinamento do modelo Out-of-Sample (OOS):**
    * Combina as features e rótulos de cães e não-cães.
    * Treina o `LogisticRegression` com `cross_val_predict` para obter probabilidades OOS.
5.  **Aplicando o `cleanlab` (Tentativa):**
    * Utiliza `cleanlab.filter.find_label_issues` para tentar encontrar erros automaticamente.
6.  **Visualizando imagens suspeitas:**
    * Plota as imagens da pasta `dogs_images` que receberam a menor probabilidade P(Dog) do modelo OOS.
7.  **Análise com `Datalab` (Outliers e Near-Duplicates - Tentativa):**
    * Utiliza o `Datalab` do `cleanlab` para procurar por outliers e imagens quase duplicadas apenas no conjunto de features dos cães.

## Como Executar

Este projeto foi inteiramente desenvolvido e testado primariamente no ambiente Google Colab.

## Possíveis Melhorias Futuras

* Separar as imagens em `similar_animals_images` em subpastas por espécie (Fox, Bear, Cat) e treinar o modelo OOS como um classificador multi-classe (Dog, Fox, Bear, Cat). Isso poderia fornecer probabilidades mais ricas para o `cleanlab` e potencialmente melhorar a detecção automática de erros.
* Experimentar modelos OOS mais complexos (ex: redes neurais pequenas customizadas) em vez do `LogisticRegression`.
* Ajustar os parâmetros internos do `cleanlab` ou usar diferentes módulos da biblioteca para uma análise mais fina, se a detecção automática for um requisito forte.
