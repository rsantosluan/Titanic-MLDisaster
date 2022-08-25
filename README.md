# Análise dos dados do Titanic ML Disaster
![Titanic](https://anakin022.files.wordpress.com/2018/01/titanic_banner.jpg)
###
## Contextualização
- Na busca de aprimorar meus conhecimentos com o incrível livro "Mãos à obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow", me deparei em uma de suas páginas de exercício com a sugestão de encarar um desafio conhecido no Kaggle. Como desde os primeiros dias em que decidi mergulhar de cabeça no mundo dos dados, ouço falar do famoso Titanic - Machine Learning from Disaster, resolvir colocar em prática o uso de algumas ferramentas e técnicas, arregaçar as mangas e encarar essa competição.
###
## O dedsafio (descrição retirada da própria plataforma)
O naufrágio do Titanic é um dos naufrágios mais infames da história.
Em 15 de abril de 1912, durante sua viagem inaugural, o amplamente considerado “inafundável” RMS Titanic afundou após colidir com um iceberg. Infelizmente, não havia botes salva-vidas suficientes para todos a bordo, resultando na morte de 1.502 dos 2.224 passageiros e tripulantes.
Embora houvesse algum elemento de sorte envolvido na sobrevivência, parece que alguns grupos de pessoas eram mais propensos a sobreviver do que outros. Neste desafio, pedimos que você construa um modelo preditivo que responda à pergunta: “que tipo de pessoa tem mais probabilidade de sobreviver?” usando dados de passageiros (ou seja, nome, idade, sexo, classe socioeconômica, etc).
###
## Proposta de Solução
Aplicando técnicas, ferramentas e métodos que aprendi até aqui, sem 'espiar' nenhum código dos 'coleguinhas kagglers', utilizando a metodologia CRISP-DM desenvolvi um código de maneira limpa, modular e organizada que passou por todas as etapas que precedem o desenvolvimento de um modelo de maneira rápida e tão eficiente quanto se espera de um primeiro ciclo de desenvolvimento. Código este que está disponível neste ![LINK](https://github.com/rsantosluan/Titanic-MLDisaster/blob/master/notebooks/eda.ipynb).
###
## Resultados obtidos
Pós a etapa de exploração, criação de variáveis e afins, decidi testar os seguintes modelos:
- **Random Forest** 
<p> Mesmo obtendo os melhores resultados nos testes com o conjunto de treino, ele se mostrou 'complexo demais' me rendendo apenas a **11473ª posição** com um score de **0.74162**. </p>
- **Ridge Classifier** 
DDecidi utilizar este modelo, pois era abordado no capítulo, se mostrava eficiente frente a conjuntos onde a maioria das características se mostram úteis e confesso nunca ter aprofundado meus estudos nele até então. Este modelo me rendeu a **3005ª posição**  com um score de **0.77990**.
Após aplicar o que descobri com as análises de correlação que havia feito durante a EDA e o os resultados da ExtraTreesClassifier para definição de importância de features consegui refinar o ajuste dos hiperparâmetros, atingindo assim a **1938ª posição** com um score de **0.78468**.
###
## Considerações Finais
-Apesar de ser um conjunto de dados muito explorado, considero este um estudo extremamente valioso onde pude refinar técnicas e conhecer melhor parâmetros de um modelo que até então não havia despendido de muito tempo para explorar. Além de me colocar a prova comparando os resultados obtidos com desenvolvedores de todo mundo.
- Ainda não estou plenamente satisfeito com a posição ocupada nem com o score obtido e pretendo, em outro ciclo, testar outros modelos e técnicas para alcançar melhores resultados.

## Ferramentas utilizadas
#### Sklearn
- MinMaxScaler;
- Cross_validate;
- RandomizedSearchCV;
- RandomForestRegressor e Classifier;
- ExtraTreesClassifier;
- RidgeClassifier e RidgeClassifierCV;
- Precision_recall_curve;
- LogisticRegressionCV
- DecisionTreeClassifier

#### Graph plot
- Seaborn
- Matplotlib

#### Outras
- Boruta
- Pickle
- Xgboost    
- Get_dummies
