# Regressão Logistica
Melhor modelo encontrado: Pipeline(steps=[('scaler', StandardScaler()),
                ('logistic',
                 LogisticRegression(C=0.00011390176182186631,
                                    class_weight='balanced',
                                    l1_ratio=0.023062425041415757,
                                    max_iter=10000, multi_class='multinomial',
                                    penalty='elasticnet', solver='saga'))])
Média de F1-score: 0.32571084009306384
Média de Accuracy: 0.34416947040498436
Média de Precision: 0.3403062297391086
Média de Recall: 0.34416947040498436
Média de AUC-ROC: 0.6731525764307411


Média de Accuracy: 0.36020456333595596
Média de Precision: 0.36522404787452406
Média de Recall: 0.36020456333595596
Média de F1-score: 0.35874923563409916
Média de AUC-ROC: 0.6697657512716342
LogisticRegression(C=0.3593813663804626, class_weight='balanced',
                   l1_ratio=0.6456456456456456, max_iter=100000,
                   multi_class='multinomial', penalty='elasticnet',
                   solver='saga')

# Arvore de decisão
Média de Accuracy: 0.4179115264797508
Média de Precision: 0.43071443054171554
Média de Recall: 0.4179115264797508
Média de F1-score: 0.415651256061819
Média de AUC-ROC: 0.7525917121922342
DecisionTreeClassifier(criterion='entropy', max_depth=24, min_samples_leaf=33,
                       min_samples_split=19)

Média de Accuracy: 0.43153422501966954
Média de Precision: 0.45269102187875265
Média de Recall: 0.43153422501966954
Média de F1-score: 0.4382254466171043
Média de AUC-ROC: 0.7450086338653037
DecisionTreeClassifier(criterion='entropy', max_depth=42, min_samples_leaf=7,
                       min_samples_split=4, splitter='random')
# Random Forest

Média de Accuracy: 0.46343676012461055
Média de Precision: 0.47860431829275163
Média de Recall: 0.46343676012461055
Média de F1-score: 0.4589623089675978
Média de AUC-ROC: 0.782338394192603
RandomForestClassifier(class_weight='balanced_subsample', max_depth=19,
                       max_features='log2', min_samples_leaf=5,
                       min_samples_split=4, n_estimators=141)

Média de Accuracy: 0.4834461054287963
Média de Precision: 0.5001732296064322
Média de Recall: 0.4834461054287963
Média de F1-score: 0.4889119827767666
Média de AUC-ROC: 0.7840051586532718
RandomForestClassifier(class_weight='balanced', max_depth=36, max_features=None,
                       min_samples_leaf=3, min_samples_split=5,
                       n_estimators=465)


# Gradient Boosting

GradientBoostingClassifier(learning_rate=0.1817904187943561, max_depth=19,
                           max_features='sqrt', min_samples_leaf=8,
                           min_samples_split=16, n_estimators=609,
                           subsample=0.7400482498643719)


# K-means com cotovelo

Melhor n: 21 e melhor score: 0.1642404524648236

Melhor n: 7 e melhor score: 0.15558273755250907
