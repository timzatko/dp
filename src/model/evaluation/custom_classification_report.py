from sklearn.metrics import classification_report,  \
    f1_score, accuracy_score, recall_score


def custom_classification_report(class_names, y_true, y_pred):
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Custom print because of incorrect formatting of original function
    for key in clf_report:
        if isinstance(clf_report[key], dict):
            print(f'\033[1m{key}\033[0m')

            for metric in clf_report[key]:
                print(f'{metric}: {clf_report[key][metric]}')
        else:
            print(f'{key}: {clf_report[key]}')

        print('\n')

    print(f'\033[1mF1\033[0m')
    for average in ['micro', 'macro']:
        print(f'{average}: {f1_score(y_true, y_pred, average=average)}')

    print('\n')
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    print(f'accuracy_score: {accuracy_score(y_true, y_pred)}')
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html?highlight=recall_score
    print(f'sensitivity_score: {recall_score(y_true, y_pred, average="micro")}')
    print(f'specificity_score: {recall_score(y_true, y_pred, average="micro", labels=[0])}')

    print('\n')