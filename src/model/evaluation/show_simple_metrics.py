def show_simple_metrics(model, test_seq):
    baseline_results = model.evaluate(test_seq, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
        print(f'{name}: {value}')
    print()
