from examples.chapter5 import cats_and_dogs, mnist, feature_extraction, feature_extraction_data_augmentation, \
    visualize_activations, visualize_filters, visualize_heatmaps

if __name__ == '__main__':

    experiments = [
        # 'mnist',
        # 'cats_and_dogs',
        # 'feature_extraction',
        # 'feature_extraction_data_augmentation',
        # 'visualize_activations',
        # 'visualize_filters',
        'visualize_heatmaps'
    ]

    if 'mnist' in experiments:
        print('\n===> MNIST <===')
        mnist.run()

    if 'cats_and_dogs' in experiments:
        print('\n===> CATS & DOGS <===')
        cats_and_dogs.run()

    if 'feature_extraction' in experiments:
        print('\n===> FEATURE EXTRACTION <===')
        feature_extraction.run(build_dataset=False)

    if 'feature_extraction_data_augmentation' in experiments:
        print('\n===> FEATURE EXTRACTION WITH DATA AUGMENTATION <===')
        feature_extraction_data_augmentation.run(fine_tune=True)

    if 'visualize_activations' in experiments:
        print('\n===> VISUALIZE ACTIVATIONS <===')
        visualize_activations.run()

    if 'visualize_filters' in experiments:
        print('\n===> VISUALIZE FILTERS <===')
        visualize_filters.run()

    if 'visualize_heatmaps' in experiments:
        print('\n===> VISUALIZE HEATMAPS <===')
        visualize_heatmaps.run()
