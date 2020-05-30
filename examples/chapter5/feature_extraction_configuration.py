from keras import optimizers

from core.network import ValidationStrategy

learning_rate_training = 2e-5
learning_rate_fine_tuning = 1e-5

classifier_configuration = [
    {'layer_type': 'Dense', 'units': 256, 'activation': 'relu', 'input_dim': 4 * 4 * 512},
    {'layer_type': 'Dropout', 'rate': 0.5},
    {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}]

training_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=learning_rate_training),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit': {
            'epochs': 30,
            'batch_size': 20}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}

training_parameters_generator = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=2e-5),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 30,
            'steps_per_epoch': 100,
            'validation_steps': 50}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}

fine_tuning_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=learning_rate_fine_tuning),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 100,
            'steps_per_epoch': 100,
            'validation_steps': 50}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}
