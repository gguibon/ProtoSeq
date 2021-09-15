import code.train.regular as regular


def train(train_data, val_data, model, args, tasks_data=None, tasks_models=None, tasks_args=None, mtl_model=None ):
    return regular.train(train_data, val_data, model, args)


def test(test_data, model, args, verbose=True, target='val', tasks_data=None, tasks_models=None, tasks_args=None, mtl_model=None, num_episodes=None):
    return regular.test(test_data, model, args, verbose, target=target)
