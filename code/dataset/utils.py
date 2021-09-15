import torch
import datetime


def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)


def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key])
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def select_subset(old_data, new_data, keys, idx, max_len=None, shape_max=1):
    '''
        modifies new_data

        @param old_data target dict
        @param new_data source dict
        @param keys list of keys to transfer
        @param idx list of indices to select
        @param max_len (optional) select first max_len entries along dim 1
    '''

    for k in keys:
        new_data[k] = old_data[k][idx]
        # print("new data", new_data[k].shape, "old data", old_data[k].shape)
        if shape_max == 1:
            if max_len is not None and len(new_data[k].shape) > shape_max:
                new_data[k] = new_data[k][:,:max_len]
        # print("new2 data", new_data[k].shape)
    return new_data


def batch_to_cuda(data, cuda, exclude_keys=[]):
    '''
        Move all values (tensors) to cuda if specified
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data
