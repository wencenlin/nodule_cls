import numpy as np
import copy
import time
import torch
import copy
import itertools


def get_all_search_space(min_len, max_len, channel_range):
    """
    get all configs of model

    :param min_len: min of the depth of model
    :param max_len: max of the depth of model
    :param channel_range: list, the range of channel
    :return: all search space
    """
    # channel_range=[4, 8, 16, 32, 64, 128], 3 <= L+M+N <= 9
    all_search_space = []
    # get all model config with max length
    max_array = get_search_space(max_len, channel_range)
    # [[4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 8],..., [64, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128]]
    max_array = np.array(max_array)
    for i in range(min_len, max_len+1):
        new_array = max_array[:, :i]  # i=3,[...[64  64 128] [ 64 128 128]...]. i=4,[...[ 64  64 128 128] [ 64 128 128 128]...]
        repeat_list = new_array.tolist()
        # remove repeated list from lists
        new_list = remove_repeated_element(repeat_list)
        for list in new_list:
            for first_split in range(1, i -1):
                for second_split in range(first_split + 1, i):
                    # split list
                    all_search_space.append(
                        [list[:first_split], list[first_split:second_split], list[second_split:]])  # [[4], [4], [4]] ... [[4], [4], [4, 4]]
    return all_search_space


def get_limited_search_space(min_len, max_len, channel_range):
    """
    get all limited configs of model,
    the depth of a stage between [model_depth//4,model_depth//2]

    :param min_len: min of the depth of model
    :param max_len: max of the depth of model
    :param channel_range: list, the range of channel
    :return: all search space
    """
    max_len = max_len+1
    all_search_space = []
    # get all model config with max length
    max_array = get_search_space(max_len, channel_range)
    max_array = np.array(max_array)
    for i in range(min_len, max_len):
        new_array = max_array[:, :i]
        repeat_list = new_array.tolist()
        # remove repeated list from lists
        new_list = remove_repeated_element(repeat_list)
        for list in new_list:
            # limit [model_depth//4,model_depth//2]
            for first_split in range(i // 4, i - i // 2 + 1):
                for second_split in range(first_split + i // 4, i - i // 4 + 1):
                    all_search_space.append(
                        [list[:first_split], list[first_split:second_split], list[second_split:]])
    return all_search_space


def get_search_space(max_len, channel_range, search_space=[], now=0):
    """
    Recursive.
    Get all configuration combinations

    :param max_len: max of the depth of model
    :param channel_range: list, the range of channel
    :param search_space: search space
    :param now: depth of model
    :return:
    """
    result = []
    if now == 0:
        for i in channel_range:
            result.append([i])
        # [[4], [8], [16], [32], [64], [128]]
    else:
        for i in search_space:
            larger_channel = get_larger_channel(channel_range, i[-1])  # channel_num: 4 ,result: [4, 8, 16, 32, 64, 128]
            for m in larger_channel:  # when larger_channel = [4, 8, 16, 32, 64, 128], i = [4]
                tmp = i.copy()      # [4].    | [4].
                tmp.append(m)       # [4,4].  | [4,8].
                result.append(tmp)  # [[4,4]].| [[4,4], [4,8]].
        # [[4, 4], [4, 8], [4, 16], [4, 32], [4, 64], [4, 128], ..., [32, 32], [32, 64], [32, 128], [64, 64], [64, 128], [128, 128]]
        # [[4, 4, 4], [4, 4, 8], ..., [4, 4, 128], [4, 8, 8], ..., [4, 8, 128], ..., [4, 128, 128], ..., [32, 32, 32], [32, 32, 64], [32, 32, 128], ..., [64, 128, 128], [128, 128, 128]]
    now = now + 1
    if now < max_len:
        return get_search_space(max_len, channel_range, search_space=result, now=now)
    else:
        return result


def get_larger_channel(channel_range, channel_num):
    """
    get channels which is larger than inputs

    :param channel_range: list,channel range
    :param channel_num: input channel
    :return: list,channels which is larger than inputs
    """
    # channel_num: 4 ,result: [4, 8, 16, 32, 64, 128]
    # channel_num: 8 ,result:    [8, 16, 32, 64, 128]
    result = filter(lambda x: x >= channel_num, channel_range)
    return list(result)


def get_smaller_channel(channel, channel_range):
    """
    get channels which is smaller than inputs

    :param channel:input channel
    :param channel_range:list,channel range
    :return:list,channels which is larger than inputs
    """

    return list(filter(lambda x: x < channel, channel_range))


def get_shallower_module(min_len, module_config, shallower_module=[]):
    """
    get module config which is shallower than module_config

    :param min_len: min depth of model
    :param module_config: input module config
    :param shallower_module:
    :return: list,module config which is shallower than module_config
    """
    new_module_config = []
    for config in module_config:  # module_config: [[[4,8,8],[8,16],[16,32,32,64]]]
        for m in range(len(config)):  # len(config): 3
            if type(config[m]) is not int:  # config[0]:[4,8,8]
                if len(config[m]) > 1:
                    for n in range(len(config[m])):
                        tmp = copy.deepcopy(config)
                        del tmp[m][n]
                        new_module_config.append(tmp)  # tmp:[[8,8],[8,16],[16,32,32,64]],m=0,n=0
    new_module_config = remove_repeated_element(new_module_config)
    shallower_module.extend(new_module_config)
    # sum(len(x) for x in a):get the count of all element
    if shallower_module != []:
        count = sum(len(x) for x in shallower_module[-1])  # [..., [[4,8,8],[8,16],[16,32,32]]],count:8
        if count > min_len:
            return get_shallower_module(min_len, new_module_config, shallower_module)
        else:
            return shallower_module
    else:
        return []


def remove_repeated_element(repeated_list):
    """
    Remove duplicate elements

    :param repeated_list: input list
    :return: List without duplicate elements
    """
    repeated_list.sort()  # [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 8]]
    new_list = [repeated_list[k] for k in range(len(repeated_list)) if
                k == 0 or repeated_list[k] != repeated_list[k - 1]]  # [[4, 4, 4], [4, 4, 8]]
    return new_list


def get_element_count(the_list):
    """
    get depth of model

    :param the_list: input model config
    :return: depth of model
    """
    count = sum(len(x) for x in the_list)
    return count


def flat_list(the_list):
    """
    flatten list

    :param the_list:
    :return: flatten list
    """
    return [item for sublist in the_list for item in sublist]


def get_narrower_module(channel_range, module_config):
    """
    get module config which is narrower than module_config

    :param channel_range: channel range
    :param module_config: input model config
    :return: list,module config which is narrower than module_config
    """
    len_list = []
    for i in module_config:       # module_config: [[4,8,8],[8,16],[16,32,32,64]]
        len_list.append(len(i))   # len_list: [3,2,4]
    count = get_element_count(module_config)  # count: 9
    config_list = get_search_space(count, channel_range)  # max of the depth of model: count
    # [[4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 8],..., [64, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128]]
    config_array = np.array(config_list)
    module_config_array = np.array(flat_list(module_config))  # [4 8 8 8 16 16 32 32 64]
    equal_module_config_array = config_array <= module_config_array  # true/false matrix
    equal_module_config_array = np.prod(equal_module_config_array, 1)  # 1/0 matrix
    index = np.where(equal_module_config_array == 1)  # config_array <= module_config_array (all element)
    narrower_config = config_array[index[0]]  # 收集符合上述性質的(channel皆小於module_config)
    narrower_config = narrower_config.tolist()
    result = []
    for i in narrower_config:
        result.append([i[:len_list[0]], i[len_list[0]:len_list[1] + len_list[0]], i[len_list[1] + len_list[0]:]])
    return remove_repeated_element(result)


def get_latency(module, input_size):
    """
    get the latency of module

    :param module:
    :param input_size:
    :return: latency
    """
    module_input = torch.randn(input_size)
    start = time.time()
    output = module(module_input)
    end = time.time()
    return end - start


def get_excellent_module(trained_module):
    """
    get model with less latency and higher acc

    :param trained_module: trained module list
    :return: excellent module
    """
    excellent_module = np.empty(shape=(0, 3))
    acc_and_lat = trained_module[:, 1:]
    for module in trained_module:
        tmp = copy.deepcopy(acc_and_lat)
        tmp[:, 0] = acc_and_lat[:, 0] <= module[1]  # higher acc
        tmp[:, 1] = acc_and_lat[:, 1] >= module[2]  # less latency
        tmp = np.sum(tmp, axis=1)
        if 0 not in tmp:  # 只要非less acc且higher latency
            excellent_module = np.append(excellent_module, [module], axis=0)
    return excellent_module



