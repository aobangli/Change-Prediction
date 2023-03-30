from config.TrainConfig import num_of_dense_feature

# 由于不同模型的接受输入的格式可能有所区别，定义每种模型输入形式


def predict_by_dnn(x, model):
    return model(x)


def predict_by_widedeep(x, model):
    return model(x)


def predict_by_deepcross(x, model):
    return model(x)


def predict_by_tabtransformer(x, model):
    dense_input, sparse_inputs = x[:, :num_of_dense_feature], x[:, num_of_dense_feature:]
    outputs = model(sparse_inputs.long(), dense_input)
    return outputs


def predict_by_mmoe(x, model):
    return model(x)


def predict_by_ple(x, model):
    return model(x)


def predict_by_sharebottom(x, model):
    return model(x)

def predict_by_ple_add_part_expert(x, model):
    return model(x)


func_dict = {
    'dnn': predict_by_dnn,
    'widedeep': predict_by_widedeep,
    'deepcross': predict_by_deepcross,
    'tabtransformer': predict_by_tabtransformer,
    'mmoe': predict_by_mmoe,
    'ple': predict_by_ple,
    'sharebottom': predict_by_sharebottom,
    'ple_add_part_expert': predict_by_ple_add_part_expert
}
